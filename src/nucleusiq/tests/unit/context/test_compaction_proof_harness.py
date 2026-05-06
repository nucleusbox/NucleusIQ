"""L0 compaction proof harness.

These tests are intentionally offline and deterministic.  They prove the
lowest context-management layer does its main job before we add workspace,
evidence dossier, document search, or autonomous phase gates:

* reduce active prompt tokens under pressure,
* reduce repeated input-token cost,
* preserve recallability of removed content,
* expose auditable telemetry.
"""

from __future__ import annotations

import re

import pytest
from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.context.config import ContextConfig, ContextStrategy
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.strategies.observation_masker import MASK_PREFIX

_REF_RE = re.compile(r"^ref:\s*(\S+)\s*$", re.MULTILINE)


def _engine(
    *,
    squeeze_threshold: float = 0.0,
    max_tokens: int = 16_000,
    response_reserve: int = 1_000,
) -> ContextEngine:
    config = ContextConfig(
        strategy=ContextStrategy.PROGRESSIVE,
        enable_observation_masking=True,
        squeeze_threshold=squeeze_threshold,
        max_context_tokens=max_tokens,
        response_reserve=response_reserve,
        cost_per_million_input=1.0,
    )
    return ContextEngine(config, DefaultTokenCounter(), max_tokens=max_tokens)


def _assistant_tool_call(
    tool_call_id: str = "call-1",
    *,
    name: str = "read_report",
    arguments: str = '{"company":"TCS","section":"revenue"}',
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCallRequest(id=tool_call_id, name=name, arguments=arguments),
        ],
    )


def _tool_result(
    content: str,
    *,
    tool_call_id: str = "call-1",
    name: str = "read_report",
) -> ChatMessage:
    return ChatMessage(
        role="tool",
        name=name,
        tool_call_id=tool_call_id,
        content=content,
    )


def _task_e_like_trace(tool_result: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="user",
            content=(
                "Compare TCS, Infosys, Wipro, and HCLTech across revenue, "
                "margin, AI strategy, risks, and recommendations."
            ),
        ),
        _assistant_tool_call(),
        _tool_result(tool_result),
        ChatMessage(
            role="assistant",
            content="I found TCS revenue and margin evidence in the report.",
        ),
    ]


def _only_tool_content(messages: list[ChatMessage]) -> str:
    content = next(m.content for m in messages if m.role == "tool")
    assert isinstance(content, str)
    return content


def _marker_ref(marker: str) -> str:
    match = _REF_RE.search(marker)
    assert match is not None, f"marker did not contain a ref line:\n{marker}"
    return match.group(1)


@pytest.mark.asyncio
async def test_prepare_leaves_small_prompt_unchanged_below_pressure() -> None:
    """Below the trigger threshold, prepare must not compact the prompt."""
    engine = _engine(squeeze_threshold=0.70)
    messages = [
        ChatMessage(role="user", content="short task"),
        ChatMessage(role="assistant", content="short answer"),
    ]

    prepared = await engine.prepare(messages)

    assert prepared == messages
    tel = engine.telemetry
    assert tel.compaction_count == 0
    assert tel.tokens_freed_total == 0
    assert tel.tokens_before_mgmt == tel.tokens_after_mgmt


def test_post_response_masks_consumed_large_tool_result_and_preserves_raw() -> None:
    """Masking should shrink active prompt while preserving exact raw content."""
    engine = _engine()
    raw = "TCS FY2024 revenue evidence. " + ("raw-financial-evidence " * 5_000)
    messages = _task_e_like_trace(raw)
    before_tokens = engine.token_counter.count_messages(messages)

    compacted = engine.post_response(messages)

    after_tokens = engine.token_counter.count_messages(compacted)
    marker = _only_tool_content(compacted)
    ref = _marker_ref(marker)
    tel = engine.telemetry

    assert after_tokens < before_tokens
    assert marker.startswith(MASK_PREFIX)
    assert raw not in marker
    assert engine.store.retrieve(ref) == raw
    assert engine.store.size == 1
    assert tel.observations_masked == 1
    assert tel.masker_tokens_freed > 0
    assert (
        tel.tokens_freed_total == tel.masker_tokens_freed + tel.compactor_tokens_freed
    )
    assert tel.tokens_before_mgmt >= before_tokens
    assert tel.tokens_after_mgmt <= after_tokens


def test_compaction_reduces_repeated_input_token_cost_on_followup_turn() -> None:
    """Later turns should resend marker text, not the full raw tool output."""
    engine = _engine()
    raw = "Infosys annual report evidence. " + ("ai-strategy-evidence " * 4_000)
    messages = _task_e_like_trace(raw)

    compacted = engine.post_response(messages)
    raw_followup = messages + [
        ChatMessage(role="user", content="Now compare this with Wipro.")
    ]
    compacted_followup = compacted + [
        ChatMessage(role="user", content="Now compare this with Wipro.")
    ]

    raw_followup_tokens = engine.token_counter.count_messages(raw_followup)
    compacted_followup_tokens = engine.token_counter.count_messages(compacted_followup)

    assert compacted_followup_tokens < raw_followup_tokens * 0.25


def test_marker_is_bounded_for_huge_tool_output() -> None:
    """Markers must not become a second prompt-sized evidence store."""
    engine = _engine()
    raw = "HCLTech risk evidence. " + ("risk-factor " * 25_000)

    compacted = engine.post_response(_task_e_like_trace(raw))
    marker = _only_tool_content(compacted)

    assert len(marker) < 1_000
    assert "tool: read_report" in marker
    assert "args:" in marker
    assert "ref:" in marker
    assert "size: ~" in marker
    assert raw[-1_000:] not in marker


def test_prepare_for_synthesis_rehydrates_selected_masked_evidence() -> None:
    """Synthesis can recover evidence without keeping raw output in every turn."""
    engine = _engine(max_tokens=64_000, response_reserve=1_000)
    raw = "Wipro margin evidence. " + ("margin-detail " * 2_000)
    compacted = engine.post_response(_task_e_like_trace(raw))

    synthesis_messages = compacted + [
        ChatMessage(role="user", content="Synthesize the competitive landscape.")
    ]
    synthesis_ready = engine.prepare_for_synthesis(synthesis_messages)
    tool_content = _only_tool_content(synthesis_ready)

    assert tool_content == raw
    assert engine.token_counter.count_messages(synthesis_ready) < 64_000
