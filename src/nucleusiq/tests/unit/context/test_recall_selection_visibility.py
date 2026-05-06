"""L1 recall selection and synthesis rehydration visibility tests."""

from __future__ import annotations

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.store import ContentStore


def _marker(key: str, *, tool_name: str = "read_report", tokens: int = 1000) -> str:
    return (
        "[observation consumed]\n"
        f"tool: {tool_name}\n"
        "args: {}\n"
        f"ref: {key}\n"
        f"size: ~{tokens} tokens\n"
        f'To retrieve: call recall_tool_result(ref="{key}")'
    )


def _engine(*, max_tokens: int = 100_000, response_reserve: int = 1_000):
    return ContextEngine(
        ContextConfig(
            max_context_tokens=max_tokens,
            response_reserve=response_reserve,
            tool_result_per_call_max_chars=50_000,
        )
    )


def test_content_store_records_recall_metadata() -> None:
    store = ContentStore()

    ref = store.store(
        key="obs:read_report:abc123",
        content="FULL REPORT CONTENT",
        original_tokens=123,
        preview_max_chars=8,
        tool_name="read_report",
    )
    meta = store.metadata(ref.key)

    assert meta is not None
    assert meta.key == "obs:read_report:abc123"
    assert meta.original_tokens == 123
    assert meta.preview == "FULL REP\n... (11 chars remaining)"
    assert meta.tool_name == "read_report"
    assert meta.order == 1


def test_prepare_for_synthesis_reports_selected_and_skipped_refs() -> None:
    engine = _engine()
    engine.store.store(
        "ref-old",
        "OLD_EVIDENCE_BODY",
        original_tokens=10,
        tool_name="read_report",
    )
    engine.store.store(
        "ref-new",
        "NEW_EVIDENCE_BODY",
        original_tokens=10,
        tool_name="read_report",
    )
    messages = [
        ChatMessage(role="user", content="task"),
        ChatMessage(
            role="tool",
            name="read_report",
            tool_call_id="ref-old",
            content=_marker("ref-old"),
        ),
        ChatMessage(
            role="tool",
            name="read_report",
            tool_call_id="ref-missing",
            content=_marker("ref-missing"),
        ),
        ChatMessage(
            role="tool",
            name="read_report",
            tool_call_id="ref-new",
            content=_marker("ref-new"),
        ),
    ]

    out = engine.prepare_for_synthesis(messages)
    tel = engine.telemetry

    assert [m.content for m in out if m.role == "tool"] == [
        "OLD_EVIDENCE_BODY",
        _marker("ref-missing"),
        "NEW_EVIDENCE_BODY",
    ]
    assert tel.synthesis_refs_selected == ("ref-new", "ref-old")
    assert tel.synthesis_refs_skipped == ("ref-missing",)
    assert tel.synthesis_rehydrated_count == 2
    assert tel.synthesis_rehydrated_tokens > 0


def test_prepare_for_synthesis_records_budget_skips() -> None:
    engine = _engine(max_tokens=8_000, response_reserve=1_000)
    big = "x" * 40_000
    engine.store.store("ref-too-big", big, original_tokens=10_000)
    messages = [
        ChatMessage(role="user", content="task"),
        ChatMessage(
            role="tool",
            name="read_report",
            tool_call_id="ref-too-big",
            content=_marker("ref-too-big"),
        ),
    ]

    out = engine.prepare_for_synthesis(messages)
    tel = engine.telemetry

    assert out[1].content == _marker("ref-too-big")
    assert tel.synthesis_refs_selected == ()
    assert tel.synthesis_refs_skipped == ("ref-too-big",)
    assert tel.synthesis_rehydrated_count == 0
