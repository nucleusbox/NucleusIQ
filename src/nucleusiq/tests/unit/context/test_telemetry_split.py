"""Tests for F3 — masker vs compactor telemetry split.

Before F3 the single ``tokens_freed_total`` field conflated two very
different mechanisms:

* ``ObservationMasker`` (Tier 0, runs unconditionally after every LLM
  response).
* ``CompactionPipeline`` (Tier 1+, runs when utilisation crosses
  thresholds).

In production we observed runs with ``compaction_count = 5`` and
``tokens_freed_total = 800`` where the compactor itself freed *0*
tokens — the masker was doing all the work.  The combined metric hid
the threshold-mismatch bug (Finding 4) that motivated the audit.

These tests verify:

1. Masker-only runs record ``compactor_tokens_freed == 0``.
2. Compactor events contribute only to ``compactor_tokens_freed``.
3. ``tokens_freed_total == masker + compactor`` on any snapshot.
4. ``ContextTelemetry.merge`` additively rolls up both fields.
"""

from __future__ import annotations

from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.telemetry import CompactionEvent, ContextTelemetry


def _assistant_with_tool_call(tc_id: str, tool_name: str, arguments: str = "{}"):
    return ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCallRequest(id=tc_id, name=tool_name, arguments=arguments)],
    )


def _build_engine() -> ContextEngine:
    return ContextEngine(ContextConfig(), max_tokens=128_000)


# ------------------------------------------------------------------ #
# Core invariants                                                    #
# ------------------------------------------------------------------ #


def test_masker_only_run_reports_zero_compactor_freed():
    """Masker does work → masker_tokens_freed > 0, compactor_tokens_freed = 0."""
    engine = _build_engine()
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Read page 7."),
        _assistant_with_tool_call("tc-1", "reader"),
        ChatMessage(
            role="tool",
            content="PAGE 7 CONTENT " * 500,
            name="reader",
            tool_call_id="tc-1",
        ),
        ChatMessage(role="assistant", content="Summary."),
    ]
    engine.post_response(messages)  # trigger the masker

    tel = engine.telemetry
    assert tel.masker_tokens_freed > 0
    assert tel.compactor_tokens_freed == 0
    assert tel.tokens_freed_total == tel.masker_tokens_freed


def test_compactor_events_contribute_only_to_compactor_freed():
    """Injecting compaction events (simulating the pipeline) must bump
    ``compactor_tokens_freed`` and NOT the masker counter."""
    engine = _build_engine()
    engine._events.append(
        CompactionEvent(
            strategy="tool_result_compactor",
            trigger_utilization=0.6,
            tokens_before=1500,
            tokens_after=200,
            tokens_freed=1300,
            artifacts_offloaded=1,
        )
    )
    engine._events.append(
        CompactionEvent(
            strategy="conversation_compactor",
            trigger_utilization=0.75,
            tokens_before=2000,
            tokens_after=900,
            tokens_freed=1100,
        )
    )

    tel = engine.telemetry
    assert tel.compactor_tokens_freed == 2400
    assert tel.masker_tokens_freed == 0
    assert tel.tokens_freed_total == 2400
    assert tel.compaction_count == 2


def test_total_always_equals_sum_of_parts():
    """``tokens_freed_total`` is literally masker + compactor at any snapshot."""
    engine = _build_engine()
    messages = [
        ChatMessage(role="system", content="s"),
        _assistant_with_tool_call("tc-1", "r"),
        ChatMessage(
            role="tool",
            content="DATA " * 500,
            name="r",
            tool_call_id="tc-1",
        ),
        ChatMessage(role="assistant", content="done"),
    ]
    engine.post_response(messages)
    engine._events.append(
        CompactionEvent(
            strategy="tool_result_compactor",
            trigger_utilization=0.6,
            tokens_before=500,
            tokens_after=100,
            tokens_freed=400,
        )
    )

    tel = engine.telemetry
    assert tel.tokens_freed_total == tel.masker_tokens_freed + tel.compactor_tokens_freed
    assert tel.masker_tokens_freed > 0
    assert tel.compactor_tokens_freed == 400


# ------------------------------------------------------------------ #
# Merge rollup                                                       #
# ------------------------------------------------------------------ #


def test_merge_additively_rolls_up_split_fields():
    """Parallel sub-agents each report their own split; merge is additive."""
    parent = ContextTelemetry(masker_tokens_freed=100, compactor_tokens_freed=50)
    child_a = ContextTelemetry(masker_tokens_freed=200, compactor_tokens_freed=300)
    child_b = ContextTelemetry(masker_tokens_freed=0, compactor_tokens_freed=400)

    merged = ContextTelemetry.merge(parent, [child_a, child_b])

    assert merged.masker_tokens_freed == 300
    assert merged.compactor_tokens_freed == 750


def test_merge_without_children_is_identity():
    """With no children, merge returns the parent's snapshot as-is."""
    parent = ContextTelemetry(
        masker_tokens_freed=42, compactor_tokens_freed=7, tokens_freed_total=49
    )
    merged = ContextTelemetry.merge(parent, [])

    assert merged.masker_tokens_freed == 42
    assert merged.compactor_tokens_freed == 7
    assert merged.tokens_freed_total == 49


# ------------------------------------------------------------------ #
# Backward compatibility                                              #
# ------------------------------------------------------------------ #


def test_default_values_are_zero():
    """New fields default to 0 — no consumer is forced to read them."""
    tel = ContextTelemetry()
    assert tel.masker_tokens_freed == 0
    assert tel.compactor_tokens_freed == 0
    assert tel.tokens_freed_total == 0


def test_tokens_freed_total_remains_backward_compatible_field():
    """Old consumers reading ``tokens_freed_total`` still see the total."""
    engine = _build_engine()
    messages = [
        _assistant_with_tool_call("tc", "t"),
        ChatMessage(
            role="tool",
            content="BIG " * 600,
            name="t",
            tool_call_id="tc",
        ),
        ChatMessage(role="assistant", content="done"),
    ]
    engine.post_response(messages)

    tel = engine.telemetry
    assert hasattr(tel, "tokens_freed_total")
    assert tel.tokens_freed_total > 0
