"""Tests for ``extract_raw_trace`` — F2 rehydration helper.

Rationale
---------
``ObservationMasker`` runs inside ``ContextEngine.post_response`` and
mutates the single shared ``messages`` list by replacing consumed
tool-result content with a structured marker.  Downstream consumers
(``Critic._extract_reasoning_trace``, the Refiner's
``summarize_tool_results``) read that same list, so they used to see
only the marker — never the actual tool evidence.  ``extract_raw_trace``
re-hydrates those markers from ``ContentStore`` so inspectors see the
raw tool payload again (without mutating the live conversation).

The tests here lock in the properties that make the helper safe to
call in *any* context:

    * pure (no mutation of input list)
    * idempotent (non-masked messages pass through unchanged)
    * fail-open (missing keys / bad store / None store → no raise)
    * bounded (respects ``max_chars_per_result``)
"""

from __future__ import annotations

from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.store import ContentStore, extract_raw_trace
from nucleusiq.agents.context.strategies.observation_masker import (
    ObservationMasker,
    build_marker,
)


def _assistant_with_tool_call(tc_id: str, tool_name: str, arguments: str):
    return ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCallRequest(id=tc_id, name=tool_name, arguments=arguments)],
    )


def _mask_one_turn(
    payload: str, tool_name: str = "reader"
) -> tuple[list[ChatMessage], ContentStore]:
    """Build a 4-message conversation, run the masker, return (masked_messages, store).

    Keeps test setup DRY: each test gets a realistic masked trace with
    honest F1 marker slots (tool/args/ref/size) already in place.
    """
    store = ContentStore()
    counter = DefaultTokenCounter()
    masker = ObservationMasker()

    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Read page 7."),
        _assistant_with_tool_call("tc-1", tool_name, '{"page":7}'),
        ChatMessage(role="tool", content=payload, name=tool_name, tool_call_id="tc-1"),
        ChatMessage(role="assistant", content="Page 7 summary..."),
    ]
    masked, count, _freed = masker.mask(messages, counter, store)
    assert count == 1  # sanity: we really masked something
    return masked, store


# ------------------------------------------------------------------ #
# Core properties                                                    #
# ------------------------------------------------------------------ #


def test_roundtrip_restores_original_content():
    """The rehydrated tool message contains the exact original payload."""
    original = "The annual report excerpt: " + "DATA " * 200
    masked, store = _mask_one_turn(original)
    tool_msg = masked[3]
    assert tool_msg.content.startswith("[observation consumed")

    rehydrated = extract_raw_trace(masked, store)

    tool_msg_rehydrated = rehydrated[3]
    assert tool_msg_rehydrated.content == original
    # Non-tool messages pass through unchanged.
    assert rehydrated[0] is masked[0]
    assert rehydrated[2] is masked[2]
    assert rehydrated[4] is masked[4]


def test_is_pure_does_not_mutate_input():
    """Caller's ``messages`` list must be untouched."""
    original = "PAYLOAD " * 100
    masked, store = _mask_one_turn(original)
    snapshot_contents = [m.content for m in masked]

    extract_raw_trace(masked, store)

    assert [m.content for m in masked] == snapshot_contents


def test_missing_key_keeps_marker():
    """If the store evicted the key, rehydration leaves the marker in place."""
    original = "EVICT_ME " * 100
    masked, store = _mask_one_turn(original)

    # Evict the only stored entry
    for k in list(store.keys()):
        store.remove(k)

    rehydrated = extract_raw_trace(masked, store)
    assert rehydrated[3].content.startswith("[observation consumed")


def test_none_store_returns_input_unchanged():
    """``None`` store is accepted (e.g. unit tests without an engine)."""
    masked, _store = _mask_one_turn("ANY PAYLOAD " * 50)
    result = extract_raw_trace(masked, None)
    assert len(result) == len(masked)
    for a, b in zip(result, masked):
        assert a.content == b.content


def test_empty_messages_returns_empty_list():
    """Defensive: empty input → empty list."""
    assert extract_raw_trace([], ContentStore()) == []


def test_idempotent_on_already_rehydrated_messages():
    """Running ``extract_raw_trace`` twice produces the same result as once."""
    masked, store = _mask_one_turn("CONTENT " * 80)
    once = extract_raw_trace(masked, store)
    twice = extract_raw_trace(once, store)
    assert [m.content for m in twice] == [m.content for m in once]


def test_respects_max_chars_per_result():
    """Very large payloads are truncated so a single tool result can't
    blow up the Critic prompt."""
    huge = "X" * 100_000
    masked, store = _mask_one_turn(huge)

    rehydrated = extract_raw_trace(masked, store, max_chars_per_result=1024)
    body = rehydrated[3].content
    assert len(body) <= 1024 + len("\n... (truncated)") + 5
    assert body.endswith("(truncated)")


def test_non_tool_roles_never_touched():
    """``extract_raw_trace`` only acts on ``role='tool'`` messages."""
    store = ContentStore()
    store.store(key="obs:oddly:abcd", content="SOMETHING", original_tokens=100)

    marker = build_marker(
        tool_name="oddly",
        args_preview="{}",
        key="obs:oddly:abcd",
        tokens=100,
        summary="hi",
    )
    messages = [
        ChatMessage(role="user", content=marker),  # user shouldn't be rehydrated
        ChatMessage(role="assistant", content=marker),  # nor assistant
    ]
    result = extract_raw_trace(messages, store)
    assert result[0].content == marker
    assert result[1].content == marker


def test_tool_message_without_marker_passes_through():
    """A tool message that is NOT a marker (e.g. fresh tool output) is untouched."""
    messages = [
        ChatMessage(
            role="tool",
            content="fresh tool result",
            name="t",
            tool_call_id="tc-fresh",
        ),
    ]
    result = extract_raw_trace(messages, ContentStore())
    assert result[0].content == "fresh tool result"


def test_malformed_marker_without_ref_line_passes_through():
    """An observation-consumed marker missing a ``ref:`` line is a no-op."""
    malformed = "[observation consumed] (missing ref line)"
    messages = [
        ChatMessage(role="tool", content=malformed, name="t", tool_call_id="tc-m"),
    ]
    result = extract_raw_trace(messages, ContentStore())
    assert result[0].content == malformed


# ------------------------------------------------------------------ #
# Wiring: Critic._extract_reasoning_trace                             #
# ------------------------------------------------------------------ #


class TestCriticReasoningTraceWiring:
    """F2 — the Critic sees raw tool content when given a store.

    F1 intentionally embeds a ~200-char summary inside the marker so the
    Critic has *something* to work with even when no store is available.
    These tests therefore compare the *amount* of evidence visible
    (rehydrated = full payload; marker-only = bounded preview), not a
    binary presence/absence check.
    """

    def test_trace_is_marker_only_when_store_is_none(self):
        """Without a store, tool evidence is bounded to the marker preview.

        v0.7.10 — the marker now carries an inline preview of the
        first 1500 chars of the original payload (Step 4 — re-fetch
        loop fix).  Without a store the trace shows only that
        preview, not the full payload — still BOUNDED, just larger
        than the pre-v0.7.10 marker.  The 200-char payload here is
        3200 chars total, so the 1500-char preview leaks ~93 of the
        200 tokens (less than half).  The contract this test
        encodes is "without a store you see the preview, not the
        full payload" — the bound numbers update with the design.
        """
        from nucleusiq.agents.components.critic import Critic

        token = "SECRET_EVIDENCE "
        masked, _store = _mask_one_turn(token * 200)  # ~3200 chars payload
        trace = Critic._extract_reasoning_trace(masked)

        assert "[observation consumed]" in trace
        # Preview caps at 1500 chars → ~93 of the 200 tokens leak.
        # Must be strictly less than full (200) to prove preview is
        # bounded, not a passthrough.
        leaked = trace.count(token)
        assert 0 < leaked < 200, (
            f"Expected bounded preview (0 < count < 200), got {leaked}"
        )

    def test_trace_contains_full_raw_content_when_store_provided(self):
        """With the matching store, the Critic sees far more evidence than
        the marker's 200-char summary would allow."""
        from nucleusiq.agents.components.critic import Critic

        token = "SECRET_EVIDENCE "
        masked, store = _mask_one_turn(token * 200)
        trace = Critic._extract_reasoning_trace(masked, content_store=store)

        # Rehydrated trace must contain substantially more evidence than
        # the marker summary alone could carry.
        assert trace.count(token) >= 50


# ------------------------------------------------------------------ #
# Wiring: summarize_tool_results                                      #
# ------------------------------------------------------------------ #


class TestSummarizeToolResultsWiring:
    """F2 — the Refiner's summary uses rehydrated content when possible."""

    def test_summary_uses_marker_without_store(self):
        from nucleusiq.agents.modes.autonomous.helpers import summarize_tool_results

        token = "THE_REAL_EVIDENCE "
        masked, _store = _mask_one_turn(token * 200)
        summary = summarize_tool_results(masked)

        assert summary is not None
        assert "observation consumed" in summary
        # v0.7.10 — the marker now carries an inline preview slot of
        # the first 1500 chars (Step 4 — re-fetch loop fix).  Summary
        # callers without a store see that preview but never the full
        # payload — proving the preview is bounded.  Pre-v0.7.10 the
        # marker had no content; v0.7.10 deliberately adds bounded
        # content so the model can reason from the marker without
        # re-fetching.
        leaked = summary.count(token)
        assert 0 < leaked < 200, (
            f"Expected bounded preview (0 < count < 200), got {leaked}"
        )

    def test_summary_uses_real_content_with_store(self):
        from nucleusiq.agents.modes.autonomous.helpers import summarize_tool_results

        token = "THE_REAL_EVIDENCE "
        masked, store = _mask_one_turn(token * 200)
        summary = summarize_tool_results(masked, content_store=store)

        assert summary is not None
        # Rehydrated summary reads from the real 3600-char payload in
        # full (no head-truncation in v0.7.8).
        assert summary.count(token) == 200

    def test_no_head_truncation_of_rehydrated_content(self):
        """v0.7.8 regression guard: summarize_tool_results must NOT
        head-slice rehydrated content.  The pre-audit behaviour sliced
        the first 500 chars of each tool result — invariably
        boilerplate for structured payloads — and silently discarded
        the rest."""
        from nucleusiq.agents.modes.autonomous.helpers import summarize_tool_results

        # First 500 chars are boilerplate; the real signal is at
        # position 2000+.  A regression would keep only the boilerplate.
        boilerplate = "BOILERPLATE " * 42  # ~500 chars of noise
        signal = "THE_REAL_NUMBER_IS_42_CRORE"
        payload = boilerplate + ("X" * 1500) + signal + ("Y" * 1000)
        masked, store = _mask_one_turn(payload)

        summary = summarize_tool_results(masked, content_store=store)
        assert summary is not None
        assert signal in summary, (
            "v0.7.8 regression: summarize_tool_results must pass "
            "rehydrated content through unchanged — not slice the "
            "first N chars (which are boilerplate)."
        )
