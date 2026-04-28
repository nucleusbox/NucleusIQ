"""Tests for ObservationMasker — Tier 0 post-response strategy."""

import json

import pytest
from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.context.strategies.observation_masker import (
    MASK_PREFIX,
    ObservationMasker,
    build_marker,
)


@pytest.fixture
def masker():
    return ObservationMasker()


@pytest.fixture
def counter():
    return DefaultTokenCounter()


@pytest.fixture
def store():
    return ContentStore()


def _msg(
    role: str, content: str, *, name: str | None = None, tool_call_id: str | None = None
):
    return ChatMessage(role=role, content=content, name=name, tool_call_id=tool_call_id)


class TestObservationMasker:
    def test_masks_consumed_tool_result(self, masker, counter, store):
        """Tool result before assistant message should be masked."""
        messages = [
            _msg("system", "You are a helpful assistant."),
            _msg("user", "Search for AI trends."),
            _msg("tool", "A" * 500, name="web_search", tool_call_id="tc1"),
            _msg("assistant", "Based on the search, here are the trends..."),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 1
        assert freed > 0
        assert "[observation consumed" in result[2].content
        assert store.size == 1

    def test_does_not_mask_unconsumed_tool_result(self, masker, counter, store):
        """Tool result after the last assistant message should NOT be masked."""
        messages = [
            _msg("system", "You are helpful."),
            _msg("assistant", "I'll search for that."),
            _msg("tool", "B" * 500, name="web_search", tool_call_id="tc2"),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 0
        assert freed == 0
        assert result[2].content == "B" * 500

    def test_does_not_mask_without_assistant(self, masker, counter, store):
        """No assistant message means nothing is consumed."""
        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "Hello"),
            _msg("tool", "C" * 500, name="calculator", tool_call_id="tc3"),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 0
        assert freed == 0

    def test_skips_already_masked_results(self, masker, counter, store):
        """Already masked tool results should not be re-masked."""
        messages = [
            _msg("system", "You are helpful."),
            _msg(
                "tool",
                "[observation consumed — 100 tokens offloaded | ref: old_key]",
                name="tool1",
                tool_call_id="tc4",
            ),
            _msg("assistant", "Done."),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 0
        assert freed == 0

    def test_skips_already_offloaded_results(self, masker, counter, store):
        """Already offloaded tool results (context_ref) should not be re-masked."""
        messages = [
            _msg("system", "You are helpful."),
            _msg(
                "tool",
                "[context_ref: some_key]\nPreview content...",
                name="tool1",
                tool_call_id="tc5",
            ),
            _msg("assistant", "Done."),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 0
        assert freed == 0

    def test_skips_tiny_tool_results(self, masker, counter, store):
        """Very short tool results (<20 tokens) should not be masked."""
        messages = [
            _msg("system", "You are helpful."),
            _msg("tool", "OK", name="ping", tool_call_id="tc6"),
            _msg("assistant", "Ping succeeded."),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 0

    def test_masks_multiple_consumed_results(self, masker, counter, store):
        """Multiple tool results before assistant should all be masked."""
        messages = [
            _msg("system", "You are helpful."),
            _msg("tool", "D" * 300, name="search", tool_call_id="tc7"),
            _msg("tool", "E" * 400, name="calculator", tool_call_id="tc8"),
            _msg("assistant", "Here's what I found..."),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 2
        assert freed > 0
        assert store.size == 2

    def test_preserves_non_tool_messages(self, masker, counter, store):
        """User and system messages should never be masked."""
        messages = [
            _msg("system", "F" * 500),
            _msg("user", "G" * 500),
            _msg("tool", "H" * 500, name="tool1", tool_call_id="tc9"),
            _msg("assistant", "Result based on tool output."),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 1
        assert result[0].content == "F" * 500
        assert result[1].content == "G" * 500

    def test_does_not_mutate_input(self, masker, counter, store):
        """Input list should not be modified."""
        messages = [
            _msg("tool", "I" * 500, name="tool1", tool_call_id="tc10"),
            _msg("assistant", "Done."),
        ]
        original = list(messages)
        masker.mask(messages, counter, store)

        assert len(messages) == len(original)
        assert messages[0].content == original[0].content

    def test_content_preserved_in_store(self, masker, counter, store):
        """Full content should be retrievable from store after masking."""
        original_content = "J" * 1000
        messages = [
            _msg("tool", original_content, name="big_tool", tool_call_id="tc11"),
            _msg("assistant", "Processed."),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 1
        keys = store.keys()
        assert len(keys) == 1
        retrieved = store.retrieve(keys[0])
        assert retrieved == original_content

    def test_partial_masking_with_interleaved_messages(self, masker, counter, store):
        """Only tool results before last assistant are masked in multi-turn."""
        messages = [
            _msg("system", "You are helpful."),
            _msg("tool", "K" * 300, name="t1", tool_call_id="tc12"),
            _msg("assistant", "First response."),
            _msg("tool", "L" * 300, name="t2", tool_call_id="tc13"),
            _msg("assistant", "Second response."),
            _msg("tool", "M" * 300, name="t3", tool_call_id="tc14"),
        ]
        result, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count == 2
        assert "[observation consumed" in result[1].content
        assert "[observation consumed" in result[3].content
        assert result[5].content == "M" * 300


# ------------------------------------------------------------------ #
# F1 — Structured marker shape                                        #
# ------------------------------------------------------------------ #


def _assistant_with_tool_call(
    text: str,
    *,
    tc_id: str,
    tool_name: str,
    arguments: str | dict,
):
    """Build an assistant message that owns a single tool_call.

    Uses the framework's canonical ``ToolCallRequest`` flat shape so
    pydantic validation passes.  The masker's runtime arg-resolution
    logic handles multiple wire shapes (flat, nested, SDK objects) —
    those are covered by tests that inject dicts directly via the
    helper ``_dict_tool_call`` below.
    """
    if isinstance(arguments, dict):
        args_str = json.dumps(arguments, separators=(",", ":"))
    else:
        args_str = str(arguments)
    return ChatMessage(
        role="assistant",
        content=text or None,
        tool_calls=[ToolCallRequest(id=tc_id, name=tool_name, arguments=args_str)],
    )


class TestMarkerStructuredSlots:
    """F1 (v0.7.8) — Marker carries tool, args, ref, size only.

    The ``summary`` slot was removed in v0.7.8 because its head-
    truncated first-200-chars were boilerplate for structured
    payloads and encouraged the Generator to hallucinate from the
    teaser instead of rehydrating the real content.
    """

    def test_marker_contains_four_honest_slots(self, masker, counter, store):
        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "Compute 2+2"),
            _assistant_with_tool_call(
                "",
                tc_id="tc-calc-1",
                tool_name="calculator",
                arguments='{"expression":"2+2"}',
            ),
            _msg(
                "tool",
                "The result is 4. " * 80,
                name="calculator",
                tool_call_id="tc-calc-1",
            ),
            _msg("assistant", "Answer: 4."),
        ]
        result, masked_count, _freed = masker.mask(messages, counter, store)

        assert masked_count == 1
        marker = result[3].content
        lines = marker.split("\n")
        assert lines[0] == "[observation consumed]"
        assert lines[1].startswith("tool: ")
        assert lines[2].startswith("args: ")
        assert lines[3].startswith("ref: ")
        assert lines[4].startswith("size: ")
        # Context Mgmt v2 — Step 2: marker carries an explicit recall
        # hint so the model learns the tool from the marker itself
        # (belt-and-suspenders with the auto-injected tool spec).
        assert lines[5].startswith("To retrieve: call recall_tool_result(")
        assert "calculator" in lines[1]
        assert "2+2" in lines[2]  # args preview looked up from upstream tool_call
        assert "obs:calculator:" in lines[3]  # store key
        assert "tokens" in lines[4]
        # The recall hint quotes the same ref that lines[3] carries —
        # cheap data integrity check.
        ref_value = lines[3].split("ref:", 1)[1].strip()
        assert f'ref="{ref_value}"' in lines[5]
        # v0.7.8 regression guard: the marker MUST NOT contain a summary
        # slot because it head-truncates to first-N chars which is
        # boilerplate for structured documents (PDF/JSON/MDX headers).
        assert "summary:" not in marker
        assert len(lines) == 6  # 4 fact slots + prefix + recall hint

    def test_marker_args_preview_falls_back_when_upstream_missing(
        self, masker, counter, store
    ):
        """If no matching assistant tool_call is found, args is '(unavailable)'."""
        messages = [
            _msg("system", "You are helpful."),
            _msg(
                "tool",
                "X" * 500,
                name="mystery_tool",
                tool_call_id="no-upstream-call",
            ),
            _msg("assistant", "Done."),
        ]
        result, masked_count, _freed = masker.mask(messages, counter, store)

        assert masked_count == 1
        marker = result[1].content
        assert "args: (unavailable)" in marker
        assert "tool: mystery_tool" in marker

    def test_marker_args_handles_dict_arguments(self, masker, counter, store):
        """Providers that send dict arguments (not JSON string) still work."""
        messages = [
            _assistant_with_tool_call(
                "",
                tc_id="tc-dict-1",
                tool_name="reader",
                arguments={"path": "/tmp/report.pdf", "pages": [1, 2, 3]},
            ),
            _msg(
                "tool",
                "CONTENT " * 100,
                name="reader",
                tool_call_id="tc-dict-1",
            ),
            _msg("assistant", "Read three pages."),
        ]
        result, masked_count, _freed = masker.mask(messages, counter, store)

        assert masked_count == 1
        marker = result[1].content
        assert "/tmp/report.pdf" in marker
        assert "pages" in marker

    def test_marker_size_is_bounded_for_huge_payloads(self, masker, counter, store):
        """Marker length is O(metadata + preview cap), independent of payload size.

        v0.7.10: the marker now optionally carries a small inline
        preview of the payload (Step 4 — re-fetch loop fix).  The size
        bound is intentionally tight so markers do not become a second
        prompt-sized evidence store.  The important property — that
        marker size is BOUNDED, not proportional to payload — is
        preserved: a 500K-char payload
        and a 5K-char payload produce markers of the same upper-
        bound size.
        """
        huge = "\n".join([f"line {i}: " + "Z" * 100 for i in range(50)])
        messages = [
            _assistant_with_tool_call(
                "",
                tc_id="tc-huge",
                tool_name="huge",
                arguments="{}",
            ),
            _msg("tool", huge, name="huge", tool_call_id="tc-huge"),
            _msg("assistant", "Done."),
        ]
        result, _masked_count, _freed = masker.mask(messages, counter, store)

        marker = result[1].content
        # Bounded by preview cap (300) + chrome (~700) → < 1100.
        # The original 5K-char payload produces the SAME bound as a
        # 500K-char payload — this is the invariant.
        assert len(marker) < 1100, (
            f"Marker grew unexpectedly: {len(marker)} chars (cap ~1100)"
        )
        assert "summary:" not in marker

    def test_marker_args_preview_truncates_long_arguments(self, masker, counter, store):
        """Large argument blobs are truncated to keep the marker compact."""
        large_args = "{" + ",".join(f'"k{i}":"v{i}"' for i in range(500)) + "}"
        messages = [
            _assistant_with_tool_call(
                "",
                tc_id="tc-large",
                tool_name="bulk",
                arguments=large_args,
            ),
            _msg("tool", "R" * 800, name="bulk", tool_call_id="tc-large"),
            _msg("assistant", "OK."),
        ]
        result, _masked_count, _freed = masker.mask(messages, counter, store)

        marker = result[1].content
        args_line = [ln for ln in marker.split("\n") if ln.startswith("args: ")][0]
        args_body = args_line[len("args: ") :]
        assert len(args_body) <= 210  # truncated + ellipsis

    def test_marker_idempotent_on_second_pass(self, masker, counter, store):
        """Running mask twice does not re-mask the already-masked message."""
        messages = [
            _assistant_with_tool_call(
                "",
                tc_id="tc-idem",
                tool_name="t",
                arguments="{}",
            ),
            _msg("tool", "Y" * 400, name="t", tool_call_id="tc-idem"),
            _msg("assistant", "First."),
        ]
        first_result, first_count, _ = masker.mask(messages, counter, store)
        assert first_count == 1

        # Add another assistant turn so the already-masked message is now
        # still consumed but must not be touched again.
        second_pass_messages = list(first_result) + [
            _msg("assistant", "Second."),
        ]
        _r2, second_count, _ = masker.mask(second_pass_messages, counter, store)
        assert second_count == 0

    def test_build_marker_public_helper_matches_mask_output(
        self, masker, counter, store
    ):
        """``build_marker`` is the single source of truth for marker layout."""
        messages = [
            _assistant_with_tool_call(
                "",
                tc_id="tc-bm",
                tool_name="sample",
                arguments='{"q":"hello"}',
            ),
            _msg("tool", "The answer " * 50, name="sample", tool_call_id="tc-bm"),
            _msg("assistant", "Got it."),
        ]
        result, _count, _freed = masker.mask(messages, counter, store)
        produced = result[1].content

        expected = build_marker(
            tool_name="sample",
            args_preview='{"q":"hello"}',
            key=[k for k in store if k.startswith("obs:sample:")][0],
            tokens=counter.count("The answer " * 50),
        )
        # v0.7.8 — four honest slots, byte-for-byte identical.
        assert produced == expected

    def test_mask_prefix_constant_matches_marker(self, masker, counter, store):
        """MASK_PREFIX is the exported constant used for idempotency checks."""
        assert MASK_PREFIX == "[observation consumed"
        marker = build_marker(
            tool_name="t",
            args_preview="{}",
            key="obs:t:abc",
            tokens=123,
        )
        assert marker.startswith(MASK_PREFIX)
