"""Tests for ObservationMasker — Tier 0 post-response strategy."""

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.context.strategies.observation_masker import ObservationMasker


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
