"""Tests for compaction strategies: ToolResult, Conversation, Emergency."""

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.budget import ContextBudget
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.context.strategies.conversation import ConversationCompactor
from nucleusiq.agents.context.strategies.emergency import EmergencyCompactor
from nucleusiq.agents.context.strategies.tool_result import ToolResultCompactor


def _budget(allocated: int, max_t: int = 100_000, reserve: int = 8192) -> ContextBudget:
    return ContextBudget(
        max_tokens=max_t,
        response_reserve=reserve,
        allocated=allocated,
        by_region={"tool_result": allocated},
    )


def _config(**overrides) -> ContextConfig:
    defaults = {
        "max_context_tokens": 100_000,
        "tool_result_threshold": 100,
        "preserve_recent_turns": 2,
    }
    defaults.update(overrides)
    return ContextConfig(**defaults)


class TestToolResultCompactor:
    @pytest.mark.asyncio
    async def test_truncates_large_tool_result(self):
        content = "\n".join(f"line {i}: {'x' * 100}" for i in range(50))
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="tool", name="search", content=content),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))

        result = await ToolResultCompactor().compact(
            msgs, budget, _config(enable_offloading=False), counter
        )
        assert result.tokens_freed > 0
        assert len(result.messages) == 2
        assert "truncated" in result.messages[1].content.lower()

    @pytest.mark.asyncio
    async def test_offloads_to_store(self):
        content = "\n".join(f"line {i}: {'x' * 100}" for i in range(50))
        msgs = [
            ChatMessage(role="tool", name="web", content=content),
        ]
        counter = DefaultTokenCounter()
        store = ContentStore()
        budget = _budget(counter.count_messages(msgs))

        result = await ToolResultCompactor().compact(
            msgs, budget, _config(), counter, store
        )
        assert result.artifacts_offloaded == 1
        assert store.size == 1
        assert "[context_ref:" in result.messages[0].content

    @pytest.mark.asyncio
    async def test_skips_small_tool_results(self):
        msgs = [
            ChatMessage(role="tool", name="calc", content="42"),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(10)
        result = await ToolResultCompactor().compact(
            msgs, budget, _config(tool_result_threshold=1000), counter
        )
        assert result.tokens_freed == 0
        assert result.messages[0].content == "42"

    @pytest.mark.asyncio
    async def test_skips_non_tool_messages(self):
        msgs = [
            ChatMessage(role="user", content="x" * 500),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(200)
        result = await ToolResultCompactor().compact(msgs, budget, _config(), counter)
        assert result.tokens_freed == 0

    @pytest.mark.asyncio
    async def test_truncates_dense_content_few_newlines(self):
        """Dense content (single long paragraph) must still be truncated."""
        dense = "word " * 2000
        msgs = [
            ChatMessage(role="tool", name="pdf_read", content=dense),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        result = await ToolResultCompactor().compact(
            msgs, budget, _config(enable_offloading=False), counter
        )
        assert result.tokens_freed > 0, (
            "Truncation must free tokens for dense content with few newlines"
        )

    @pytest.mark.asyncio
    async def test_offloads_dense_content_frees_tokens(self):
        """Offloading dense content must free tokens, not increase them."""
        dense = "word " * 2000
        msgs = [
            ChatMessage(role="tool", name="pdf_read", content=dense),
        ]
        counter = DefaultTokenCounter()
        store = ContentStore()
        budget = _budget(counter.count_messages(msgs))
        result = await ToolResultCompactor().compact(
            msgs, budget, _config(), counter, store
        )
        assert result.tokens_freed > 0, (
            "Offloading dense content must free tokens — "
            "preview must be smaller than original"
        )
        assert result.artifacts_offloaded == 1


class TestConversationCompactor:
    @pytest.mark.asyncio
    async def test_removes_old_turns(self):
        filler = " ".join(["word"] * 50)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"old question 1 {filler}"),
            ChatMessage(role="assistant", content=f"old answer 1 {filler}"),
            ChatMessage(role="user", content=f"old question 2 {filler}"),
            ChatMessage(role="assistant", content=f"old answer 2 {filler}"),
            ChatMessage(role="user", content="recent question"),
            ChatMessage(role="assistant", content="recent answer"),
            ChatMessage(role="user", content="latest question"),
            ChatMessage(role="assistant", content="latest answer"),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(msgs, budget, config, counter)
        assert result.tokens_freed > 0
        assert result.entries_removed > 0
        assert any(
            "compacted" in m.content.lower()
            for m in result.messages
            if isinstance(m.content, str)
        )

    @pytest.mark.asyncio
    async def test_preserves_system_and_recent(self):
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="a1"),
            ChatMessage(role="user", content="q2"),
            ChatMessage(role="assistant", content="a2"),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(500)
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(msgs, budget, config, counter)
        assert result.messages[0].role == "system"
        assert result.messages[-1].content == "a2"

    @pytest.mark.asyncio
    async def test_nothing_to_evict(self):
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(100)
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(msgs, budget, config, counter)
        assert result.tokens_freed == 0


class TestEmergencyCompactor:
    @pytest.mark.asyncio
    async def test_emergency_drops_most_messages(self):
        filler = " ".join(["word"] * 30)
        msgs = [ChatMessage(role="system", content="sys")]
        for i in range(20):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=10_000, reserve=1000)
        config = _config(preserve_recent_turns=2)

        result = await EmergencyCompactor().compact(msgs, budget, config, counter)
        assert result.tokens_freed > 0
        assert result.entries_removed > 0
        assert result.messages[0].role == "system"
        assert "CONTEXT COMPACTED" in result.messages[1].content
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_nothing_to_evict_when_short(self):
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ]
        counter = DefaultTokenCounter()
        budget = _budget(50)
        config = _config(preserve_recent_turns=2)

        result = await EmergencyCompactor().compact(msgs, budget, config, counter)
        assert result.tokens_freed == 0
