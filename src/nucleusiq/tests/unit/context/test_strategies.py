"""Tests for compaction strategies: ToolResult, Conversation, Emergency."""

import pytest
from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.context.budget import ContextBudget
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.context.strategies.conversation import ConversationCompactor
from nucleusiq.agents.context.strategies.emergency import EmergencyCompactor
from nucleusiq.agents.context.strategies.tool_result import ToolResultCompactor


def _masked_marker(*, tool_name: str, key: str, tokens: int = 1234) -> str:
    """Build a marker that exactly matches what ObservationMasker emits.

    Strategies parse the ``ref:`` line to identify hot-recalled tool
    results.  The format must therefore stay in lock-step with
    :mod:`nucleusiq.agents.context.strategies.observation_masker`.
    """
    return (
        "[observation consumed]\n"
        f"tool: {tool_name}\n"
        f"args: {{}}\n"
        f"ref: {key}\n"
        f"size: ~{tokens} tokens\n"
        f'To retrieve: call recall_tool_result(ref="{key}")'
    )


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
        # Context Mgmt v2 — Step 2: layout is
        # [system, first_user_task, COMPACTED marker, ...last_group].
        # The original task pin is invariant **I1**.
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user", (
            "EmergencyCompactor must pin the original user task (I1)"
        )
        assert result.messages[1].content.startswith("q0"), (
            "Pinned user message must be the first user task, not a later turn"
        )
        assert "CONTEXT COMPACTED" in result.messages[2].content
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


# ====================================================================== #
# Context Mgmt v2 — Step 2: I1 (task pin) + hot-set rescue                #
# ====================================================================== #


class TestConversationCompactorPinning:
    """Invariant **I1**: the first user message ("the task") must
    survive every conversation compaction, even when it falls outside
    ``preserve_recent_turns``.  Without this pin the synthesis pass
    routinely produces ``"I don't have the instructions you're
    referring to..."`` refusals on long autonomous runs.
    """

    @pytest.mark.asyncio
    async def test_first_user_message_pinned_when_far_outside_window(self):
        """The original task is preserved when buried under many later turns."""
        filler = " ".join(["word"] * 50)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"ORIGINAL_TASK {filler}"),
        ]
        for i in range(10):
            msgs.append(ChatMessage(role="user", content=f"follow-up {i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"answer {i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(msgs, budget, config, counter)

        assert result.tokens_freed > 0
        assert any(
            isinstance(m.content, str) and m.content.startswith("ORIGINAL_TASK")
            for m in result.messages
        ), "ConversationCompactor must pin the original user task (I1)"

    @pytest.mark.asyncio
    async def test_first_user_message_pinned_without_system_prompt(self):
        """Pin is unconditional — works even with no system header."""
        filler = " ".join(["word"] * 30)
        msgs = [ChatMessage(role="user", content=f"TASK {filler}")]
        for i in range(8):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(msgs, budget, config, counter)

        first = result.messages[0]
        assert first.role == "user"
        assert isinstance(first.content, str) and first.content.startswith("TASK")


class TestConversationCompactorHotSet:
    """Hot-set rescue: tool messages whose ``ref`` was recalled in
    the last few turns must NOT be evicted — re-evicting them would
    silently undo the model's just-issued recall request.
    """

    @pytest.mark.asyncio
    async def test_hot_ref_rescues_tool_group_from_eviction(self):
        """An old tool turn referencing a hot ref is moved to the pinned head."""
        filler = " ".join(["word"] * 30)
        hot_key = "store_key_alpha"
        cold_key = "store_key_beta"

        tc_alpha = ToolCallRequest(id="call_alpha", name="search", arguments="{}")
        tc_beta = ToolCallRequest(id="call_beta", name="search", arguments="{}")

        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"original task {filler}"),
            ChatMessage(role="assistant", content="", tool_calls=[tc_alpha]),
            ChatMessage(
                role="tool",
                name="search",
                tool_call_id="call_alpha",
                content=_masked_marker(tool_name="search", key=hot_key),
            ),
            ChatMessage(role="assistant", content="", tool_calls=[tc_beta]),
            ChatMessage(
                role="tool",
                name="search",
                tool_call_id="call_beta",
                content=_masked_marker(tool_name="search", key=cold_key),
            ),
        ]
        for i in range(6):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(
            msgs, budget, config, counter, hot_set=frozenset({hot_key})
        )

        rendered = "\n".join(
            m.content for m in result.messages if isinstance(m.content, str)
        )
        assert hot_key in rendered, (
            "Hot-recalled tool result must be rescued from eviction"
        )
        assert cold_key not in rendered, (
            "Non-hot tool results in evictable range should still be evicted"
        )

    @pytest.mark.asyncio
    async def test_rescued_group_appears_before_marker(self):
        """Rescued evidence is part of the *head*, never inside the marker tail."""
        filler = " ".join(["word"] * 30)
        hot_key = "store_key_hot"

        tc = ToolCallRequest(id="call_hot", name="read_pdf", arguments="{}")
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"task {filler}"),
            ChatMessage(role="assistant", content="", tool_calls=[tc]),
            ChatMessage(
                role="tool",
                name="read_pdf",
                tool_call_id="call_hot",
                content=_masked_marker(tool_name="read_pdf", key=hot_key),
            ),
        ]
        for i in range(6):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(
            msgs, budget, config, counter, hot_set=frozenset({hot_key})
        )

        marker_idx = next(
            i
            for i, m in enumerate(result.messages)
            if isinstance(m.content, str) and "compacted" in m.content.lower()
        )
        hot_indices = [
            i
            for i, m in enumerate(result.messages)
            if isinstance(m.content, str) and hot_key in m.content
        ]
        assert hot_indices, "Hot evidence missing from output"
        assert all(i < marker_idx for i in hot_indices), (
            "Rescued hot evidence must precede the compaction marker"
        )

    @pytest.mark.asyncio
    async def test_no_hot_set_means_normal_eviction(self):
        """When ``hot_set`` is None or empty, behaviour is unchanged."""
        filler = " ".join(["word"] * 30)
        old_key = "old_key"
        tc = ToolCallRequest(id="call_old", name="search", arguments="{}")
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"task {filler}"),
            ChatMessage(role="assistant", content="", tool_calls=[tc]),
            ChatMessage(
                role="tool",
                name="search",
                tool_call_id="call_old",
                content=_masked_marker(tool_name="search", key=old_key),
            ),
        ]
        for i in range(6):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        config = _config(preserve_recent_turns=2)

        result_none = await ConversationCompactor().compact(
            msgs, budget, config, counter, hot_set=None
        )
        result_empty = await ConversationCompactor().compact(
            msgs, budget, config, counter, hot_set=frozenset()
        )

        for result in (result_none, result_empty):
            rendered = "\n".join(
                m.content for m in result.messages if isinstance(m.content, str)
            )
            assert old_key not in rendered, (
                "Without a hot set, the old tool result should be evicted normally"
            )


class TestEmergencyCompactorHotSet:
    """Even under emergency pressure, hot-recalled tool turns survive."""

    @pytest.mark.asyncio
    async def test_hot_ref_rescued_under_emergency(self):
        filler = " ".join(["word"] * 30)
        hot_key = "store_key_emergency"

        tc = ToolCallRequest(id="call_e", name="read_pdf", arguments="{}")
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"original task {filler}"),
            ChatMessage(role="assistant", content="", tool_calls=[tc]),
            ChatMessage(
                role="tool",
                name="read_pdf",
                tool_call_id="call_e",
                content=_masked_marker(tool_name="read_pdf", key=hot_key),
            ),
        ]
        for i in range(20):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=10_000, reserve=1000)
        config = _config(preserve_recent_turns=2)

        result = await EmergencyCompactor().compact(
            msgs, budget, config, counter, hot_set=frozenset({hot_key})
        )

        rendered = "\n".join(
            m.content for m in result.messages if isinstance(m.content, str)
        )
        assert hot_key in rendered, (
            "EmergencyCompactor must rescue hot-recalled tool turns"
        )
        assert any("rescued" in w for w in result.warnings), (
            "Rescue count must be reported in warnings for telemetry visibility"
        )

    @pytest.mark.asyncio
    async def test_emergency_pins_first_user_even_without_hot_set(self):
        """I1 holds in EmergencyCompactor too, independent of hot-set."""
        filler = " ".join(["word"] * 30)
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"ORIGINAL_TASK {filler}"),
        ]
        for i in range(20):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=10_000, reserve=1000)
        config = _config(preserve_recent_turns=2)

        result = await EmergencyCompactor().compact(msgs, budget, config, counter)

        assert result.messages[1].role == "user"
        assert isinstance(result.messages[1].content, str) and result.messages[
            1
        ].content.startswith("ORIGINAL_TASK"), (
            "EmergencyCompactor must pin the original task even with no hot set"
        )
