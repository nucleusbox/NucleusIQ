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
        assert cold_key in rendered, (
            "Evicted cold evidence must stay addressable in the catalog"
        )
        assert not any(
            m.role == "tool" and isinstance(m.content, str) and cold_key in m.content
            for m in result.messages
        ), "Cold evidence must not remain as a live tool turn"

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
    async def test_no_hot_set_evicts_payload_but_keeps_ref(self):
        """Without a hot set the bulky turn goes, the store ref stays."""
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
            assert old_key in rendered
            assert "[evidence available for recall]" in rendered
            assert not any(
                m.role == "tool" and isinstance(m.content, str) and old_key in m.content
                for m in result.messages
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

    @pytest.mark.asyncio
    async def test_emergency_catalogues_receipts_without_hot_set(self):
        """Evicted receipts stay addressable via a system catalog.

        Dedup / masker both tell the model to recall by ref. Dropping
        the receipt without a catalog produces [recall_error] and a
        re-ask loop. The catalog is chat-template safe (no orphaned
        tool messages).
        """
        filler = " ".join(["word"] * 30)
        receipt_key = "obs:read_pdf:keepme"
        tc = ToolCallRequest(id="call_old", name="read_pdf", arguments="{}")
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"original task {filler}"),
            ChatMessage(role="assistant", content="", tool_calls=[tc]),
            ChatMessage(
                role="tool",
                name="read_pdf",
                tool_call_id="call_old",
                content=_masked_marker(tool_name="read_pdf", key=receipt_key),
            ),
        ]
        for i in range(20):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=10_000, reserve=1000)
        config = _config(preserve_recent_turns=2)

        result = await EmergencyCompactor().compact(msgs, budget, config, counter)

        rendered = "\n".join(
            m.content for m in result.messages if isinstance(m.content, str)
        )
        assert receipt_key in rendered
        assert "[evidence available for recall]" in rendered
        assert not any(
            m.role == "tool" and isinstance(m.content, str) and receipt_key in m.content
            for m in result.messages
        )
        assert any("catalogued" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_emergency_offloads_raw_tool_result_then_catalogues_ref(self):
        """Unmasked tool payloads must enter the store before eviction."""
        filler = " ".join(["word"] * 30)
        raw = "FULL_PDF_PAGE " + " ".join(["word"] * 80)
        tc = ToolCallRequest(id="call_raw", name="read_pdf", arguments="{}")
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"original task {filler}"),
            ChatMessage(role="assistant", content="", tool_calls=[tc]),
            ChatMessage(
                role="tool",
                name="read_pdf",
                tool_call_id="call_raw",
                content=raw,
            ),
        ]
        for i in range(20):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        store = ContentStore()
        budget = _budget(counter.count_messages(msgs), max_t=10_000, reserve=1000)
        config = _config(preserve_recent_turns=2)

        result = await EmergencyCompactor().compact(
            msgs, budget, config, counter, store
        )

        assert store.size == 1
        stored_key = store.keys()[0]
        assert store.retrieve(stored_key) == raw
        rendered = "\n".join(
            m.content for m in result.messages if isinstance(m.content, str)
        )
        assert stored_key in rendered
        assert "FULL_PDF_PAGE" not in rendered
        assert "[evidence available for recall]" in rendered

    @pytest.mark.asyncio
    async def test_conversation_catalogues_context_ref_receipts(self):
        """Tier-1 [context_ref:] receipts are the same contract as masker refs."""
        filler = " ".join(["word"] * 30)
        offload_key = "web:abc123def456"
        tc = ToolCallRequest(id="call_web", name="web", arguments="{}")
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"task {filler}"),
            ChatMessage(role="assistant", content="", tool_calls=[tc]),
            ChatMessage(
                role="tool",
                name="web",
                tool_call_id="call_web",
                content=(
                    f"[context_ref: {offload_key}]\n"
                    "[original size: ~400 tokens, offloaded]\n"
                    "--- preview ---\n"
                    "snippet\n"
                    "--- end preview ---"
                ),
            ),
        ]
        for i in range(6):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        config = _config(preserve_recent_turns=2)

        result = await ConversationCompactor().compact(msgs, budget, config, counter)

        rendered = "\n".join(
            m.content for m in result.messages if isinstance(m.content, str)
        )
        assert offload_key in rendered
        assert "[evidence available for recall]" in rendered

    @pytest.mark.asyncio
    async def test_catalog_names_each_refs_arguments(self):
        """Without the arguments a ref is anonymous: the model cannot tell
        which input it belongs to and re-fetches it (0.7.13 dedup loop)."""
        filler = " ".join(["word"] * 30)
        raw_a = "PAGE_A " + " ".join(["word"] * 80)
        raw_b = "PAGE_B " + " ".join(["word"] * 80)
        msgs: list[ChatMessage] = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content=f"task {filler}"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="c_a", name="read_pdf", arguments='{"path": "docs/a.pdf"}'
                    ),
                    ToolCallRequest(
                        id="c_b", name="read_pdf", arguments='{"path": "docs/b.pdf"}'
                    ),
                ],
            ),
            ChatMessage(
                role="tool", name="read_pdf", tool_call_id="c_a", content=raw_a
            ),
            ChatMessage(
                role="tool", name="read_pdf", tool_call_id="c_b", content=raw_b
            ),
        ]
        for i in range(20):
            msgs.append(ChatMessage(role="user", content=f"q{i} {filler}"))
            msgs.append(ChatMessage(role="assistant", content=f"a{i} {filler}"))

        counter = DefaultTokenCounter()
        store = ContentStore()
        budget = _budget(counter.count_messages(msgs), max_t=10_000, reserve=1000)
        result = await EmergencyCompactor().compact(
            msgs, budget, _config(preserve_recent_turns=2), counter, store
        )

        catalog = next(
            m.content
            for m in result.messages
            if isinstance(m.content, str)
            and m.content.startswith("[evidence available for recall]")
        )
        assert 'args={"path": "docs/a.pdf"}' in catalog
        assert 'args={"path": "docs/b.pdf"}' in catalog
        assert "PAGE_A" not in catalog


class TestTaskHeadPinning:
    """I1: everything the model was *given* before its first turn survives
    both compaction tiers — not just the first user message."""

    @staticmethod
    def _transcript() -> list[ChatMessage]:
        filler = " ".join(["word"] * 30)
        msgs = [
            ChatMessage(role="system", content="You are an analyst."),
            ChatMessage(role="user", content="Help the user."),
            ChatMessage(role="user", content="Resources: docs/a.pdf, docs/b.pdf"),
            ChatMessage(role="user", content=f"OBJECTIVE extract totals {filler}"),
        ]
        for i in range(20):
            tc = ToolCallRequest(id=f"c{i}", name="read_pdf", arguments="{}")
            msgs.append(ChatMessage(role="assistant", content="", tool_calls=[tc]))
            msgs.append(
                ChatMessage(
                    role="tool", name="read_pdf", tool_call_id=f"c{i}", content=filler
                )
            )
        return msgs

    @staticmethod
    def _head_texts(messages: list[ChatMessage]) -> str:
        return "\n".join(
            m.content
            for m in messages
            if m.role in ("system", "user") and isinstance(m.content, str)
        )

    @pytest.mark.asyncio
    async def test_emergency_keeps_resources_and_objective(self):
        msgs = self._transcript()
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=10_000, reserve=1000)
        result = await EmergencyCompactor().compact(
            msgs, budget, _config(preserve_recent_turns=1), counter, ContentStore()
        )
        head = self._head_texts(result.messages)
        assert "Help the user." in head
        assert "Resources: docs/a.pdf, docs/b.pdf" in head
        assert "OBJECTIVE extract totals" in head
        assert len(result.messages) < len(msgs)

    @pytest.mark.asyncio
    async def test_conversation_keeps_resources_and_objective(self):
        msgs = self._transcript()
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs))
        result = await ConversationCompactor().compact(
            msgs, budget, _config(preserve_recent_turns=2), counter
        )
        head = self._head_texts(result.messages)
        assert "Resources: docs/a.pdf, docs/b.pdf" in head
        assert "OBJECTIVE extract totals" in head
        assert len(result.messages) < len(msgs)

    def test_memory_first_transcript_still_pins_first_user(self):
        from nucleusiq.agents.context.compactor import _split_task_head

        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="assistant", content="earlier answer"),
            ChatMessage(role="user", content="the task"),
            ChatMessage(role="assistant", content="working"),
        ]
        head, rest = _split_task_head(msgs)
        assert [m.content for m in head] == ["sys", "the task"]
        assert [m.content for m in rest] == ["earlier answer", "working"]

        head, rest = _split_task_head(msgs[:1])
        assert [m.content for m in head] == ["sys"] and rest == []


class TestUnseenRoundOverflow:
    """One round of tool calls returns more than the working budget holds.

    Pre-fix: every result was under the absolute ``tool_result_threshold``,
    so Tier 1 freed nothing, Tier 2 had nothing older to evict, and Tier 3
    either dropped the whole round (when anything followed it) or gave up
    (when it was the last group) — either way the model answered without
    the evidence it had just asked for.
    """

    @staticmethod
    def _round(n_results: int = 9, words: int = 700, trailing_user: bool = False):
        body = " ".join(f"w{i}" for i in range(words))
        calls = [
            ToolCallRequest(
                id=f"c{i}", name="read_pdf", arguments=f'{{"path": "d{i}.pdf"}}'
            )
            for i in range(n_results)
        ]
        msgs = [
            ChatMessage(role="system", content="You are an analyst."),
            ChatMessage(
                role="user",
                content="Resources: "
                + ", ".join(f"d{i}.pdf" for i in range(n_results)),
            ),
            ChatMessage(role="assistant", content="", tool_calls=calls),
        ]
        for i in range(n_results):
            msgs.append(
                ChatMessage(
                    role="tool",
                    name="read_pdf",
                    tool_call_id=f"c{i}",
                    content=f"[d{i}.pdf]\nInvoice {i} total {i * 1000}\n" + body,
                )
            )
        if trailing_user:
            msgs.append(ChatMessage(role="user", content="Now answer in JSON."))
        return msgs

    @pytest.mark.asyncio
    async def test_adaptive_tier1_keeps_the_round_recallable(self):
        from nucleusiq.agents.context.compactor import Compactor

        msgs = self._round()
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=6_000, reserve=500)
        assert budget.utilization >= 1.0  # the round alone overflows the window
        store = ContentStore()
        config = _config(max_context_tokens=6_000, tool_result_threshold=20_000)

        out, events = await Compactor().compact(msgs, budget, config, counter, store)

        strategies = [e.strategy for e in events]
        assert "tool_result_compactor" in strategies
        assert "emergency_compactor" not in strategies, strategies
        ev = next(e for e in events if e.strategy == "tool_result_compactor")
        assert ev.artifacts_offloaded >= 1 and ev.unseen_evicted == 0
        # Under the Tier-1 trigger again, nothing dropped: same message count.
        assert (
            counter.count_messages(out)
            <= config.tool_compaction_trigger * budget.effective_limit
        )
        assert len(out) == len(msgs)
        receipts = [
            m for m in out if m.role == "tool" and m.content.startswith("[context_ref:")
        ]
        assert receipts, "results should be offloaded, not evicted"
        for r in receipts:
            assert "before you saw it" in r.content
            assert "recall_tool_result(ref=" in r.content
            assert "Do NOT call the original tool again" in r.content
            assert "--- preview ---" in r.content and "Invoice" in r.content
        # Largest-first: with equal sizes, all nine were needed here.
        assert (
            len(receipts) == len(store.list_all())
            if hasattr(store, "list_all")
            else True
        )

    @pytest.mark.asyncio
    async def test_emergency_squeezes_a_single_oversized_tail(self):
        from nucleusiq.agents.context.compactor import Compactor

        msgs = self._round()
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=6_000, reserve=500)
        config = _config(
            max_context_tokens=6_000,
            tool_result_threshold=20_000,
            enable_offloading=False,
        )

        out, events = await Compactor().compact(msgs, budget, config, counter, None)

        ev = next(e for e in events if e.strategy == "emergency_compactor")
        assert ev.unseen_evicted >= 1
        assert ev.tokens_freed > 0
        assert counter.count_messages(out) <= budget.effective_limit
        # Truncated in place, not dropped — the model still sees head + tail
        # of each result and the transcript stays well-formed.
        assert len(out) == len(msgs)
        assert all(m.tool_call_id for m in out if m.role == "tool")
        assert any("[...truncated" in m.content for m in out if m.role == "tool")

    @pytest.mark.asyncio
    async def test_emergency_counts_unseen_results_it_drops(self):
        from nucleusiq.agents.context.compactor import Compactor

        msgs = self._round(trailing_user=True)
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=6_000, reserve=500)
        config = _config(
            max_context_tokens=6_000,
            tool_result_threshold=20_000,
            enable_offloading=False,
        )

        out, events = await Compactor().compact(msgs, budget, config, counter, None)

        ev = next(e for e in events if e.strategy == "emergency_compactor")
        # The whole unread round sat in front of the trailing user message.
        assert ev.unseen_evicted == 9
        assert "Now answer in JSON." in [m.content for m in out]
        assert "Resources: " in "".join(m.content for m in out if m.role == "user")

    @pytest.mark.asyncio
    async def test_seen_results_are_not_counted(self):
        from nucleusiq.agents.context.compactor import Compactor

        msgs = self._round()
        msgs.append(ChatMessage(role="assistant", content="I have read all nine."))
        msgs.append(ChatMessage(role="user", content="Now answer in JSON."))
        counter = DefaultTokenCounter()
        budget = _budget(counter.count_messages(msgs), max_t=6_000, reserve=500)
        config = _config(
            max_context_tokens=6_000,
            tool_result_threshold=20_000,
            enable_offloading=False,
        )
        out, events = await Compactor().compact(msgs, budget, config, counter, None)
        assert events, "the transcript overflowed; something must have compacted"
        assert sum(e.unseen_evicted for e in events) == 0
        assert counter.count_messages(out) < counter.count_messages(msgs)
