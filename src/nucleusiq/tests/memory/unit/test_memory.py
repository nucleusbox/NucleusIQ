"""
Comprehensive test suite for NucleusIQ Memory Module.

Covers:
  - BaseMemory contract enforcement
  - All 5 built-in strategies
  - MemoryFactory creation + registration
  - Async variants
  - State export / import round-trips
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional

from nucleusiq.memory import (
    BaseMemory,
    MemoryFactory,
    MemoryStrategy,
    FullHistoryMemory,
    SlidingWindowMemory,
    TokenBudgetMemory,
    SummaryMemory,
    SummaryWindowMemory,
)


# ===================================================================
# BaseMemory — abstract contract
# ===================================================================

class TestBaseMemoryContract:
    """Verify BaseMemory cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseMemory()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement(self):
        class IncompleteMemory(BaseMemory):
            @property
            def strategy_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteMemory()


# ===================================================================
# FullHistoryMemory
# ===================================================================

class TestFullHistoryMemory:

    def test_add_and_get(self):
        mem = FullHistoryMemory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi!")
        ctx = mem.get_context()
        assert len(ctx) == 2
        assert ctx[0] == {"role": "user", "content": "Hello"}
        assert ctx[1] == {"role": "assistant", "content": "Hi!"}

    def test_strategy_name(self):
        assert FullHistoryMemory().strategy_name == "full_history"

    def test_clear(self):
        mem = FullHistoryMemory()
        mem.add_message("user", "msg")
        mem.clear()
        assert mem.get_context() == []

    def test_export_import_roundtrip(self):
        mem = FullHistoryMemory()
        mem.add_message("user", "one")
        mem.add_message("assistant", "two")
        state = mem.export_state()

        mem2 = FullHistoryMemory()
        mem2.import_state(state)
        assert mem2.get_context() == mem.get_context()

    @pytest.mark.asyncio
    async def test_async_variants(self):
        mem = FullHistoryMemory()
        await mem.aadd_message("user", "async msg")
        ctx = await mem.aget_context()
        assert len(ctx) == 1
        await mem.aclear()
        assert await mem.aget_context() == []

    def test_user_session_metadata(self):
        mem = FullHistoryMemory(
            user_id="u1", session_id="s1", metadata={"env": "test"}
        )
        assert mem.user_id == "u1"
        assert mem.session_id == "s1"
        assert mem.metadata["env"] == "test"

    def test_query_param_ignored(self):
        mem = FullHistoryMemory()
        mem.add_message("user", "data")
        ctx = mem.get_context(query="search term")
        assert len(ctx) == 1


# ===================================================================
# SlidingWindowMemory
# ===================================================================

class TestSlidingWindowMemory:

    def test_window_limit(self):
        mem = SlidingWindowMemory(window_size=3)
        for i in range(5):
            mem.add_message("user", f"msg-{i}")
        ctx = mem.get_context()
        assert len(ctx) == 3
        assert ctx[0]["content"] == "msg-2"
        assert ctx[2]["content"] == "msg-4"

    def test_strategy_name(self):
        assert SlidingWindowMemory().strategy_name == "sliding_window"

    def test_default_window(self):
        mem = SlidingWindowMemory()
        for i in range(15):
            mem.add_message("user", f"m{i}")
        assert len(mem.get_context()) == 10  # default window_size=10

    def test_clear(self):
        mem = SlidingWindowMemory(window_size=5)
        mem.add_message("user", "x")
        mem.clear()
        assert mem.get_context() == []

    def test_export_import_roundtrip(self):
        mem = SlidingWindowMemory(window_size=3)
        for i in range(5):
            mem.add_message("user", f"msg-{i}")
        state = mem.export_state()

        mem2 = SlidingWindowMemory(window_size=3)
        mem2.import_state(state)
        assert mem2.get_context() == mem.get_context()


# ===================================================================
# TokenBudgetMemory
# ===================================================================

class TestTokenBudgetMemory:

    def test_eviction(self):
        mem = TokenBudgetMemory(
            max_tokens=10,
            token_counter=lambda t: len(t),  # 1 char = 1 token
        )
        mem.add_message("user", "12345")      # 5 tokens
        mem.add_message("user", "67890")      # 5 tokens  -> 10 total
        assert len(mem.get_context()) == 2

        mem.add_message("user", "ab")          # 2 tokens -> 12 -> evict first
        ctx = mem.get_context()
        assert len(ctx) == 2
        assert ctx[0]["content"] == "67890"
        assert ctx[1]["content"] == "ab"

    def test_strategy_name(self):
        assert TokenBudgetMemory().strategy_name == "token_budget"

    def test_clear_resets_tokens(self):
        mem = TokenBudgetMemory(max_tokens=100)
        mem.add_message("user", "data")
        mem.clear()
        assert mem.get_context() == []

    def test_export_import_roundtrip(self):
        counter = lambda t: len(t)
        mem = TokenBudgetMemory(max_tokens=20, token_counter=counter)
        mem.add_message("user", "hello")
        mem.add_message("assistant", "world")
        state = mem.export_state()

        mem2 = TokenBudgetMemory(max_tokens=20, token_counter=counter)
        mem2.import_state(state)
        assert mem2.get_context() == mem.get_context()

    def test_single_oversized_message(self):
        mem = TokenBudgetMemory(
            max_tokens=3,
            token_counter=lambda t: len(t),
        )
        mem.add_message("user", "toolong")  # 7 tokens > 3
        assert mem.get_context() == []       # evicts itself


# ===================================================================
# SummaryMemory (no LLM)
# ===================================================================

class TestSummaryMemoryNoLLM:

    def test_fallback_without_llm(self):
        mem = SummaryMemory()
        mem.add_message("user", "Hello world")
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert "[user]: Hello world" in ctx[0]["content"]

    def test_strategy_name(self):
        assert SummaryMemory().strategy_name == "summary"

    def test_clear(self):
        mem = SummaryMemory()
        mem.add_message("user", "msg")
        mem.clear()
        assert mem.get_context() == []

    def test_export_import_roundtrip(self):
        mem = SummaryMemory()
        mem.add_message("user", "stuff")
        state = mem.export_state()

        mem2 = SummaryMemory()
        mem2.import_state(state)
        assert mem2.get_context() == mem.get_context()


# ===================================================================
# SummaryWindowMemory (no LLM)
# ===================================================================

class TestSummaryWindowMemoryNoLLM:

    def test_basic_windowing(self):
        mem = SummaryWindowMemory(window_size=3)
        for i in range(3):
            mem.add_message("user", f"msg-{i}")
        ctx = mem.get_context()
        assert len(ctx) == 3

    def test_overflow_creates_summary(self):
        mem = SummaryWindowMemory(window_size=2)
        mem.add_message("user", "first")
        mem.add_message("user", "second")
        mem.add_message("user", "third")
        ctx = mem.get_context()
        assert len(ctx) == 3  # 1 summary + 2 recent
        assert ctx[0]["role"] == "system"
        assert "first" in ctx[0]["content"]

    def test_strategy_name(self):
        assert SummaryWindowMemory().strategy_name == "summary_window"

    def test_clear(self):
        mem = SummaryWindowMemory(window_size=2)
        mem.add_message("user", "a")
        mem.add_message("user", "b")
        mem.add_message("user", "c")
        mem.clear()
        assert mem.get_context() == []

    def test_export_import_roundtrip(self):
        mem = SummaryWindowMemory(window_size=2)
        mem.add_message("user", "first")
        mem.add_message("user", "second")
        mem.add_message("user", "third")
        state = mem.export_state()

        mem2 = SummaryWindowMemory(window_size=2)
        mem2.import_state(state)
        assert mem2.get_context() == mem.get_context()


# ===================================================================
# MemoryFactory
# ===================================================================

class TestMemoryFactory:

    def test_create_full_history(self):
        mem = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)
        assert isinstance(mem, FullHistoryMemory)

    def test_create_sliding_window(self):
        mem = MemoryFactory.create_memory(
            MemoryStrategy.SLIDING_WINDOW, window_size=5
        )
        assert isinstance(mem, SlidingWindowMemory)
        assert mem.window_size == 5

    def test_create_token_budget(self):
        mem = MemoryFactory.create_memory(
            MemoryStrategy.TOKEN_BUDGET, max_tokens=1024
        )
        assert isinstance(mem, TokenBudgetMemory)

    def test_create_summary(self):
        mem = MemoryFactory.create_memory(MemoryStrategy.SUMMARY)
        assert isinstance(mem, SummaryMemory)

    def test_create_summary_window(self):
        mem = MemoryFactory.create_memory(
            MemoryStrategy.SUMMARY_WINDOW, window_size=5
        )
        assert isinstance(mem, SummaryWindowMemory)

    def test_register_custom_strategy(self):
        class CustomMemory(BaseMemory):
            @property
            def strategy_name(self) -> str:
                return "custom"

            def add_message(self, role: str, content: str, **kw: Any) -> None:
                pass

            def get_context(self, query: Optional[str] = None, **kw: Any) -> List[Dict[str, str]]:
                return []

            def clear(self) -> None:
                pass

        MemoryFactory.register_memory("custom", CustomMemory)
        mem = MemoryFactory.create_memory("custom")
        assert isinstance(mem, CustomMemory)

        # Cleanup
        del MemoryFactory._registry["custom"]

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            MemoryFactory.register_memory(
                MemoryStrategy.FULL_HISTORY, FullHistoryMemory
            )

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            MemoryFactory.create_memory("nonexistent")


# ===================================================================
# MemoryStrategy enum
# ===================================================================

class TestMemoryStrategy:

    def test_all_values(self):
        values = {s.value for s in MemoryStrategy}
        assert values == {
            "full_history",
            "sliding_window",
            "token_budget",
            "summary",
            "summary_window",
        }

    def test_from_string(self):
        s = MemoryStrategy("sliding_window")
        assert s == MemoryStrategy.SLIDING_WINDOW


# ===================================================================
# Async state round-trip
# ===================================================================

class TestAsyncStateRoundTrip:

    @pytest.mark.asyncio
    async def test_async_export_import(self):
        mem = SlidingWindowMemory(window_size=3)
        await mem.aadd_message("user", "a")
        await mem.aadd_message("assistant", "b")

        state = await mem.aexport_state()
        assert "messages" in state

        mem2 = SlidingWindowMemory(window_size=3)
        await mem2.aimport_state(state)
        ctx = await mem2.aget_context()
        assert len(ctx) == 2

    @pytest.mark.asyncio
    async def test_async_initialize(self):
        mem = FullHistoryMemory()
        await mem.ainitialize()  # should not raise
