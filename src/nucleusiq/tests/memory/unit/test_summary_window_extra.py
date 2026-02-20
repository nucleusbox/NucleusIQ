"""Extra tests for SummaryWindowMemory — sync compact paths with LLM."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from nucleusiq.memory.summary_window import SummaryWindowMemory


def _mock_llm(content="Summary"):
    llm = MagicMock()
    msg = MagicMock()
    msg.content = content
    llm.call = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=msg)])
    )
    return llm


class TestSummaryWindowSyncCompact:

    def test_sync_compact_no_overflow(self):
        mem = SummaryWindowMemory(window_size=5)
        mem.add_message("user", "a")
        mem.add_message("user", "b")
        assert len(mem.get_context()) == 2

    def test_sync_compact_overflow_no_llm(self):
        mem = SummaryWindowMemory(window_size=2)
        mem.add_message("user", "first")
        mem.add_message("user", "second")
        mem.add_message("user", "third")
        ctx = mem.get_context()
        assert ctx[0]["role"] == "system"
        assert "first" in ctx[0]["content"]

    def test_sync_compact_with_llm(self):
        """The sync _compact_sync path with an LLM: runs the async via event loop."""
        llm = _mock_llm("LLM compressed summary")
        mem = SummaryWindowMemory(window_size=2, llm=llm)
        mem.add_message("user", "a")
        mem.add_message("user", "b")
        mem.add_message("user", "c")
        ctx = mem.get_context()
        assert ctx[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_async_compact_with_llm(self):
        llm = _mock_llm("Compressed history")
        mem = SummaryWindowMemory(window_size=2, llm=llm)
        await mem.aadd_message("user", "a")
        await mem.aadd_message("user", "b")
        await mem.aadd_message("user", "c")
        ctx = await mem.aget_context()
        assert ctx[0]["role"] == "system"
        assert "Compressed history" in ctx[0]["content"]

    @pytest.mark.asyncio
    async def test_drain_empty(self):
        mem = SummaryWindowMemory(window_size=5)
        overflow = mem._drain_overflow()
        assert overflow == []

    def test_format_messages(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        result = SummaryWindowMemory._format_messages(msgs)
        assert "[user]: hi" in result
        assert "[assistant]: hello" in result

    @pytest.mark.asyncio
    async def test_async_compact_no_overflow(self):
        mem = SummaryWindowMemory(window_size=5)
        await mem.aadd_message("user", "a")
        ctx = await mem.aget_context()
        assert len(ctx) == 1

    @pytest.mark.asyncio
    async def test_multiple_overflows(self):
        llm = _mock_llm("Summary so far")
        mem = SummaryWindowMemory(window_size=2, llm=llm)
        for i in range(6):
            await mem.aadd_message("user", f"msg-{i}")
        ctx = await mem.aget_context()
        assert ctx[0]["role"] == "system"
        assert len(ctx) == 3  # summary + 2 recent

    @pytest.mark.asyncio
    async def test_export_import_state(self):
        mem = SummaryWindowMemory(window_size=3)
        await mem.aadd_message("user", "hello")
        state = await mem.aexport_state()
        assert "messages" in state

        mem2 = SummaryWindowMemory(window_size=3)
        await mem2.aimport_state(state)
        ctx = await mem2.aget_context()
        assert len(ctx) == 1

    @pytest.mark.asyncio
    async def test_clear(self):
        mem = SummaryWindowMemory(window_size=3)
        await mem.aadd_message("user", "data")
        await mem.aclear()
        ctx = await mem.aget_context()
        assert len(ctx) == 0
