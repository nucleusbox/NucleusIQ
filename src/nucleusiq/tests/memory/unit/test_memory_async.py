"""Extended async tests for SummaryMemory and SummaryWindowMemory with mock LLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nucleusiq.memory.summary import SummaryMemory
from nucleusiq.memory.summary_window import SummaryWindowMemory


def _mock_llm(content="Summary result"):
    llm = MagicMock()
    msg = MagicMock()
    msg.content = content
    llm.call = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=msg)])
    )
    return llm


# ═══════════════════════════════════════════════════════════════════════════════
# SummaryMemory with LLM
# ═══════════════════════════════════════════════════════════════════════════════


class TestSummaryMemoryWithLLM:

    @pytest.mark.asyncio
    async def test_aadd_message_with_llm(self):
        llm = _mock_llm("Condensed summary")
        mem = SummaryMemory(llm=llm)
        await mem.aadd_message("user", "Hello there")
        ctx = await mem.aget_context()
        assert len(ctx) == 1
        assert "Condensed summary" in ctx[0]["content"]

    @pytest.mark.asyncio
    async def test_aadd_message_without_llm(self):
        mem = SummaryMemory()
        await mem.aadd_message("user", "Hi")
        ctx = await mem.aget_context()
        assert "[user]" in ctx[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_messages(self):
        llm = _mock_llm("Updated summary")
        mem = SummaryMemory(llm=llm)
        await mem.aadd_message("user", "First")
        await mem.aadd_message("assistant", "Second")
        ctx = await mem.aget_context()
        assert len(ctx) == 1

    @pytest.mark.asyncio
    async def test_clear(self):
        mem = SummaryMemory()
        await mem.aadd_message("user", "msg")
        await mem.aclear()
        assert await mem.aget_context() == []

    @pytest.mark.asyncio
    async def test_export_import(self):
        mem = SummaryMemory()
        await mem.aadd_message("user", "content")
        state = await mem.aexport_state()
        mem2 = SummaryMemory()
        await mem2.aimport_state(state)
        assert await mem2.aget_context() == await mem.aget_context()

    def test_sync_add_with_llm_in_async_context(self):
        llm = _mock_llm()
        mem = SummaryMemory(llm=llm)
        mem.add_message("user", "test")
        ctx = mem.get_context()
        assert len(ctx) == 1

    def test_extract_text_dict(self):
        msg = MagicMock()
        msg.choices = [MagicMock(message={"content": "dict msg"})]
        assert SummaryMemory._extract_text(msg) == "dict msg"

    def test_extract_text_fallback(self):
        result = SummaryMemory._extract_text("raw_string")
        assert result == "raw_string"


# ═══════════════════════════════════════════════════════════════════════════════
# SummaryWindowMemory with LLM
# ═══════════════════════════════════════════════════════════════════════════════


class TestSummaryWindowMemoryWithLLM:

    @pytest.mark.asyncio
    async def test_aadd_within_window(self):
        llm = _mock_llm()
        mem = SummaryWindowMemory(window_size=5, llm=llm)
        await mem.aadd_message("user", "msg1")
        await mem.aadd_message("user", "msg2")
        ctx = await mem.aget_context()
        assert len(ctx) == 2

    @pytest.mark.asyncio
    async def test_aadd_overflow_triggers_summary(self):
        llm = _mock_llm("Compact summary")
        mem = SummaryWindowMemory(window_size=2, llm=llm)
        await mem.aadd_message("user", "first")
        await mem.aadd_message("user", "second")
        await mem.aadd_message("user", "third")
        ctx = await mem.aget_context()
        assert ctx[0]["role"] == "system"
        assert "Compact summary" in ctx[0]["content"]
        assert len(ctx) == 3  # summary + 2 recent

    @pytest.mark.asyncio
    async def test_overflow_no_llm(self):
        mem = SummaryWindowMemory(window_size=2)
        mem.add_message("user", "first")
        mem.add_message("user", "second")
        mem.add_message("user", "third")
        ctx = mem.get_context()
        assert ctx[0]["role"] == "system"
        assert "first" in ctx[0]["content"]

    @pytest.mark.asyncio
    async def test_clear(self):
        mem = SummaryWindowMemory(window_size=2)
        await mem.aadd_message("user", "x")
        await mem.aadd_message("user", "y")
        await mem.aadd_message("user", "z")
        await mem.aclear()
        assert await mem.aget_context() == []

    @pytest.mark.asyncio
    async def test_export_import(self):
        mem = SummaryWindowMemory(window_size=2)
        await mem.aadd_message("user", "a")
        await mem.aadd_message("user", "b")
        await mem.aadd_message("user", "c")
        state = await mem.aexport_state()
        mem2 = SummaryWindowMemory(window_size=2)
        await mem2.aimport_state(state)
        assert await mem2.aget_context() == await mem.aget_context()
