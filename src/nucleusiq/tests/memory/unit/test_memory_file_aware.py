"""Tests for file-aware memory — attachment metadata storage and retrieval.

Verifies that all 5 memory strategies correctly store and return
attachment metadata passed via ``add_message(..., metadata=...)``,
and that the agent wiring passes attachment info to memory.
"""

from __future__ import annotations

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.modes.base_mode import (
    build_attachment_metadata,
    build_attachment_prefix,
)
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory import (
    FullHistoryMemory,
    SlidingWindowMemory,
    SummaryMemory,
    SummaryWindowMemory,
    TokenBudgetMemory,
)

from nucleusiq.tests.conftest import make_test_prompt

# ================================================================== #
# Helper utilities                                                     #
# ================================================================== #


class TestBuildAttachmentMetadata:
    def test_none_attachments(self) -> None:
        assert build_attachment_metadata(None) is None

    def test_empty_list(self) -> None:
        assert build_attachment_metadata([]) is None

    def test_single_text_attachment(self) -> None:
        atts = [Attachment(type=AttachmentType.TEXT, data="hello", name="readme.txt")]
        meta = build_attachment_metadata(atts)
        assert meta is not None
        assert len(meta["attachments"]) == 1
        entry = meta["attachments"][0]
        assert entry["type"] == "text"
        assert entry["name"] == "readme.txt"
        assert entry["size"] == 5

    def test_mixed_attachments(self) -> None:
        atts = [
            Attachment(type=AttachmentType.TEXT, data="x" * 100, name="doc.txt"),
            Attachment(
                type=AttachmentType.IMAGE_URL, data="https://example.com/img.png"
            ),
            Attachment(type=AttachmentType.PDF, data=b"%PDF-fake", name="report.pdf"),
        ]
        meta = build_attachment_metadata(atts)
        assert meta is not None
        assert len(meta["attachments"]) == 3
        types = [e["type"] for e in meta["attachments"]]
        assert types == ["text", "image_url", "pdf"]

    def test_no_name_omitted(self) -> None:
        atts = [
            Attachment(type=AttachmentType.IMAGE_URL, data="https://example.com/x.png")
        ]
        meta = build_attachment_metadata(atts)
        assert "name" not in meta["attachments"][0]


class TestBuildAttachmentPrefix:
    def test_none_attachments(self) -> None:
        assert build_attachment_prefix(None) == ""

    def test_empty_list(self) -> None:
        assert build_attachment_prefix([]) == ""

    def test_single_small_file(self) -> None:
        atts = [Attachment(type=AttachmentType.TEXT, data="hi", name="note.txt")]
        prefix = build_attachment_prefix(atts)
        assert prefix.startswith("[Attached:")
        assert "note.txt" in prefix
        assert "text" in prefix

    def test_large_file_shows_size(self) -> None:
        atts = [Attachment(type=AttachmentType.TEXT, data="x" * 50_000, name="big.txt")]
        prefix = build_attachment_prefix(atts)
        assert "KB" in prefix

    def test_multiple_files(self) -> None:
        atts = [
            Attachment(type=AttachmentType.TEXT, data="a", name="a.txt"),
            Attachment(type=AttachmentType.PDF, data=b"pdf", name="b.pdf"),
        ]
        prefix = build_attachment_prefix(atts)
        assert "a.txt" in prefix
        assert "b.pdf" in prefix


# ================================================================== #
# All 5 strategies: metadata round-trip                                #
# ================================================================== #

METADATA = {"attachments": [{"name": "report.pdf", "type": "pdf", "size": 32000}]}


class TestFullHistoryMetadata:
    def test_stores_metadata(self) -> None:
        mem = FullHistoryMemory()
        mem.add_message("user", "Summarize the report", metadata=METADATA)
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["content"] == "Summarize the report"
        assert ctx[0]["metadata"] == METADATA

    def test_no_metadata_key_when_absent(self) -> None:
        mem = FullHistoryMemory()
        mem.add_message("user", "Hello")
        ctx = mem.get_context()
        assert "metadata" not in ctx[0]

    def test_export_import_preserves_metadata(self) -> None:
        mem = FullHistoryMemory()
        mem.add_message("user", "data", metadata=METADATA)
        state = mem.export_state()
        mem2 = FullHistoryMemory()
        mem2.import_state(state)
        ctx = mem2.get_context()
        assert ctx[0]["metadata"] == METADATA


class TestSlidingWindowMetadata:
    def test_stores_metadata(self) -> None:
        mem = SlidingWindowMemory(window_size=5)
        mem.add_message("user", "Analyze", metadata=METADATA)
        ctx = mem.get_context()
        assert ctx[0]["metadata"] == METADATA

    def test_evicted_messages_lose_metadata(self) -> None:
        mem = SlidingWindowMemory(window_size=2)
        mem.add_message("user", "old", metadata=METADATA)
        mem.add_message("assistant", "response")
        mem.add_message("user", "new")
        ctx = mem.get_context()
        assert len(ctx) == 2
        assert "metadata" not in ctx[0] or ctx[0].get("content") != "old"


class TestTokenBudgetMetadata:
    def test_stores_metadata(self) -> None:
        mem = TokenBudgetMemory(max_tokens=4096)
        mem.add_message("user", "Check this", metadata=METADATA)
        ctx = mem.get_context()
        assert ctx[0]["metadata"] == METADATA


class TestSummaryMemoryMetadata:
    def test_accepts_metadata_kwarg(self) -> None:
        """SummaryMemory collapses content into a summary; metadata is not
        stored separately but passing it must not raise."""
        mem = SummaryMemory()
        mem.add_message("user", "Summarize", metadata=METADATA)
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_async_accepts_metadata_kwarg(self) -> None:
        mem = SummaryMemory()
        await mem.aadd_message("user", "Summarize", metadata=METADATA)
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "system"


class TestSummaryWindowMetadata:
    def test_stores_metadata_in_window(self) -> None:
        mem = SummaryWindowMemory(window_size=5)
        mem.add_message("user", "Analyze", metadata=METADATA)
        ctx = mem.get_context()
        recent = [c for c in ctx if c.get("role") == "user"]
        assert recent[0]["metadata"] == METADATA

    @pytest.mark.asyncio
    async def test_async_stores_metadata(self) -> None:
        mem = SummaryWindowMemory(window_size=5)
        await mem.aadd_message("user", "Analyze", metadata=METADATA)
        ctx = mem.get_context()
        recent = [c for c in ctx if c.get("role") == "user"]
        assert recent[0]["metadata"] == METADATA


# ================================================================== #
# Agent integration: attachment prefix in memory                       #
# ================================================================== #


class TestAgentMemoryFileAware:
    @pytest.mark.asyncio
    async def test_attachment_prefix_stored_in_memory(self) -> None:
        mem = FullHistoryMemory()
        agent = Agent(
            name="MemBot",
            role="Analyst",
            objective="Analyze docs",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            memory=mem,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        task = Task(
            id="mem-1",
            objective="Summarize this document",
            attachments=[
                Attachment(
                    type=AttachmentType.TEXT, data="Content here.", name="doc.txt"
                ),
            ],
        )
        await agent.execute(task)

        ctx = mem.get_context()
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert "[Attached:" in user_msgs[0]["content"]
        assert "doc.txt" in user_msgs[0]["content"]
        assert "metadata" in user_msgs[0]
        assert user_msgs[0]["metadata"]["attachments"][0]["name"] == "doc.txt"

    @pytest.mark.asyncio
    async def test_no_prefix_without_attachments(self) -> None:
        mem = FullHistoryMemory()
        agent = Agent(
            name="PlainBot",
            role="Helper",
            objective="Help",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            memory=mem,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        task = Task(id="mem-2", objective="What is Python?")
        await agent.execute(task)

        ctx = mem.get_context()
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert "[Attached:" not in user_msgs[0]["content"]
        assert "metadata" not in user_msgs[0]
