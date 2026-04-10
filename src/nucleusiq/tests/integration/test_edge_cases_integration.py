"""Integration edge-case tests for v0.4.0.

Covers scenarios missing from the base integration suite:
  - Tool error propagation through agent loop
  - IMAGE_BASE64, PDF, FILE_BASE64 attachment types in e2e flow
  - Attachment + tools in the same task
  - Multi-turn memory with attachments across turns
  - AutonomousMode streaming stores task in memory
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory import FullHistoryMemory
from nucleusiq.tools.builtin import FileReadTool

from nucleusiq.tests.conftest import make_test_prompt

# ================================================================== #
# Helpers                                                              #
# ================================================================== #


class ErrorToolMockLLM(MockLLM):
    """MockLLM that asks to read a file that doesn't exist."""

    def _tool_call_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> MockLLM.LLMResponse:
        fn_call = {
            "name": "file_read",
            "arguments": json.dumps({"path": "nonexistent_file.txt"}),
        }
        tool_calls = [{"id": "call_err_1", "type": "function", "function": fn_call}]
        msg = self.Message(content=None, function_call=fn_call, tool_calls=tool_calls)
        return self.LLMResponse([self.Choice(msg)])


# ================================================================== #
# 1. Tool error propagation                                            #
# ================================================================== #


class TestToolErrorPropagation:
    @pytest.mark.asyncio
    async def test_agent_handles_tool_error_gracefully(self, tmp_path: Path) -> None:
        agent = Agent(
            name="ErrAgent",
            role="Tester",
            objective="Test error handling",
            prompt=make_test_prompt(),
            llm=ErrorToolMockLLM(),
            tools=[FileReadTool(workspace_root=str(tmp_path))],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        result = await agent.execute(Task(id="err-1", objective="Read a missing file"))
        assert result is not None

    @pytest.mark.asyncio
    async def test_stream_handles_tool_error(self, tmp_path: Path) -> None:
        agent = Agent(
            name="ErrStream",
            role="Tester",
            objective="Test error handling",
            prompt=make_test_prompt(),
            llm=ErrorToolMockLLM(),
            tools=[FileReadTool(workspace_root=str(tmp_path))],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        events = []
        async for event in agent.execute_stream(
            Task(id="err-2", objective="Read a missing file")
        ):
            events.append(event)
        types = {e.type for e in events}
        assert "tool_call_start" in types or "complete" in types or "error" in types


# ================================================================== #
# 2. All attachment types in integration                               #
# ================================================================== #


class TestAllAttachmentTypesIntegration:
    @pytest.mark.asyncio
    async def test_image_base64_attachment(self) -> None:
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        agent = Agent(
            name="ImgB64Bot",
            role="Vision",
            objective="Describe images",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        task = Task(
            id="int-imgb64",
            objective="Describe this image",
            attachments=[
                Attachment(
                    type=AttachmentType.IMAGE_BASE64,
                    data=png_header,
                    name="photo.png",
                    mime_type="image/png",
                ),
            ],
        )
        result = await agent.execute(task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_file_base64_text_attachment(self) -> None:
        content = base64.b64encode(b"Hello from base64").decode()
        agent = Agent(
            name="FB64Bot",
            role="Reader",
            objective="Read files",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        task = Task(
            id="int-fb64",
            objective="Read this file",
            attachments=[
                Attachment(
                    type=AttachmentType.FILE_BASE64,
                    data=content,
                    name="encoded.txt",
                ),
            ],
        )
        result = await agent.execute(task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_file_base64_binary_attachment(self) -> None:
        content = base64.b64encode(b"\x00\x01\x02\xff" * 50).decode()
        agent = Agent(
            name="BinB64Bot",
            role="Analyst",
            objective="Analyze binary",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        task = Task(
            id="int-binb64",
            objective="Inspect this binary",
            attachments=[
                Attachment(
                    type=AttachmentType.FILE_BASE64,
                    data=content,
                    name="data.bin",
                ),
            ],
        )
        result = await agent.execute(task)
        assert result is not None

    @pytest.mark.asyncio
    async def test_file_bytes_text_decodable(self) -> None:
        agent = Agent(
            name="BytesBot",
            role="Reader",
            objective="Read bytes",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        task = Task(
            id="int-bytes",
            objective="Read this data",
            attachments=[
                Attachment(
                    type=AttachmentType.FILE_BYTES,
                    data=b"Plain text as bytes",
                    name="plain.txt",
                ),
            ],
        )
        result = await agent.execute(task)
        assert result is not None


# ================================================================== #
# 3. Multi-turn memory with attachments                                #
# ================================================================== #


class TestMultiTurnMemoryWithAttachments:
    @pytest.mark.asyncio
    async def test_attachment_context_persists_across_turns(self) -> None:
        mem = FullHistoryMemory()
        agent = Agent(
            name="MemBot",
            role="Analyst",
            objective="Analyze over multiple turns",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            memory=mem,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        task1 = Task(
            id="turn-1",
            objective="Read the attached document",
            attachments=[
                Attachment(
                    type=AttachmentType.TEXT,
                    data="Q1 revenue: $1.2M",
                    name="report.txt",
                ),
            ],
        )
        await agent.execute(task1)

        task2 = Task(id="turn-2", objective="What was in the report?")
        await agent.execute(task2)

        ctx = mem.get_context()
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert len(user_msgs) >= 2
        assert "[Attached:" in user_msgs[0]["content"]
        assert "report.txt" in user_msgs[0]["content"]
        assert "[Attached:" not in user_msgs[1]["content"]

    @pytest.mark.asyncio
    async def test_metadata_preserved_in_memory_across_turns(self) -> None:
        mem = FullHistoryMemory()
        agent = Agent(
            name="MetaBot",
            role="Analyst",
            objective="Analyze",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            memory=mem,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        task = Task(
            id="meta-turn",
            objective="Check this image",
            attachments=[
                Attachment(
                    type=AttachmentType.IMAGE_URL,
                    data="https://example.com/photo.jpg",
                    name="photo.jpg",
                ),
            ],
        )
        await agent.execute(task)

        ctx = mem.get_context()
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert len(user_msgs) >= 1
        meta = user_msgs[0].get("metadata")
        assert meta is not None
        assert "attachments" in meta


# ================================================================== #
# 4. Attachment + tools combined                                       #
# ================================================================== #


class FileReadMockLLM(MockLLM):
    """MockLLM that calls file_read on the first call, then answers."""

    def __init__(self, target_path: str) -> None:
        super().__init__()
        self._target_path = target_path

    def _tool_call_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> MockLLM.LLMResponse:
        fn_call = {
            "name": "file_read",
            "arguments": json.dumps({"path": self._target_path}),
        }
        tool_calls = [{"id": "call_combo_1", "type": "function", "function": fn_call}]
        msg = self.Message(content=None, function_call=fn_call, tool_calls=tool_calls)
        return self.LLMResponse([self.Choice(msg)])


class TestAttachmentPlusTools:
    @pytest.mark.asyncio
    async def test_attachment_and_tool_in_same_task(self, tmp_path: Path) -> None:
        (tmp_path / "extra.txt").write_text("Extra context from filesystem\n")

        agent = Agent(
            name="ComboBot",
            role="Analyst",
            objective="Analyze all inputs",
            prompt=make_test_prompt(),
            llm=FileReadMockLLM("extra.txt"),
            tools=[FileReadTool(workspace_root=str(tmp_path))],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )

        task = Task(
            id="combo-1",
            objective="Compare the attached data with the file on disk",
            attachments=[
                Attachment(
                    type=AttachmentType.TEXT,
                    data="Inline data from attachment",
                    name="inline.txt",
                ),
            ],
        )
        result = await agent.execute(task)
        assert result is not None


# ================================================================== #
# 5. AutonomousMode streaming + memory                                 #
# ================================================================== #


class TestAutonomousStreamingMemory:
    @pytest.mark.asyncio
    async def test_autonomous_stream_stores_task_in_memory(self) -> None:
        mem = FullHistoryMemory()
        agent = Agent(
            name="AutoBot",
            role="Orchestrator",
            objective="Orchestrate tasks",
            prompt=make_test_prompt(),
            llm=MockLLM(),
            memory=mem,
            config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS),
        )
        task = Task(
            id="auto-stream",
            objective="Analyze this document thoroughly",
            attachments=[
                Attachment(
                    type=AttachmentType.TEXT,
                    data="Important data here.",
                    name="doc.txt",
                ),
            ],
        )
        events = []
        async for event in agent.execute_stream(task):
            events.append(event)

        ctx = mem.get_context()
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert "[Attached:" in user_msgs[0]["content"]
        assert "doc.txt" in user_msgs[0]["content"]
