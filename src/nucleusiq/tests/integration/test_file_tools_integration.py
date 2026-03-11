"""Integration tests for built-in file tools in agent tool loop.

Verifies the full pipeline:

    Agent (Standard mode)
      -> MockLLM decides to call a file tool
        -> Tool loop executes the tool (sandboxed)
          -> Result fed back to LLM
            -> LLM produces final answer incorporating tool output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.builtin import (
    DirectoryListTool,
    FileExtractTool,
    FileReadTool,
    FileSearchTool,
)


class FileToolMockLLM(MockLLM):
    """MockLLM that calls a specific file tool with explicit arguments."""

    def __init__(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        super().__init__()
        self._target_tool = tool_name
        self._target_args = tool_args

    def _tool_call_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> MockLLM.LLMResponse:
        fn_call = {
            "name": self._target_tool,
            "arguments": json.dumps(self._target_args),
        }
        tool_calls = [
            {
                "id": "call_file_tool_1",
                "type": "function",
                "function": fn_call,
            }
        ]
        msg = self.Message(content=None, function_call=fn_call, tool_calls=tool_calls)
        return self.LLMResponse([self.Choice(msg)])


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Workspace with sample files for integration tests."""
    (tmp_path / "report.txt").write_text(
        "Q1 Revenue: $1.2M\nQ2 Revenue: $1.5M\nQ3 Revenue: $1.8M\n"
    )
    (tmp_path / "data.csv").write_text(
        "product,sales,region\nWidget,100,US\nGadget,200,EU\nDoohickey,150,APAC\n"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"version": "2.0", "env": "production", "replicas": 3})
    )
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "notes.txt").write_text("Meeting notes: discuss budget\n")
    return tmp_path


def _make_agent(
    workspace: Path,
    tool_name: str,
    tool_args: dict[str, Any],
) -> Agent:
    ws = str(workspace)
    return Agent(
        name="FileAgent",
        role="Data Analyst",
        objective="Analyze files",
        llm=FileToolMockLLM(tool_name, tool_args),
        tools=[
            FileReadTool(workspace_root=ws),
            FileSearchTool(workspace_root=ws),
            DirectoryListTool(workspace_root=ws),
            FileExtractTool(workspace_root=ws),
        ],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )


class TestFileReadToolIntegration:
    @pytest.mark.asyncio
    async def test_agent_reads_file_through_tool_loop(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "file_read", {"path": "report.txt"})
        result = await agent.execute(
            Task(id="t1", objective="Read the report and summarize revenue")
        )
        assert result is not None
        assert "Revenue" in result or "revenue" in result or "function output" in result

    @pytest.mark.asyncio
    async def test_agent_reads_file_with_line_range(self, workspace: Path) -> None:
        agent = _make_agent(
            workspace,
            "file_read",
            {"path": "report.txt", "start_line": 1, "end_line": 2},
        )
        result = await agent.execute(
            Task(id="t2", objective="Read lines 1-2 of the report")
        )
        assert result is not None


class TestFileSearchToolIntegration:
    @pytest.mark.asyncio
    async def test_agent_searches_for_pattern(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "file_search", {"pattern": "Revenue"})
        result = await agent.execute(
            Task(id="t3", objective="Find all revenue mentions")
        )
        assert result is not None
        assert "Revenue" in result or "function output" in result

    @pytest.mark.asyncio
    async def test_agent_searches_specific_file(self, workspace: Path) -> None:
        agent = _make_agent(
            workspace, "file_search", {"pattern": "Widget", "path": "data.csv"}
        )
        result = await agent.execute(Task(id="t4", objective="Find Widget in data.csv"))
        assert result is not None


class TestDirectoryListToolIntegration:
    @pytest.mark.asyncio
    async def test_agent_lists_workspace(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "directory_list", {})
        result = await agent.execute(
            Task(id="t5", objective="List all files in the workspace")
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_lists_with_glob(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "directory_list", {"pattern": "*.csv"})
        result = await agent.execute(Task(id="t6", objective="List all CSV files"))
        assert result is not None


class TestFileExtractToolIntegration:
    @pytest.mark.asyncio
    async def test_agent_extracts_csv(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "file_extract", {"path": "data.csv"})
        result = await agent.execute(Task(id="t7", objective="Extract data from CSV"))
        assert result is not None
        assert "product" in result or "function output" in result

    @pytest.mark.asyncio
    async def test_agent_extracts_json(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "file_extract", {"path": "config.json"})
        result = await agent.execute(Task(id="t8", objective="Read JSON config"))
        assert result is not None


class TestFileToolsStreaming:
    @pytest.mark.asyncio
    async def test_stream_with_file_tool(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "file_read", {"path": "report.txt"})
        events = []
        async for event in agent.execute_stream(
            Task(id="t9", objective="Read the report and summarize")
        ):
            events.append(event)
        types = {e.type for e in events}
        assert "complete" in types


class TestFileToolsUsageTracking:
    @pytest.mark.asyncio
    async def test_usage_tracked_after_tool_loop(self, workspace: Path) -> None:
        agent = _make_agent(workspace, "file_read", {"path": "report.txt"})
        await agent.execute(Task(id="t10", objective="Read and analyze the report"))
        usage = agent.last_usage
        assert hasattr(usage, "call_count")
        assert hasattr(usage, "by_purpose")
        assert hasattr(usage, "total")
