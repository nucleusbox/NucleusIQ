"""Agent + ExecutionTracer integration (non-streaming)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

src_dir = Path(__file__).resolve().parent.parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tests.conftest import make_test_prompt
from nucleusiq.tools import BaseTool


class _CalcTool(BaseTool):
    def __init__(self):
        super().__init__(name="add", description="add")

    async def initialize(self) -> None:
        pass

    async def execute(self, a: int, b: int) -> int:
        return a + b

    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        }


@pytest.mark.asyncio
async def test_standard_mode_populates_llm_and_tool_traces():
    agent = Agent(
        name="T",
        role="r",
        objective="o",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=[_CalcTool()],
        config=AgentConfig(
            execution_mode="standard", verbose=False, enable_tracing=True
        ),
    )
    await agent.initialize()
    result = await agent.execute(Task(id="1", objective="add 2 and 3"))

    assert result.status.value == "success"
    assert len(result.llm_calls) >= 2
    assert result.llm_calls[0].purpose == "main"
    assert result.llm_calls[0].duration_ms >= 0
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "add"
    assert result.tool_calls[0].success is True
    assert result.tool_calls[0].round >= 1


@pytest.mark.asyncio
async def test_direct_mode_tool_round_traced():
    agent = Agent(
        name="D",
        role="r",
        objective="o",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=[_CalcTool()],
        config=AgentConfig(execution_mode="direct", verbose=False, enable_tracing=True),
    )
    await agent.initialize()
    result = await agent.execute(Task(id="2", objective="add 1 and 2"))

    assert result.status.value == "success"
    assert len(result.llm_calls) >= 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].round == 1


@pytest.mark.asyncio
async def test_tracing_disabled_by_default_returns_empty_traces():
    """enable_tracing=False (default) means AgentResult trace fields are empty."""
    agent = Agent(
        name="NoTrace",
        role="r",
        objective="o",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        tools=[_CalcTool()],
        config=AgentConfig(execution_mode="standard", verbose=False),
    )
    await agent.initialize()
    result = await agent.execute(Task(id="3", objective="add 5 and 6"))

    assert result.status.value == "success"
    assert result.llm_calls == ()
    assert result.tool_calls == ()
    assert result.warnings == ()
    assert str(result) != ""
