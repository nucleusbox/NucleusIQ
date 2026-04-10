"""Integration tests: Provider portability — same Agent code, Gemini LLM.

Proves that NucleusIQ's BaseLLM contract lets you swap providers
without changing any Agent or Tool code. These tests mirror the
patterns used with OpenAI/MockLLM throughout the core test suite.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

_repo_root = Path(__file__).resolve().parents[6]
_nucleusiq_core = str(_repo_root / "src" / "nucleusiq")
if _nucleusiq_core not in sys.path:
    sys.path.insert(0, _nucleusiq_core)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq.tools import BaseTool
from nucleusiq_gemini import BaseGemini

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="GEMINI_API_KEY not set"),
]


class AddTool(BaseTool):
    """Reusable tool definition — identical to what OpenAI tests use."""

    def __init__(self):
        super().__init__(name="add", description="Add two numbers")

    async def initialize(self) -> None:
        pass

    async def execute(self, a: int, b: int) -> int:
        return a + b

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        }


def _make_agent(**overrides) -> Agent:
    """Factory mirroring the MockLLM-based _make_agent in core tests."""
    defaults = dict(
        name="TestAgent",
        role="Assistant",
        objective="Help users",
        prompt=ZeroShotPrompt().configure(system="You are a helpful assistant."),
        llm=BaseGemini(model_name="gemini-2.5-flash", temperature=0.0),
        config=AgentConfig(verbose=False),
    )
    defaults.update(overrides)
    return Agent(**defaults)


class TestBaseLLMContractPortability:
    """These tests would pass with ANY BaseLLM implementation."""

    @pytest.mark.asyncio
    async def test_execute_returns_non_none(self):
        agent = _make_agent()
        await agent.initialize()
        result = await agent.execute({"id": "t1", "objective": "Say hello."})
        assert result is not None

    @pytest.mark.asyncio
    async def test_streaming_yields_events(self):
        agent = _make_agent()
        await agent.initialize()
        events = []
        async for event in agent.execute_stream(
            {"id": "t2", "objective": "Say hello."}
        ):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_memory_round_trip(self):
        memory = FullHistoryMemory()
        agent = _make_agent(memory=memory)
        await agent.initialize()

        await agent.execute({"id": "intro", "objective": "My name is Alice."})
        result = await agent.execute({"id": "recall", "objective": "What is my name?"})

        assert result is not None
        assert "alice" in str(result).lower()

    @pytest.mark.asyncio
    async def test_tool_calling_round_trip(self):
        agent = _make_agent(
            tools=[AddTool()],
            prompt=ZeroShotPrompt().configure(
                system=(
                    "You are a calculator. Use the add tool for arithmetic. "
                    "Report the result."
                )
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            {
                "id": "calc",
                "objective": "What is 5 + 3?",
            }
        )
        assert result is not None
        assert "8" in str(result)

    @pytest.mark.asyncio
    async def test_direct_mode_works(self):
        agent = _make_agent(
            config=AgentConfig(
                verbose=False,
                execution_mode=ExecutionMode.DIRECT,
            ),
        )
        await agent.initialize()
        result = await agent.execute({"id": "d1", "objective": "Hello!"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_standard_mode_works(self):
        agent = _make_agent(
            config=AgentConfig(
                verbose=False,
                execution_mode=ExecutionMode.STANDARD,
            ),
        )
        await agent.initialize()
        result = await agent.execute({"id": "s1", "objective": "Hello!"})
        assert result is not None


class TestMultiTurnConversation:
    """Tests that prove multi-turn context flows correctly through Gemini."""

    @pytest.mark.asyncio
    async def test_three_turn_context(self):
        memory = FullHistoryMemory()
        agent = _make_agent(memory=memory)
        await agent.initialize()

        await agent.execute({"id": "t1", "objective": "I have a dog named Max."})
        await agent.execute({"id": "t2", "objective": "Max is a golden retriever."})
        result = await agent.execute({"id": "t3", "objective": "What breed is my dog?"})

        content = str(result).lower()
        assert "golden retriever" in content or "retriever" in content

    @pytest.mark.asyncio
    async def test_streaming_multi_turn(self):
        """Multi-turn conversation works in streaming mode."""
        memory = FullHistoryMemory()
        agent = _make_agent(memory=memory)
        await agent.initialize()

        events1 = []
        async for event in agent.execute_stream(
            {"id": "t1", "objective": "I live in Tokyo."}
        ):
            events1.append(event)
        assert len(events1) > 0

        events2 = []
        async for event in agent.execute_stream(
            {"id": "t2", "objective": "Where do I live?"}
        ):
            events2.append(event)

        all_text = "".join(e.token for e in events2 if e.type == "token" and e.token)
        complete = [e for e in events2 if e.type == "complete"]
        content = complete[0].content if complete else all_text
        assert "tokyo" in content.lower()
