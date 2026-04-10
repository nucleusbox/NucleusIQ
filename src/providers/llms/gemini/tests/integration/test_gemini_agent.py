"""Integration tests: NucleusIQ Agent with Gemini provider.

Tests the full Agent framework wired with a real Gemini LLM.
Mirrors patterns from the memory integration test but uses Gemini
instead of OpenAI — proving provider portability.

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
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.memory.sliding_window import SlidingWindowMemory
from nucleusiq.tools import BaseTool
from nucleusiq_gemini import BaseGemini, GeminiLLMParams

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="GEMINI_API_KEY not set"),
]

INTRO_MSG = "Hello My name is Brijesh and I am a creator of NucleusIQ"
RECALL_MSG = "Who is the creator of NucleusIQ?"


def _make_gemini_llm():
    return BaseGemini(model_name="gemini-2.5-flash", temperature=0.0)


class MockCalculatorTool(BaseTool):
    """Calculator tool for agent tool-calling tests."""

    def __init__(self):
        super().__init__(name="add", description="Add two numbers together")

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


# ====================================================================== #
# Basic Agent execution                                                    #
# ====================================================================== #


class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_agent_simple_execute(self):
        """Agent.execute() works with Gemini LLM."""
        llm = _make_gemini_llm()
        agent = Agent(
            name="GeminiBot",
            role="Assistant",
            objective="Answer questions.",
            prompt=ZeroShotPrompt().configure(system="You are a helpful assistant."),
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()

        result = await agent.execute(
            {
                "id": "q1",
                "objective": "What is the capital of France?",
            }
        )
        assert result is not None
        content = str(result)
        assert "paris" in content.lower()

    @pytest.mark.asyncio
    async def test_agent_with_llm_params(self):
        """Agent respects GeminiLLMParams for per-call tuning."""
        llm = _make_gemini_llm()
        params = GeminiLLMParams(temperature=0.0, max_output_tokens=50)

        agent = Agent(
            name="GeminiBot",
            role="Assistant",
            objective="Answer briefly.",
            prompt=ZeroShotPrompt().configure(system="You are concise."),
            llm=llm,
            config=AgentConfig(verbose=False, llm_params=params),
        )
        await agent.initialize()

        result = await agent.execute(
            {
                "id": "q2",
                "objective": "Say hello.",
            }
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_system_prompt(self):
        """Agent system prompt is passed as system message to Gemini."""
        llm = _make_gemini_llm()
        agent = Agent(
            name="FrenchBot",
            role="Translator",
            objective="Translate to French.",
            prompt=ZeroShotPrompt().configure(
                system="You always respond in French, no matter what language the user uses."
            ),
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()

        result = await agent.execute(
            {
                "id": "q3",
                "objective": "Hello, how are you?",
            }
        )
        assert result is not None


# ====================================================================== #
# Agent + Memory with Gemini                                               #
# ====================================================================== #


class TestAgentMemory:
    @pytest.mark.asyncio
    async def test_full_history_memory_recall(self):
        """FullHistoryMemory + Gemini: agent recalls info from previous turn."""
        llm = _make_gemini_llm()
        memory = FullHistoryMemory()

        agent = Agent(
            name="MemoryBot",
            role="Assistant",
            objective="Answer using conversation history.",
            prompt=ZeroShotPrompt().configure(
                system="You are a helpful assistant. Use the conversation history."
            ),
            llm=llm,
            memory=memory,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()

        result1 = await agent.execute({"id": "intro", "objective": INTRO_MSG})
        assert result1 is not None

        result2 = await agent.execute({"id": "recall", "objective": RECALL_MSG})
        assert result2 is not None
        assert "brijesh" in str(result2).lower(), (
            f"Agent failed to recall 'Brijesh'. Got: {result2}"
        )

    @pytest.mark.asyncio
    async def test_sliding_window_memory_recall(self):
        """SlidingWindowMemory + Gemini: agent recalls within window."""
        llm = _make_gemini_llm()
        memory = SlidingWindowMemory(window_size=10)

        agent = Agent(
            name="MemoryBot",
            role="Assistant",
            objective="Answer using conversation history.",
            prompt=ZeroShotPrompt().configure(
                system="You are a helpful assistant. Use the conversation history."
            ),
            llm=llm,
            memory=memory,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()

        result1 = await agent.execute({"id": "intro", "objective": INTRO_MSG})
        assert result1 is not None

        result2 = await agent.execute({"id": "recall", "objective": RECALL_MSG})
        assert result2 is not None
        assert "brijesh" in str(result2).lower(), (
            f"SlidingWindowMemory: Agent failed to recall 'Brijesh'. Got: {result2}"
        )


# ====================================================================== #
# Agent + Streaming with Gemini                                            #
# ====================================================================== #


class TestAgentStreaming:
    @pytest.mark.asyncio
    async def test_agent_execute_stream(self):
        """Agent.execute_stream() produces StreamEvent objects with Gemini."""
        llm = _make_gemini_llm()
        agent = Agent(
            name="StreamBot",
            role="Assistant",
            objective="Answer questions.",
            prompt=ZeroShotPrompt().configure(system="You are a helpful assistant."),
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()

        events = []
        async for event in agent.execute_stream(
            {
                "id": "stream_q",
                "objective": "Count from 1 to 3.",
            }
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "token" in types or "complete" in types, (
            f"Expected streaming events, got: {types}"
        )

    @pytest.mark.asyncio
    async def test_agent_stream_event_ordering(self):
        """Streaming events follow the correct order: llm_start → tokens → complete → llm_end."""
        llm = _make_gemini_llm()
        agent = Agent(
            name="StreamBot",
            role="Assistant",
            objective="Answer questions.",
            prompt=ZeroShotPrompt().configure(system="You are concise."),
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()

        events = []
        async for event in agent.execute_stream(
            {
                "id": "order_q",
                "objective": "Say hi.",
            }
        ):
            events.append(event)

        types = [e.type for e in events]
        if "llm_start" in types:
            start_idx = types.index("llm_start")
            assert any(t in types[start_idx:] for t in ["token", "complete"])


# ====================================================================== #
# Agent + Tools with Gemini                                                #
# ====================================================================== #


class TestAgentTools:
    @pytest.mark.asyncio
    async def test_agent_tool_calling(self):
        """Agent uses Gemini to call a custom tool and returns the result."""
        llm = _make_gemini_llm()
        calculator = MockCalculatorTool()

        agent = Agent(
            name="CalcBot",
            role="Calculator",
            objective="Perform calculations using the add tool.",
            prompt=ZeroShotPrompt().configure(
                system=(
                    "You are a calculator assistant. "
                    "When asked to add numbers, always use the add tool. "
                    "Report the tool result to the user."
                )
            ),
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()

        result = await agent.execute(
            {
                "id": "calc",
                "objective": "What is 7 + 13?",
            }
        )
        assert result is not None
        content = str(result)
        assert "20" in content, f"Expected '20' in response. Got: {content}"


# ====================================================================== #
# Execution modes with Gemini                                              #
# ====================================================================== #


class TestExecutionModes:
    @pytest.mark.asyncio
    async def test_direct_mode(self):
        """Direct mode: single LLM call, no tool loop."""
        llm = _make_gemini_llm()
        agent = Agent(
            name="DirectBot",
            role="Assistant",
            objective="Answer directly.",
            prompt=ZeroShotPrompt().configure(system="You are concise."),
            llm=llm,
            config=AgentConfig(
                verbose=False,
                execution_mode=ExecutionMode.DIRECT,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            {
                "id": "direct_q",
                "objective": "What is 2+2?",
            }
        )
        assert result is not None
        assert "4" in str(result)

    @pytest.mark.asyncio
    async def test_standard_mode(self):
        """Standard mode: supports tool calling loop."""
        llm = _make_gemini_llm()
        agent = Agent(
            name="StandardBot",
            role="Assistant",
            objective="Answer questions.",
            prompt=ZeroShotPrompt().configure(system="You are helpful."),
            llm=llm,
            config=AgentConfig(
                verbose=False,
                execution_mode=ExecutionMode.STANDARD,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            {
                "id": "standard_q",
                "objective": "What is the capital of Japan?",
            }
        )
        assert result is not None
        assert "tokyo" in str(result).lower()
