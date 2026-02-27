"""Tests for execution mode streaming (run_stream).

Covers:
    1. BaseExecutionMode.run_stream() — default fallback
    2. BaseExecutionMode._streaming_tool_call_loop() — shared loop logic
    3. DirectMode.run_stream() — fast path, optional tools
    4. StandardMode.run_stream() — tool loop + memory persistence
    5. AutonomousMode.run_stream() — simple/complex paths + critic events
    6. Error handling across all modes
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.direct_mode import DirectMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.streaming.events import StreamEvent
from nucleusiq.tools import BaseTool

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


class MockCalculatorTool(BaseTool):
    """Simple calculator tool for testing tool-call streaming."""

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
    defaults = dict(
        name="TestAgent",
        role="Assistant",
        objective="Help users",
        narrative="Test agent",
        llm=MockLLM(),
    )
    defaults.update(overrides)
    return Agent(**defaults)


def _make_task(objective: str = "Say hello") -> Task:
    return Task(id="t1", objective=objective)


async def _collect(async_gen) -> list[StreamEvent]:
    """Collect all events from an async generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# BaseExecutionMode — default run_stream() fallback
# ═══════════════════════════════════════════════════════════════════════════════


class ConcreteMode(BaseExecutionMode):
    """Minimal mode for testing the base class fallback."""

    async def run(self, agent, task):
        return "fallback result"


class TestBaseExecutionModeStreamFallback:
    @pytest.mark.asyncio
    async def test_fallback_yields_single_complete(self):
        mode = ConcreteMode()
        agent = _make_agent()
        task = _make_task()
        events = await _collect(mode.run_stream(agent, task))

        assert len(events) == 1
        assert events[0].type == "complete"
        assert events[0].content == "fallback result"

    @pytest.mark.asyncio
    async def test_fallback_none_result(self):
        class NoneMode(BaseExecutionMode):
            async def run(self, agent, task):
                return None

        mode = NoneMode()
        agent = _make_agent()
        events = await _collect(mode.run_stream(agent, _make_task()))

        assert len(events) == 1
        assert events[0].content == ""


# ═══════════════════════════════════════════════════════════════════════════════
# BaseExecutionMode._streaming_tool_call_loop() — shared loop tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamingToolCallLoop:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """No tools — loop should yield LLM_CALL_START, TOKEN(s), LLM_CALL_END, COMPLETE."""
        agent = _make_agent()
        mode = DirectMode()
        messages = mode.build_messages(agent, _make_task())

        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        types = [e.type for e in events]
        assert types[0] == "llm_start"
        assert "token" in types
        assert "llm_end" in types
        assert types[-1] == "complete"
        assert events[-1].content  # non-empty

    @pytest.mark.asyncio
    async def test_tool_call_detected_and_executed(self):
        """When LLM returns tool calls, the loop should execute them and loop."""
        tool = MockCalculatorTool()
        agent = _make_agent(tools=[tool])
        mode = DirectMode()
        mode._ensure_executor(agent)
        messages = mode.build_messages(agent, _make_task(objective="Add 2 and 3"))
        tool_specs = agent.llm.convert_tool_specs(agent.tools)

        events = await _collect(
            mode._streaming_tool_call_loop(
                agent, messages, tool_specs, max_tool_calls=5
            )
        )

        types = [e.type for e in events]

        assert "llm_start" in types
        assert "llm_end" in types
        assert "tool_start" in types
        assert "tool_end" in types
        assert types[-1] == "complete"

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) >= 1
        assert tool_starts[0].tool_name == "add"

        tool_ends = [e for e in events if e.type == "tool_end"]
        assert len(tool_ends) >= 1
        assert tool_ends[0].tool_result is not None

    @pytest.mark.asyncio
    async def test_empty_response_retries_once(self):
        """Empty LLM response → retry once → then error."""
        agent = _make_agent()
        mode = DirectMode()

        call_count = 0

        async def mock_stream(**kwargs):
            nonlocal call_count
            call_count += 1
            yield StreamEvent.complete_event("")

        agent.llm.call_stream = mock_stream

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        assert call_count == 2  # original + one retry
        assert events[-1].type == "error"

    @pytest.mark.asyncio
    async def test_error_event_stops_loop(self):
        """LLM stream yielding ERROR should stop the loop immediately."""
        agent = _make_agent()
        mode = DirectMode()

        async def error_stream(**kwargs):
            yield StreamEvent.error_event("LLM failed")

        agent.llm.call_stream = error_stream

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        types = [e.type for e in events]
        assert "llm_start" in types
        assert "llm_end" in types
        assert "error" in types

    @pytest.mark.asyncio
    async def test_max_tool_calls_limit(self):
        """Loop should stop when max_tool_calls is reached."""
        agent = _make_agent()
        mode = DirectMode()

        round_count = 0

        async def always_tool_stream(**kwargs):
            nonlocal round_count
            round_count += 1
            tool_calls = [
                {
                    "id": f"call_{round_count}",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }
            ]
            yield StreamEvent.complete_event("", metadata={"tool_calls": tool_calls})

        agent.llm.call_stream = always_tool_stream

        async def noop_tool(tc):
            return "ok"

        mode.call_tool = AsyncMock(return_value="ok")

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=2)
        )

        tool_ends = [e for e in events if e.type == "tool_end"]
        assert len(tool_ends) == 2
        assert events[-1].type == "error"
        assert "Maximum" in events[-1].message


# ═══════════════════════════════════════════════════════════════════════════════
# DirectMode.run_stream()
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectModeRunStream:
    @pytest.mark.asyncio
    async def test_basic_text_streaming(self):
        agent = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.DIRECT))
        task = _make_task("Tell me a joke")
        mode = DirectMode()

        events = await _collect(mode.run_stream(agent, task))

        types = [e.type for e in events]
        assert "llm_start" in types
        assert "token" in types
        assert "llm_end" in types
        assert "complete" in types
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_no_llm_echo_fallback(self):
        agent = _make_agent(llm=None)
        mode = DirectMode()

        events = await _collect(mode.run_stream(agent, _make_task("hello")))

        assert len(events) == 1
        assert events[0].type == "complete"
        assert "hello" in events[0].content

    @pytest.mark.asyncio
    async def test_with_tool_calls(self):
        tool = MockCalculatorTool()
        agent = _make_agent(
            tools=[tool],
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        mode = DirectMode()
        task = _make_task("Add 5 and 7")

        events = await _collect(mode.run_stream(agent, task))

        types = [e.type for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert "complete" in types
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_exception_yields_error_event(self):
        agent = _make_agent()
        mode = DirectMode()

        async def exploding_stream(**kwargs):
            raise RuntimeError("boom")
            yield  # make it a generator  # noqa: E501

        agent.llm.call_stream = exploding_stream

        events = await _collect(mode.run_stream(agent, _make_task()))

        assert any(e.type == "error" for e in events)
        assert agent.state == AgentState.ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# StandardMode.run_stream()
# ═══════════════════════════════════════════════════════════════════════════════


class TestStandardModeRunStream:
    @pytest.mark.asyncio
    async def test_basic_text_streaming(self):
        agent = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.STANDARD))
        mode = StandardMode()

        events = await _collect(mode.run_stream(agent, _make_task("What is 2+2?")))

        types = [e.type for e in events]
        assert "llm_start" in types
        assert "token" in types
        assert "complete" in types
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_with_tool_loop(self):
        tool = MockCalculatorTool()
        agent = _make_agent(
            tools=[tool],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        mode = StandardMode()

        events = await _collect(mode.run_stream(agent, _make_task("Add 10 and 20")))

        types = [e.type for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        assert "complete" in types
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_memory_persistence(self):
        memory = FullHistoryMemory()
        agent = _make_agent(
            memory=memory,
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        mode = StandardMode()

        events = await _collect(mode.run_stream(agent, _make_task("hello")))

        complete_events = [e for e in events if e.type == "complete"]
        assert len(complete_events) == 1
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_no_llm_echo(self):
        agent = _make_agent(llm=None)
        mode = StandardMode()

        events = await _collect(mode.run_stream(agent, _make_task("echo me")))

        assert len(events) == 1
        assert events[0].type == "complete"
        assert "echo me" in events[0].content

    @pytest.mark.asyncio
    async def test_exception_yields_error(self):
        agent = _make_agent()
        mode = StandardMode()

        async def broken(**kwargs):
            raise ValueError("model exploded")
            yield  # noqa: E501

        agent.llm.call_stream = broken

        events = await _collect(mode.run_stream(agent, _make_task()))

        assert any(e.type == "error" for e in events)
        assert agent.state == AgentState.ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousMode.run_stream() — simple path
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutonomousModeStreamSimple:
    @pytest.mark.asyncio
    async def test_simple_task_emits_thinking_and_tokens(self):
        """Simple task should emit THINKING (analysis) + LLM stream + Critic THINKING."""
        agent = _make_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=1,
            ),
        )
        mode = AutonomousMode()

        events = await _collect(mode.run_stream(agent, _make_task("Say hi")))

        types = [e.type for e in events]
        assert "thinking" in types
        assert "llm_start" in types
        assert "token" in types
        assert "complete" in types

    @pytest.mark.asyncio
    async def test_no_llm_falls_back_to_standard_stream(self):
        agent = _make_agent(llm=None)
        mode = AutonomousMode()

        events = await _collect(mode.run_stream(agent, _make_task("test")))

        assert any(e.type == "complete" for e in events)


# ═══════════════════════════════════════════════════════════════════════════════
# Event ordering and structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventOrdering:
    @pytest.mark.asyncio
    async def test_llm_start_before_tokens_before_llm_end(self):
        """LLM_CALL_START must precede TOKEN events, which precede LLM_CALL_END."""
        agent = _make_agent()
        mode = DirectMode()

        events = await _collect(mode.run_stream(agent, _make_task()))

        indices = {t: [] for t in ["llm_start", "token", "llm_end"]}
        for i, e in enumerate(events):
            if e.type in indices:
                indices[e.type].append(i)

        assert indices["llm_start"]
        assert indices["token"]
        assert indices["llm_end"]

        assert max(indices["llm_start"]) < min(indices["token"])
        assert max(indices["token"]) < min(indices["llm_end"])

    @pytest.mark.asyncio
    async def test_tool_events_between_llm_rounds(self):
        """TOOL_CALL_START/END should appear between two LLM rounds."""
        tool = MockCalculatorTool()
        agent = _make_agent(tools=[tool])
        mode = DirectMode()

        events = await _collect(mode.run_stream(agent, _make_task("Add 1 and 2")))

        types = [e.type for e in events]

        llm_start_indices = [i for i, t in enumerate(types) if t == "llm_start"]
        tool_start_indices = [i for i, t in enumerate(types) if t == "tool_start"]

        if len(llm_start_indices) >= 2 and tool_start_indices:
            first_llm_end = types.index("llm_end")
            assert tool_start_indices[0] > first_llm_end

    @pytest.mark.asyncio
    async def test_complete_event_has_content(self):
        """The final COMPLETE event must carry content."""
        agent = _make_agent()
        mode = StandardMode()

        events = await _collect(mode.run_stream(agent, _make_task("test content")))

        completes = [e for e in events if e.type == "complete"]
        assert len(completes) == 1
        assert completes[0].content
        assert len(completes[0].content) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# call_llm_stream() helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestCallLlmStream:
    @pytest.mark.asyncio
    async def test_forwards_all_events(self):
        agent = _make_agent()
        mode = DirectMode()

        call_kwargs = mode.build_call_kwargs(
            agent, mode.build_messages(agent, _make_task()), None
        )

        events = await _collect(mode.call_llm_stream(agent, call_kwargs))

        assert any(e.type == "token" for e in events)
        assert any(e.type == "complete" for e in events)


# ═══════════════════════════════════════════════════════════════════════════════
# _streaming_tool_call_loop — edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamingToolLoopEdgeCases:
    @pytest.mark.asyncio
    async def test_tool_call_with_invalid_json_args(self):
        """Tool call with malformed JSON arguments should still execute."""
        agent = _make_agent()
        mode = DirectMode()

        call_num = 0

        async def bad_args_then_content(**kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                tool_calls = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "noop", "arguments": "not-json"},
                    }
                ]
                yield StreamEvent.complete_event(
                    "", metadata={"tool_calls": tool_calls}
                )
            else:
                yield StreamEvent.token_event("done")
                yield StreamEvent.complete_event("done")

        agent.llm.call_stream = bad_args_then_content
        mode.call_tool = AsyncMock(return_value="ok")

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        starts = [e for e in events if e.type == "tool_start"]
        assert len(starts) == 1
        assert starts[0].tool_args == {}

    @pytest.mark.asyncio
    async def test_tool_call_with_missing_name_skipped(self):
        """Tool call without a name should be silently skipped."""
        agent = _make_agent()
        mode = DirectMode()

        call_num = 0

        async def nameless_then_content(**kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                tool_calls = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "", "arguments": "{}"},
                    }
                ]
                yield StreamEvent.complete_event(
                    "", metadata={"tool_calls": tool_calls}
                )
            else:
                yield StreamEvent.token_event("done")
                yield StreamEvent.complete_event("done")

        agent.llm.call_stream = nameless_then_content

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) == 0

    @pytest.mark.asyncio
    async def test_tool_execution_failure_yields_error(self):
        """If a tool raises an exception, the loop should yield ERROR."""
        agent = _make_agent()
        mode = DirectMode()

        async def tool_stream(**kwargs):
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "explode", "arguments": "{}"},
                }
            ]
            yield StreamEvent.complete_event("", metadata={"tool_calls": tool_calls})

        agent.llm.call_stream = tool_stream
        mode.call_tool = AsyncMock(side_effect=RuntimeError("tool exploded"))

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        assert events[-1].type == "error"
        assert "explode" in events[-1].message

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response(self):
        """Multiple tool calls in one LLM response should all be executed."""
        agent = _make_agent()
        mode = DirectMode()

        call_num = 0

        async def multi_tool_then_content(**kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                tool_calls = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": "{}"},
                    },
                ]
                yield StreamEvent.complete_event(
                    "", metadata={"tool_calls": tool_calls}
                )
            else:
                yield StreamEvent.token_event("ok")
                yield StreamEvent.complete_event("ok")

        agent.llm.call_stream = multi_tool_then_content
        mode.call_tool = AsyncMock(return_value="result")

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=10)
        )

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) == 2
        assert tool_starts[0].tool_name == "tool_a"
        assert tool_starts[1].tool_name == "tool_b"

    @pytest.mark.asyncio
    async def test_no_complete_event_yields_error(self):
        """If LLM stream yields no COMPLETE event, loop should error."""
        agent = _make_agent()
        mode = DirectMode()

        async def no_complete(**kwargs):
            yield StreamEvent.token_event("orphan")

        agent.llm.call_stream = no_complete

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        assert events[-1].type == "error"
        assert "no COMPLETE" in events[-1].message

    @pytest.mark.asyncio
    async def test_content_and_tool_calls_together_prefers_tools(self):
        """If LLM returns both content and tool_calls, tools take priority."""
        agent = _make_agent()
        mode = DirectMode()

        call_num = 0

        async def both(**kwargs):
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                tool_calls = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ]
                yield StreamEvent.token_event("text")
                yield StreamEvent.complete_event(
                    "text", metadata={"tool_calls": tool_calls}
                )
            else:
                yield StreamEvent.token_event("final")
                yield StreamEvent.complete_event("final")

        agent.llm.call_stream = both
        mode.call_tool = AsyncMock(return_value="ok")

        messages = mode.build_messages(agent, _make_task())
        events = await _collect(
            mode._streaming_tool_call_loop(agent, messages, None, max_tool_calls=5)
        )

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) == 1

    @pytest.mark.asyncio
    async def test_call_round_increments(self):
        """Each LLM round should have incrementing call_count."""
        tool = MockCalculatorTool()
        agent = _make_agent(tools=[tool])
        mode = DirectMode()
        mode._ensure_executor(agent)
        messages = mode.build_messages(agent, _make_task("Add 1 and 2"))
        tool_specs = agent.llm.convert_tool_specs(agent.tools)

        events = await _collect(
            mode._streaming_tool_call_loop(
                agent, messages, tool_specs, max_tool_calls=5
            )
        )

        starts = [e for e in events if e.type == "llm_start"]
        if len(starts) >= 2:
            assert starts[0].call_count == 1
            assert starts[1].call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# StandardMode.run_stream — memory verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestStandardModeMemoryVerification:
    @pytest.mark.asyncio
    async def test_final_content_stored_in_memory(self):
        """After streaming completes, the final content should be in memory."""
        memory = FullHistoryMemory()
        agent = _make_agent(
            memory=memory,
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        mode = StandardMode()

        events = await _collect(mode.run_stream(agent, _make_task("store me")))

        complete = next(e for e in events if e.type == "complete")
        context = memory.get_context()
        assistant_msgs = [m for m in context if m["role"] == "assistant"]
        assert any(complete.content in m["content"] for m in assistant_msgs)


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousMode.run_stream — critic/refiner and complex path
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutonomousModeStreamCriticRefiner:
    @pytest.mark.asyncio
    async def test_thinking_event_for_critic(self):
        """Simple path should emit THINKING event for critic verification."""
        agent = _make_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=1,
            ),
        )
        mode = AutonomousMode()

        events = await _collect(mode.run_stream(agent, _make_task("verify me")))

        thinking_msgs = [e.message for e in events if e.type == "thinking"]
        assert any("Analyzing" in m for m in thinking_msgs)
        assert any("Critic" in m or "Verif" in m for m in thinking_msgs)

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """With multiple retries, mode should complete without infinite loop."""
        agent = _make_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=2,
            ),
        )
        mode = AutonomousMode()

        events = await _collect(mode.run_stream(agent, _make_task("retry test")))

        assert any(e.type == "complete" for e in events)
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_error_in_simple_path(self):
        """Exception in _stream_simple should yield error."""
        agent = _make_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=1,
            ),
        )
        mode = AutonomousMode()

        async def explode(**kwargs):
            raise RuntimeError("boom in autonomous")
            yield  # noqa: E501

        agent.llm.call_stream = explode

        events = await _collect(mode.run_stream(agent, _make_task("fail")))

        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_dict_task_auto_converted(self):
        """AutonomousMode.run_stream should accept dict tasks."""
        agent = _make_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=1,
            ),
        )
        mode = AutonomousMode()

        events = await _collect(
            mode.run_stream(agent, {"id": "d1", "objective": "dict task"})
        )

        assert any(e.type == "complete" for e in events)

    @pytest.mark.asyncio
    async def test_autonomous_simple_path_completes(self):
        """Simple path should reach COMPLETED state even with tools registered."""
        tool = MockCalculatorTool()
        agent = _make_agent(
            tools=[tool],
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=1,
            ),
        )
        mode = AutonomousMode()

        events = await _collect(mode.run_stream(agent, _make_task("Add 3 and 7")))

        types = [e.type for e in events]
        assert "thinking" in types
        assert "complete" in types
        assert agent.state == AgentState.COMPLETED
