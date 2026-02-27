"""Tests for Agent.execute_stream() — the public streaming API.

Covers:
    1. Full lifecycle: task conversion, LLM params, plugins, memory, mode routing
    2. Token-by-token streaming across all execution modes
    3. Tool call streaming (TOOL_CALL_START / TOOL_CALL_END)
    4. PluginHalt handling (before-agent, during execution)
    5. LLM params override passthrough
    6. Event ordering invariants
    7. DRY: _setup_execution / _resolve_mode shared with execute()
"""

from __future__ import annotations

from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.llm_params import LLMParams
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.plugins.base import BasePlugin
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.streaming.events import StreamEvent
from nucleusiq.tools import BaseTool

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


class MockCalculatorTool(BaseTool):
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
    events = []
    async for event in async_gen:
        events.append(event)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# Agent.execute_stream() — basic streaming
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamBasic:
    @pytest.mark.asyncio
    async def test_yields_stream_events(self):
        """execute_stream should yield StreamEvent objects."""
        agent = _make_agent()
        events = await _collect(agent.execute_stream(_make_task()))

        assert len(events) > 0
        assert all(isinstance(e, StreamEvent) for e in events)

    @pytest.mark.asyncio
    async def test_contains_complete_event(self):
        agent = _make_agent()
        events = await _collect(agent.execute_stream(_make_task()))

        completes = [e for e in events if e.type == "complete"]
        assert len(completes) == 1
        assert completes[0].content

    @pytest.mark.asyncio
    async def test_token_events_present(self):
        agent = _make_agent()
        events = await _collect(agent.execute_stream(_make_task()))

        tokens = [e for e in events if e.type == "token"]
        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_llm_call_start_and_end(self):
        agent = _make_agent()
        events = await _collect(agent.execute_stream(_make_task()))

        types = [e.type for e in events]
        assert "llm_start" in types
        assert "llm_end" in types


# ═══════════════════════════════════════════════════════════════════════════════
# Task conversion (dict → Task)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamTaskConversion:
    @pytest.mark.asyncio
    async def test_dict_task_accepted(self):
        agent = _make_agent()
        events = await _collect(
            agent.execute_stream({"id": "t2", "objective": "test dict"})
        )

        assert any(e.type == "complete" for e in events)

    @pytest.mark.asyncio
    async def test_task_object_accepted(self):
        agent = _make_agent()
        events = await _collect(agent.execute_stream(_make_task()))

        assert any(e.type == "complete" for e in events)


# ═══════════════════════════════════════════════════════════════════════════════
# Mode routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamModeRouting:
    @pytest.mark.asyncio
    async def test_direct_mode(self):
        agent = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.DIRECT))
        events = await _collect(agent.execute_stream(_make_task()))

        assert any(e.type == "complete" for e in events)
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_standard_mode(self):
        agent = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.STANDARD))
        events = await _collect(agent.execute_stream(_make_task()))

        assert any(e.type == "complete" for e in events)
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_autonomous_mode(self):
        agent = _make_agent(
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=1,
            ),
        )
        events = await _collect(agent.execute_stream(_make_task()))

        types = [e.type for e in events]
        assert "thinking" in types
        assert "complete" in types

    @pytest.mark.asyncio
    async def test_unknown_mode_raises(self):
        agent = _make_agent()
        agent.config.execution_mode = "nonexistent"

        with pytest.raises(ValueError, match="Unknown execution mode"):
            await _collect(agent.execute_stream(_make_task()))


# ═══════════════════════════════════════════════════════════════════════════════
# Tool call streaming
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamToolCalls:
    @pytest.mark.asyncio
    async def test_tool_events_emitted(self):
        tool = MockCalculatorTool()
        agent = _make_agent(
            tools=[tool],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        events = await _collect(agent.execute_stream(_make_task("Add 3 and 5")))

        types = [e.type for e in events]
        assert "tool_start" in types
        assert "tool_end" in types

    @pytest.mark.asyncio
    async def test_tool_start_has_name_and_args(self):
        tool = MockCalculatorTool()
        agent = _make_agent(
            tools=[tool],
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        events = await _collect(agent.execute_stream(_make_task("Add 4 and 6")))

        starts = [e for e in events if e.type == "tool_start"]
        assert len(starts) >= 1
        assert starts[0].tool_name == "add"
        assert starts[0].tool_args is not None

    @pytest.mark.asyncio
    async def test_tool_end_has_result(self):
        tool = MockCalculatorTool()
        agent = _make_agent(
            tools=[tool],
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )
        events = await _collect(agent.execute_stream(_make_task("Add 4 and 6")))

        ends = [e for e in events if e.type == "tool_end"]
        assert len(ends) >= 1
        assert ends[0].tool_result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Memory integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamMemory:
    @pytest.mark.asyncio
    async def test_user_input_stored_in_memory(self):
        memory = FullHistoryMemory()
        agent = _make_agent(memory=memory)
        await _collect(agent.execute_stream(_make_task("remember this")))

        context = memory.get_context()
        user_msgs = [m for m in context if m["role"] == "user"]
        assert any("remember this" in m["content"] for m in user_msgs)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM params override
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamLLMParams:
    @pytest.mark.asyncio
    async def test_llm_params_applied(self):
        agent = _make_agent()
        params = LLMParams(temperature=0.1, max_tokens=500)

        events = await _collect(agent.execute_stream(_make_task(), llm_params=params))

        assert any(e.type == "complete" for e in events)

    @pytest.mark.asyncio
    async def test_overrides_cleaned_up_after_stream(self):
        agent = _make_agent()
        params = LLMParams(temperature=0.9)

        await _collect(agent.execute_stream(_make_task(), llm_params=params))

        assert agent._current_llm_overrides == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class HaltBeforePlugin(BasePlugin):
    """Plugin that halts execution in the before_agent hook."""

    async def before_agent(self, context):
        raise PluginHalt(result="halted by plugin")


class TestExecuteStreamPlugins:
    @pytest.mark.asyncio
    async def test_plugin_halt_yields_complete(self):
        agent = _make_agent(plugins=[HaltBeforePlugin()])
        events = await _collect(agent.execute_stream(_make_task()))

        assert len(events) == 1
        assert events[0].type == "complete"
        assert "halted by plugin" in events[0].content


# ═══════════════════════════════════════════════════════════════════════════════
# No LLM (echo fallback)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamNoLLM:
    @pytest.mark.asyncio
    async def test_no_llm_yields_echo(self):
        agent = _make_agent(llm=None)
        events = await _collect(agent.execute_stream(_make_task("echo me")))

        assert len(events) == 1
        assert events[0].type == "complete"
        assert "echo me" in events[0].content


# ═══════════════════════════════════════════════════════════════════════════════
# DRY: _setup_execution / _resolve_mode
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetupExecution:
    @pytest.mark.asyncio
    async def test_returns_task_mode_and_context(self):
        agent = _make_agent()
        task, mode, ctx = await agent._setup_execution(
            {"id": "t1", "objective": "test"}
        )

        assert isinstance(task, Task)
        assert isinstance(mode, BaseExecutionMode)
        assert ctx.agent_name == "TestAgent"

    @pytest.mark.asyncio
    async def test_plugin_halt_propagates(self):
        agent = _make_agent(plugins=[HaltBeforePlugin()])

        with pytest.raises(PluginHalt):
            await agent._setup_execution(_make_task())


class TestResolveMode:
    def test_direct_mode(self):
        agent = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.DIRECT))
        mode = agent._resolve_mode()
        assert type(mode).__name__ == "DirectMode"

    def test_standard_mode(self):
        agent = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.STANDARD))
        mode = agent._resolve_mode()
        assert type(mode).__name__ == "StandardMode"

    def test_autonomous_mode(self):
        agent = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS))
        mode = agent._resolve_mode()
        assert type(mode).__name__ == "AutonomousMode"

    def test_unknown_raises(self):
        agent = _make_agent()
        agent.config.execution_mode = "nonexistent"
        with pytest.raises(ValueError, match="Unknown execution mode"):
            agent._resolve_mode()


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end: Agent → Mode → LLM → StreamEvent
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_text_only(self):
        """Agent → Standard mode → MockLLM → stream events → final text."""
        agent = _make_agent()
        await agent.initialize()

        collected_tokens: list[str] = []
        final_text: str | None = None

        async for event in agent.execute_stream(_make_task("What is AI?")):
            if event.type == "token":
                collected_tokens.append(event.token)
            elif event.type == "complete":
                final_text = event.content

        assert len(collected_tokens) > 0
        assert final_text is not None
        reconstructed = "".join(collected_tokens)
        assert reconstructed == final_text

    @pytest.mark.asyncio
    async def test_full_pipeline_with_tools(self):
        """Agent → Standard mode → MockLLM → tool call → final text."""
        tool = MockCalculatorTool()
        agent = _make_agent(
            tools=[tool],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        await agent.initialize()

        event_types: list[str] = []
        final_text: str | None = None

        async for event in agent.execute_stream(_make_task("Add 10 and 20")):
            event_types.append(event.type)
            if event.type == "complete":
                final_text = event.content

        assert "llm_start" in event_types
        assert "tool_start" in event_types
        assert "tool_end" in event_types
        assert "complete" in event_types
        assert final_text is not None

    @pytest.mark.asyncio
    async def test_execute_and_execute_stream_produce_same_content(self):
        """Both APIs should produce equivalent final content."""
        agent1 = _make_agent()
        agent2 = _make_agent()

        result = await agent1.execute(_make_task("What is NucleusIQ?"))

        final_content = None
        async for event in agent2.execute_stream(_make_task("What is NucleusIQ?")):
            if event.type == "complete":
                final_content = event.content

        assert result == final_content

    @pytest.mark.asyncio
    async def test_sse_serialization(self):
        """Events should serialize to valid SSE format."""
        agent = _make_agent()

        async for event in agent.execute_stream(_make_task()):
            sse = event.to_sse()
            assert sse.startswith("data: ")
            assert sse.endswith("\n\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Lifecycle edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamLifecycleEdges:
    @pytest.mark.asyncio
    async def test_overrides_cleaned_on_error(self):
        """_current_llm_overrides must be cleared even if streaming errors."""
        agent = _make_agent()
        params = LLMParams(temperature=0.5)

        async def error_stream(**kwargs):
            raise RuntimeError("fail")
            yield  # noqa: E501

        agent.llm.call_stream = error_stream

        await _collect(agent.execute_stream(_make_task(), llm_params=params))

        assert agent._current_llm_overrides == {}

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls_isolated(self):
        """Multiple execute_stream calls should not leak state."""
        agent = _make_agent()

        events1 = await _collect(agent.execute_stream(_make_task("first")))
        events2 = await _collect(agent.execute_stream(_make_task("second")))

        content1 = next(e.content for e in events1 if e.type == "complete")
        content2 = next(e.content for e in events2 if e.type == "complete")

        assert "first" in content1
        assert "second" in content2
        assert agent._current_llm_overrides == {}

    @pytest.mark.asyncio
    async def test_tool_count_exceeds_limit_raises(self):
        """Too many tools for the mode should raise ValueError."""
        tools = [MockCalculatorTool() for _ in range(100)]
        agent = _make_agent(
            tools=tools,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        )

        with pytest.raises(ValueError, match="tools"):
            await _collect(agent.execute_stream(_make_task()))

    @pytest.mark.asyncio
    async def test_custom_mode_via_register_mode(self):
        """A custom mode registered at runtime should work with streaming."""

        class EchoStreamMode(BaseExecutionMode):
            async def run(self, agent, task):
                return "custom echo"

        Agent.register_mode("echo_stream", EchoStreamMode)
        try:
            agent = _make_agent()
            agent.config.execution_mode = "echo_stream"

            events = await _collect(agent.execute_stream(_make_task()))

            assert len(events) == 1
            assert events[0].type == "complete"
            assert "custom echo" in events[0].content
        finally:
            del Agent._mode_registry["echo_stream"]


# ═══════════════════════════════════════════════════════════════════════════════
# Memory: full round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStreamMemoryRoundTrip:
    @pytest.mark.asyncio
    async def test_memory_stores_user_and_assistant(self):
        """After streaming, memory should contain both user input and assistant content."""
        memory = FullHistoryMemory()
        agent = _make_agent(
            memory=memory,
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )

        events = await _collect(agent.execute_stream(_make_task("What is 2+2?")))

        assert any(e.type == "complete" for e in events)
        context = memory.get_context()
        roles = [m["role"] for m in context]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_empty_objective_still_streams(self):
        """Agent should handle empty objective gracefully."""
        agent = _make_agent()
        events = await _collect(agent.execute_stream(_make_task("")))

        assert any(e.type == "complete" for e in events)


# ═══════════════════════════════════════════════════════════════════════════════
# Parity: execute() vs execute_stream() across modes
# ═══════════════════════════════════════════════════════════════════════════════


class TestParityAcrossModes:
    @pytest.mark.asyncio
    async def test_direct_mode_parity(self):
        a1 = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.DIRECT))
        a2 = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.DIRECT))

        result = await a1.execute(_make_task("direct parity"))
        final = None
        async for e in a2.execute_stream(_make_task("direct parity")):
            if e.type == "complete":
                final = e.content

        assert result == final

    @pytest.mark.asyncio
    async def test_standard_mode_parity(self):
        a1 = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.STANDARD))
        a2 = _make_agent(config=AgentConfig(execution_mode=ExecutionMode.STANDARD))

        result = await a1.execute(_make_task("standard parity"))
        final = None
        async for e in a2.execute_stream(_make_task("standard parity")):
            if e.type == "complete":
                final = e.content

        assert result == final

    @pytest.mark.asyncio
    async def test_tool_call_parity(self):
        """Both APIs should give the same final answer after tool execution."""
        tool1 = MockCalculatorTool()
        tool2 = MockCalculatorTool()
        a1 = _make_agent(
            tools=[tool1],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )
        a2 = _make_agent(
            tools=[tool2],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        )

        result = await a1.execute(_make_task("Add 8 and 9"))
        final = None
        async for e in a2.execute_stream(_make_task("Add 8 and 9")):
            if e.type == "complete":
                final = e.content

        assert result == final
