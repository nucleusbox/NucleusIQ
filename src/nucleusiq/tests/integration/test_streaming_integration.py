"""Cross-milestone integration tests for the streaming stack.

Verifies that all layers work together end-to-end:
    Agent.execute_stream()
      → Mode.run_stream()
        → BaseExecutionMode._streaming_tool_call_loop()
          → BaseLLM.call_stream()
            → StreamEvent

Tests are grouped by scenario complexity:
    1. Simple text (Agent → Mode → LLM → tokens → COMPLETE)
    2. Tool calls (Agent → Mode → LLM → tool call → tool exec → LLM → COMPLETE)
    3. Multi-mode (same task across Direct/Standard/Autonomous)
    4. Memory round-trip (streaming + memory persistence)
    5. LLM params passthrough (config + per-execute merge)
    6. Event stream invariants (ordering, uniqueness, completeness)
"""

from __future__ import annotations

from typing import Any

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.llms.llm_params import LLMParams
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.streaming.events import StreamEvent
from nucleusiq.tools import BaseTool

# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════════


class AddTool(BaseTool):
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


def _agent(mode: str = "standard", **kw) -> Agent:
    defaults = dict(
        name="IntegrationAgent",
        role="Assistant",
        objective="Help users",
        narrative="Integration test agent",
        llm=MockLLM(),
        config=AgentConfig(execution_mode=mode),
    )
    defaults.update(kw)
    return Agent(**defaults)


async def _stream(agent: Agent, objective: str) -> list[StreamEvent]:
    events = []
    async for e in agent.execute_stream({"id": "int-1", "objective": objective}):
        events.append(e)
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Simple text: full stack without tools
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimpleTextIntegration:
    @pytest.mark.asyncio
    async def test_tokens_reconstruct_complete(self):
        """Accumulated TOKEN events must equal the COMPLETE content."""
        agent = _agent("standard")
        events = await _stream(agent, "Hello world")

        tokens = "".join(e.token for e in events if e.type == "token")
        complete = next(e.content for e in events if e.type == "complete")
        assert tokens == complete

    @pytest.mark.asyncio
    async def test_minimal_event_set(self):
        """Every stream must have at least: LLM_START, TOKEN, LLM_END, COMPLETE."""
        agent = _agent("direct")
        events = await _stream(agent, "test")

        types = {e.type for e in events}
        assert {"llm_start", "token", "llm_end", "complete"} <= types

    @pytest.mark.asyncio
    async def test_agent_state_completed(self):
        agent = _agent("standard")
        await _stream(agent, "state check")
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_chunk_size_affects_token_count(self):
        """Larger chunk_size → fewer TOKEN events."""
        a1 = _agent("direct", llm=MockLLM(stream_chunk_size=1))
        a2 = _agent("direct", llm=MockLLM(stream_chunk_size=10))

        events1 = await _stream(a1, "chunk test")
        events2 = await _stream(a2, "chunk test")

        tokens1 = [e for e in events1 if e.type == "token"]
        tokens2 = [e for e in events2 if e.type == "token"]
        assert len(tokens1) >= len(tokens2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tool calls: LLM → tool exec → follow-up LLM
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCallIntegration:
    @pytest.mark.asyncio
    async def test_tool_round_trip(self):
        """Full: LLM detects tool → execute → second LLM call with result."""
        agent = _agent("standard", tools=[AddTool()])
        events = await _stream(agent, "Add 5 and 3")

        types = [e.type for e in events]
        assert types.count("llm_start") >= 2
        assert "tool_start" in types
        assert "tool_end" in types
        assert "complete" in types

    @pytest.mark.asyncio
    async def test_tool_result_in_final_answer(self):
        """The final answer should incorporate the tool result."""
        agent = _agent("standard", tools=[AddTool()])
        events = await _stream(agent, "Add 5 and 3")

        complete = next(e for e in events if e.type == "complete")
        assert "8" in complete.content

    @pytest.mark.asyncio
    async def test_tool_name_in_events(self):
        agent = _agent("direct", tools=[AddTool()])
        events = await _stream(agent, "Add 10 and 20")

        starts = [e for e in events if e.type == "tool_start"]
        assert all(e.tool_name == "add" for e in starts)

    @pytest.mark.asyncio
    async def test_tool_args_correct(self):
        agent = _agent("direct", tools=[AddTool()])
        events = await _stream(agent, "Add 10 and 20")

        starts = [e for e in events if e.type == "tool_start"]
        if starts:
            assert starts[0].tool_args.get("a") == 10
            assert starts[0].tool_args.get("b") == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-mode: same task across modes
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiModeIntegration:
    @pytest.mark.asyncio
    async def test_all_modes_produce_complete(self):
        """All modes should produce at least one COMPLETE event."""
        for mode in ("direct", "standard"):
            agent = _agent(mode)
            events = await _stream(agent, "hello")
            assert any(e.type == "complete" for e in events), f"{mode} failed"

    @pytest.mark.asyncio
    async def test_autonomous_produces_thinking(self):
        agent = _agent(
            "autonomous",
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                max_retries=1,
            ),
        )
        events = await _stream(agent, "think about this")

        types = [e.type for e in events]
        assert "thinking" in types
        assert "complete" in types

    @pytest.mark.asyncio
    async def test_mode_tool_streaming_direct_vs_standard(self):
        """Both Direct and Standard should handle tools via streaming."""
        for mode in ("direct", "standard"):
            agent = _agent(mode, tools=[AddTool()])
            events = await _stream(agent, "Add 7 and 8")

            types = [e.type for e in events]
            assert "tool_start" in types, f"{mode} missing tool_start"
            assert "tool_end" in types, f"{mode} missing tool_end"
            assert "complete" in types, f"{mode} missing complete"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Memory round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryIntegration:
    @pytest.mark.asyncio
    async def test_user_input_and_assistant_response_stored(self):
        memory = FullHistoryMemory()
        agent = _agent("standard", memory=memory)

        await _stream(agent, "Remember 42")

        context = memory.get_context()
        assert any(m["role"] == "user" and "42" in m["content"] for m in context)

    @pytest.mark.asyncio
    async def test_consecutive_streams_accumulate_memory(self):
        memory = FullHistoryMemory()
        agent = _agent("standard", memory=memory)

        await _stream(agent, "First message")
        await _stream(agent, "Second message")

        context = memory.get_context()
        user_msgs = [m for m in context if m["role"] == "user"]
        assert len(user_msgs) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LLM params passthrough
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMParamsIntegration:
    @pytest.mark.asyncio
    async def test_per_execute_params_used_and_cleared(self):
        agent = _agent("standard")

        events: list[StreamEvent] = []
        async for e in agent.execute_stream(
            {"id": "p1", "objective": "params test"},
            llm_params=LLMParams(temperature=0.1, max_tokens=200),
        ):
            events.append(e)

        assert any(e.type == "complete" for e in events)
        assert agent._current_llm_overrides == {}

    @pytest.mark.asyncio
    async def test_config_level_params(self):
        config = AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            llm_params=LLMParams(temperature=0.8),
        )
        agent = _agent("standard", config=config)
        events = await _stream(agent, "config params")

        assert any(e.type == "complete" for e in events)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Event stream invariants
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventStreamInvariants:
    @pytest.mark.asyncio
    async def test_exactly_one_complete_event(self):
        """Stream must have exactly one COMPLETE event."""
        agent = _agent("standard")
        events = await _stream(agent, "invariant check")

        completes = [e for e in events if e.type == "complete"]
        assert len(completes) == 1

    @pytest.mark.asyncio
    async def test_llm_starts_equal_llm_ends(self):
        """Every LLM_CALL_START must have a matching LLM_CALL_END."""
        agent = _agent("standard")
        events = await _stream(agent, "balanced calls")

        starts = sum(1 for e in events if e.type == "llm_start")
        ends = sum(1 for e in events if e.type == "llm_end")
        assert starts == ends

    @pytest.mark.asyncio
    async def test_tool_starts_equal_tool_ends(self):
        """Every TOOL_CALL_START must have a matching TOOL_CALL_END."""
        agent = _agent("standard", tools=[AddTool()])
        events = await _stream(agent, "Add 1 and 2")

        starts = sum(1 for e in events if e.type == "tool_start")
        ends = sum(1 for e in events if e.type == "tool_end")
        assert starts == ends

    @pytest.mark.asyncio
    async def test_complete_is_last_non_thinking_event(self):
        """COMPLETE should be the last data event (THINKING may follow for autonomous)."""
        agent = _agent("standard")
        events = await _stream(agent, "order check")

        data_events = [e for e in events if e.type not in ("thinking",)]
        assert data_events[-1].type == "complete"

    @pytest.mark.asyncio
    async def test_no_events_after_error(self):
        """After an ERROR event, no further events should be emitted."""
        agent = _agent("standard")

        async def failing(**kwargs):
            yield StreamEvent.error_event("fail")

        agent.llm.call_stream = failing

        events = await _stream(agent, "error test")

        error_idx = next(i for i, e in enumerate(events) if e.type == "error")
        after_error = [e for e in events[error_idx + 1 :] if e.type not in ("llm_end",)]
        assert len(after_error) == 0

    @pytest.mark.asyncio
    async def test_all_events_are_stream_event_instances(self):
        agent = _agent("standard", tools=[AddTool()])
        events = await _stream(agent, "Add 2 and 3")

        assert all(isinstance(e, StreamEvent) for e in events)

    @pytest.mark.asyncio
    async def test_sse_format_valid_for_all_events(self):
        """Every event must serialize to valid SSE format."""
        agent = _agent("standard", tools=[AddTool()])
        events = await _stream(agent, "Add 2 and 3")

        for event in events:
            sse = event.to_sse()
            assert sse.startswith("data: ")
            assert sse.endswith("\n\n")
            assert '"type"' in sse
