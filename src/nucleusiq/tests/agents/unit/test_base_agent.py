"""Tests for agents/builder/base_agent.py and agents/agent.py utility methods."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, AgentMetrics, AgentState
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.full_history import FullHistoryMemory


def _make_agent(**overrides):
    defaults = dict(
        name="TestAgent",
        role="Assistant",
        objective="Help",
        narrative="Test",
    )
    defaults.update(overrides)
    return Agent(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# BaseAgent - initialization & config
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseAgentInit:

    def test_default_state(self):
        agent = _make_agent()
        assert agent.state == AgentState.INITIALIZING

    def test_verbose_logging(self):
        agent = _make_agent(config=AgentConfig(verbose=True))
        assert agent._logger.level == 10  # DEBUG

    def test_non_verbose_logging(self):
        agent = _make_agent(config=AgentConfig(verbose=False))
        assert agent._logger.level == 20  # INFO

    def test_unsafe_code_warning(self):
        with patch("logging.Logger.warning") as mock_warn:
            _make_agent(
                config=AgentConfig(
                    allow_code_execution=True, code_execution_mode="unsafe"
                )
            )

    def test_metrics_default(self):
        agent = _make_agent()
        assert agent.metrics.tasks_completed == 0
        assert agent.metrics.error_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _execute_with_retry, _execute_step, _check_execution_timeout
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetryMechanism:

    @pytest.mark.asyncio
    async def test_execute_step_updates_metrics_on_success(self):
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()
        agent._start_time = 1e10

        result = await agent._execute_step({"id": "1", "objective": "hi"})
        assert agent.metrics.successful_executions >= 1

    @pytest.mark.asyncio
    async def test_update_metrics_failure(self):
        """_update_metrics(success=False) increments error counters."""
        agent = _make_agent()
        agent._update_metrics(success=False)
        assert agent.metrics.failed_executions == 1
        assert agent.metrics.error_count == 1

    @pytest.mark.asyncio
    async def test_update_metrics_success(self):
        agent = _make_agent()
        agent._update_metrics(success=True, execution_time=1.5)
        assert agent.metrics.successful_executions == 1
        assert agent.metrics.total_execution_time == 1.5
        assert agent.metrics.average_response_time == 1.5

    def test_check_timeout_false_when_not_started(self):
        agent = _make_agent()
        assert agent._check_execution_timeout() is False

    def test_validate_result(self):
        agent = _make_agent()
        assert agent._validate_result("something") is True
        assert agent._validate_result(None) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Agent.initialize with memory
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentMemoryInit:

    @pytest.mark.asyncio
    async def test_initialize_with_memory(self):
        mem = FullHistoryMemory()
        llm = MockLLM()
        agent = _make_agent(llm=llm, memory=mem)
        await agent.initialize()

    @pytest.mark.asyncio
    async def test_initialize_without_memory(self):
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()


# ═══════════════════════════════════════════════════════════════════════════════
# Agent state save / load
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentState:

    @pytest.mark.asyncio
    async def test_save_state_without_memory(self):
        agent = _make_agent()
        state = await agent.save_state()
        assert "id" in state
        assert "memory" not in state

    @pytest.mark.asyncio
    async def test_save_state_with_memory(self):
        mem = FullHistoryMemory()
        mem.add_message("user", "hello")
        agent = _make_agent(memory=mem)
        state = await agent.save_state()
        assert "memory" in state
        assert state["memory"]["messages"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_load_state_with_memory(self):
        mem = FullHistoryMemory()
        agent = _make_agent(memory=mem)
        state = {
            "state": AgentState.COMPLETED,
            "metrics": AgentMetrics().model_dump(),
            "current_task": None,
            "timestamp": "2024-01-01",
            "memory": {"messages": [{"role": "user", "content": "restored"}]},
        }
        await agent.load_state(state)
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["content"] == "restored"

    @pytest.mark.asyncio
    async def test_load_state_without_memory(self):
        agent = _make_agent()
        state = {
            "state": AgentState.EXECUTING,
            "metrics": AgentMetrics().model_dump(),
            "current_task": None,
            "timestamp": "2024-01-01",
        }
        await agent.load_state(state)
        assert agent.state == AgentState.EXECUTING


# ═══════════════════════════════════════════════════════════════════════════════
# Agent._process_result and _handle_error with memory
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentProcessResult:

    @pytest.mark.asyncio
    async def test_process_result_stores_in_memory(self):
        mem = FullHistoryMemory()
        agent = _make_agent(memory=mem)
        result = await agent._process_result("task output")
        assert result == "task output"
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_handle_error_stores_in_memory(self):
        mem = FullHistoryMemory()
        agent = _make_agent(memory=mem)
        await agent._handle_error(RuntimeError("boom"), {"ctx": "test"})
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert "boom" in ctx[0]["content"]
        assert agent.metrics.error_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# BaseAgent.remove_tool
# ═══════════════════════════════════════════════════════════════════════════════


class TestRemoveTool:

    @pytest.mark.asyncio
    async def test_remove_tool(self):
        tool = MagicMock()
        tool.name = "calc"
        agent = _make_agent(tools=[tool])
        assert len(agent.tools) == 1
        await agent.remove_tool("calc")
        assert len(agent.tools) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_tool(self):
        agent = _make_agent()
        await agent.remove_tool("nope")
        assert len(agent.tools) == 0
