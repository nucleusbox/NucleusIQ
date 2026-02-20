"""Tests covering uncovered lines in modes/ (base_mode, direct_mode, standard_mode, autonomous_mode)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.task import Task
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.direct_mode import DirectMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.full_history import FullHistoryMemory


def _make_agent(**overrides):
    defaults = dict(
        name="TestAgent",
        role="Assistant",
        objective="Help users",
        narrative="Test agent",
        llm=MockLLM(),
    )
    defaults.update(overrides)
    return Agent(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# BaseExecutionMode helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseExecutionModeHelpers:

    def test_get_objective_dict(self):
        result = BaseExecutionMode.get_objective({"objective": "do stuff"})
        assert result == "do stuff"

    def test_get_objective_task_object(self):
        task = Task(id="t1", objective="from task")
        result = BaseExecutionMode.get_objective(task)
        assert result == "from task"

    def test_echo_fallback_no_llm(self):
        agent = _make_agent(llm=None)
        mode = DirectMode()
        result = mode.echo_fallback(agent, {"objective": "hello"})
        assert "hello" in result
        assert agent.state == AgentState.COMPLETED

    def test_echo_fallback_with_llm(self):
        agent = _make_agent()
        mode = DirectMode()
        result = mode.echo_fallback(agent, {"objective": "hello"})
        assert result is None

    def test_validate_response_raises_on_none(self):
        with pytest.raises(ValueError, match="empty response"):
            BaseExecutionMode.validate_response(None)

    def test_validate_response_raises_on_empty_choices(self):
        resp = MagicMock()
        resp.choices = []
        with pytest.raises(ValueError, match="empty response"):
            BaseExecutionMode.validate_response(resp)

    def test_validate_response_ok(self):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        BaseExecutionMode.validate_response(resp)

    def test_extract_content_str(self):
        msg = MagicMock()
        msg.content = "hello"
        assert BaseExecutionMode.extract_content(msg) == "hello"

    def test_extract_content_dict(self):
        msg = {"content": "hi"}
        assert BaseExecutionMode.extract_content(msg) == "hi"

    def test_extract_content_none_returns_none(self):
        msg = MagicMock()
        msg.content = None
        assert BaseExecutionMode.extract_content(msg) is None

    def test_extract_content_list_format(self):
        msg = MagicMock()
        msg.content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        result = BaseExecutionMode.extract_content(msg)
        assert "part1" in result
        assert "part2" in result

    def test_extract_content_empty_list(self):
        msg = MagicMock()
        msg.content = []
        assert BaseExecutionMode.extract_content(msg) is None


# ═══════════════════════════════════════════════════════════════════════════════
# DirectMode
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectMode:

    @pytest.mark.asyncio
    async def test_no_llm_echo(self):
        agent = _make_agent(llm=None)
        mode = DirectMode()
        result = await mode.run(agent, {"id": "1", "objective": "test"})
        assert "test" in result

    @pytest.mark.asyncio
    async def test_with_llm(self):
        agent = _make_agent()
        await agent.initialize()
        mode = DirectMode()
        result = await mode.run(agent, {"id": "1", "objective": "hello"})
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# StandardMode with memory
# ═══════════════════════════════════════════════════════════════════════════════


class TestStandardModeMemory:

    @pytest.mark.asyncio
    async def test_store_in_memory(self):
        mem = FullHistoryMemory()
        agent = _make_agent(memory=mem)
        await StandardMode()._store_in_memory(agent, "task", "result text")
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["content"] == "result text"

    @pytest.mark.asyncio
    async def test_store_in_memory_no_memory(self):
        agent = _make_agent()
        await StandardMode()._store_in_memory(agent, "task", "result text")

    @pytest.mark.asyncio
    async def test_store_in_memory_error_is_caught(self):
        """Memory failure doesn't crash — just logs a warning."""

        class FailingMemory(FullHistoryMemory):
            async def aadd_message(self, role, content, **kw):
                raise RuntimeError("fail")

        mem = FailingMemory()
        agent = _make_agent(memory=mem)
        await StandardMode()._store_in_memory(agent, "task", "text")
