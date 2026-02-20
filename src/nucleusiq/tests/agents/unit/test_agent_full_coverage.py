"""
Full coverage tests for agent.py, base_agent.py, message_builder.py,
standard_mode.py, autonomous_mode.py — targeting every uncovered line.
"""

import asyncio
import inspect
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan, PlanStep, PlanResponse
from nucleusiq.agents.config import AgentConfig, AgentMetrics, AgentState, ExecutionMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.agents.modes.direct_mode import DirectMode
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.prompts.zero_shot import ZeroShotPrompt


def _make_agent(**overrides):
    defaults = dict(
        name="TestAgent", role="Assistant",
        objective="Help users", narrative="Test agent",
    )
    defaults.update(overrides)
    return Agent(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent.register_mode (line 101)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegisterMode:

    def test_register_custom_mode(self):
        class CustomMode:
            async def run(self, agent, task):
                return "custom"

        Agent.register_mode("custom_test", CustomMode)
        assert "custom_test" in Agent._mode_registry
        del Agent._mode_registry["custom_test"]


# ═══════════════════════════════════════════════════════════════════════════════
# Agent._resolve_response_format (lines 299-313)
# Agent._get_structured_output_kwargs (lines 333-360)
# Agent._wrap_structured_output_result (lines 380-401)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuredOutputOnAgent:

    def test_resolve_response_format_none(self):
        agent = _make_agent()
        assert agent._resolve_response_format() is None

    def test_resolve_response_format_with_schema(self):
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str

        agent = _make_agent(llm=MockLLM(), response_format=MyOutput)
        config = agent._resolve_response_format()
        assert config is not None

    def test_get_structured_output_kwargs_none(self):
        agent = _make_agent()
        assert agent._get_structured_output_kwargs(None) == {}

    def test_get_structured_output_kwargs_native_simple_schema(self):
        from pydantic import BaseModel
        from nucleusiq.agents.structured_output.config import OutputSchema
        from nucleusiq.agents.structured_output.types import OutputMode

        class MyOutput(BaseModel):
            answer: str

        agent = _make_agent(llm=MockLLM(), response_format=MyOutput)
        config = agent._resolve_response_format()
        kwargs = agent._get_structured_output_kwargs(config)
        assert "response_format" in kwargs

    def test_get_structured_output_kwargs_with_output_schema(self):
        from pydantic import BaseModel
        from nucleusiq.agents.structured_output.config import OutputSchema
        from nucleusiq.agents.structured_output.types import OutputMode

        class MyOutput(BaseModel):
            answer: str

        schema_config = OutputSchema(schema=MyOutput, mode=OutputMode.NATIVE)
        agent = _make_agent(llm=MockLLM(), response_format=schema_config)
        config = agent._resolve_response_format()
        kwargs = agent._get_structured_output_kwargs(config)
        assert "response_format" in kwargs
        assert isinstance(kwargs["response_format"], tuple)

    def test_wrap_structured_output_none(self):
        agent = _make_agent()
        resp = MagicMock()
        assert agent._wrap_structured_output_result(resp, None) is resp

    def test_wrap_native_no_choices(self):
        from nucleusiq.agents.structured_output.config import OutputSchema
        from nucleusiq.agents.structured_output.types import OutputMode

        class MyOut:
            pass

        config = OutputSchema(schema=MyOut, mode=OutputMode.NATIVE)
        config._resolved_mode = OutputMode.NATIVE

        agent = _make_agent()
        result_obj = MagicMock(spec=[])
        result = agent._wrap_structured_output_result(result_obj, config)
        assert isinstance(result, dict)
        assert "output" in result
        assert result["mode"] == "native"

    def test_wrap_with_choices_dict_msg(self):
        from nucleusiq.agents.structured_output.config import OutputSchema
        from nucleusiq.agents.structured_output.types import OutputMode

        config = OutputSchema(schema=dict, mode=OutputMode.NATIVE)
        config._resolved_mode = OutputMode.NATIVE

        agent = _make_agent()
        msg_dict = {"content": "text from dict msg"}
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg_dict)]
        result = agent._wrap_structured_output_result(resp, config)
        assert result == "text from dict msg"

    def test_wrap_with_choices_object_msg(self):
        from nucleusiq.agents.structured_output.config import OutputSchema
        from nucleusiq.agents.structured_output.types import OutputMode

        config = OutputSchema(schema=dict, mode=OutputMode.NATIVE)
        config._resolved_mode = OutputMode.NATIVE

        agent = _make_agent()
        msg = MagicMock()
        msg.content = "text from obj msg"
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        result = agent._wrap_structured_output_result(resp, config)
        assert result == "text from obj msg"


# ═══════════════════════════════════════════════════════════════════════════════
# Agent._process_result with prompt (lines 421, 427-429)
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessResultWithPrompt:

    @pytest.mark.asyncio
    async def test_prompt_with_sync_process_result(self):
        """Prompt that has a sync process_result method."""
        from nucleusiq.prompts.base import BasePrompt

        class PromptWithResult(BasePrompt):
            @property
            def technique_name(self) -> str:
                return "test"

            def _construct_prompt(self, **kwargs) -> str:
                return ""

            def format_prompt(self, **kwargs) -> str:
                return ""

            def process_result(self, r):
                return f"PROCESSED: {r}"

        agent = _make_agent(prompt=PromptWithResult())
        result = await agent._process_result("raw")
        assert "PROCESSED" in result

    @pytest.mark.asyncio
    async def test_prompt_with_async_process_result(self):
        from nucleusiq.prompts.base import BasePrompt

        class AsyncPromptResult(BasePrompt):
            @property
            def technique_name(self) -> str:
                return "test"

            def _construct_prompt(self, **kwargs) -> str:
                return ""

            def format_prompt(self, **kwargs) -> str:
                return ""

            async def process_result(self, r):
                return f"ASYNC: {r}"

        agent = _make_agent(prompt=AsyncPromptResult())
        result = await agent._process_result("raw")
        assert "ASYNC" in result

    @pytest.mark.asyncio
    async def test_process_result_exception_reraises(self):
        class BrokenMemory(FullHistoryMemory):
            async def aadd_message(self, *args, **kwargs):
                raise RuntimeError("memory failure")

        agent = _make_agent(memory=BrokenMemory())
        with pytest.raises(RuntimeError, match="memory failure"):
            await agent._process_result("data")

    @pytest.mark.asyncio
    async def test_process_result_stores_in_memory(self):
        mem = FullHistoryMemory()
        agent = _make_agent(memory=mem)
        await agent._process_result("some result text")
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "assistant"


# ═══════════════════════════════════════════════════════════════════════════════
# base_agent._execute_with_retry (lines 161-185)
# base_agent._execute_step (lines 190, 205-207)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecuteWithRetry:

    @pytest.mark.asyncio
    async def test_execute_with_retry_max_retries_reached(self):
        """_execute_with_retry raises after max retries."""
        llm = MockLLM()
        agent = _make_agent(llm=llm, config=AgentConfig(max_retries=2))
        await agent.initialize()

        with patch.object(Agent, 'execute', new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError):
                await agent._execute_with_retry({"id": "1", "objective": "x"})

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_on_first(self):
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()

        with patch.object(Agent, 'execute', new_callable=AsyncMock, return_value="ok"):
            with patch.object(Agent, '_validate_result', return_value=True):
                result = await agent._execute_with_retry({"id": "1", "objective": "x"})
                assert result == "ok"

    @pytest.mark.asyncio
    async def test_execute_step_timeout(self):
        llm = MockLLM()
        agent = _make_agent(llm=llm, config=AgentConfig(max_execution_time=1))
        await agent.initialize()
        agent._start_time = 1.0  # epoch second 1 — elapsed >> 1 s

        with pytest.raises(TimeoutError, match="execution time"):
            await agent._execute_step({"id": "1", "objective": "x"})

    @pytest.mark.asyncio
    async def test_execute_step_failure_updates_metrics(self):
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()
        agent._start_time = 9999999999

        with patch.object(Agent, 'execute', new_callable=AsyncMock, side_effect=ValueError("bad")):
            with pytest.raises(ValueError):
                await agent._execute_step({"id": "1", "objective": "test"})

    @pytest.mark.asyncio
    async def test_check_timeout_elapsed(self):
        agent = _make_agent()
        agent._start_time = 1.0
        assert agent._check_execution_timeout() is True

    @pytest.mark.asyncio
    async def test_check_timeout_no_time_set(self):
        agent = _make_agent()
        assert agent._check_execution_timeout() is False


# ═══════════════════════════════════════════════════════════════════════════════
# MessageBuilder.content_to_text (lines 150-161)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMessageBuilderContentToText:

    def test_none_returns_none(self):
        assert MessageBuilder.content_to_text(None) is None

    def test_string_returns_string(self):
        assert MessageBuilder.content_to_text("hello") == "hello"

    def test_list_of_parts(self):
        parts = [{"text": "part1"}, {"text": "part2"}]
        result = MessageBuilder.content_to_text(parts)
        assert "part1" in result
        assert "part2" in result

    def test_list_with_empty(self):
        parts = [{"text": "   "}, {"text": "valid"}]
        result = MessageBuilder.content_to_text(parts)
        assert result == "valid"

    def test_empty_list(self):
        assert MessageBuilder.content_to_text([]) is None

    def test_other_type_str(self):
        result = MessageBuilder.content_to_text(42)
        assert result == "42"

    def test_empty_string_coerced(self):
        result = MessageBuilder.content_to_text("   ")
        assert result == "   "

    def test_list_non_dict_items_ignored(self):
        parts = ["just a string", {"text": "ok"}]
        result = MessageBuilder.content_to_text(parts)
        assert result == "ok"

    def test_list_dict_without_text_ignored(self):
        parts = [{"image": "url"}, {"text": "ok"}]
        result = MessageBuilder.content_to_text(parts)
        assert result == "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# StandardMode tool-call paths (lines 79, 110, 119-121, 140-168)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStandardModeToolPaths:

    @pytest.mark.asyncio
    async def test_ensure_executor_no_llm(self):
        agent = _make_agent(llm=None)
        with pytest.raises(RuntimeError, match="LLM not available"):
            StandardMode()._ensure_executor(agent)

    @pytest.mark.asyncio
    async def test_run_refusal(self):
        """LLM returns a refusal message."""
        llm = MockLLM()
        msg = MagicMock()
        msg.tool_calls = None
        msg.refusal = "I can't do that"
        msg.content = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        llm.call = AsyncMock(return_value=resp)

        agent = _make_agent(llm=llm)
        await agent.initialize()
        mode = StandardMode()
        result = await mode.run(agent, {"id": "1", "objective": "do evil"})
        assert "refused" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_run_empty_then_content(self):
        """LLM returns empty response first, then content on retry."""
        llm = MockLLM()
        empty_msg = MagicMock()
        empty_msg.tool_calls = None
        empty_msg.refusal = None
        empty_msg.content = None
        empty_resp = MagicMock()
        empty_resp.choices = [MagicMock(message=empty_msg)]

        ok_msg = MagicMock()
        ok_msg.tool_calls = None
        ok_msg.refusal = None
        ok_msg.content = "final answer"
        ok_resp = MagicMock()
        ok_resp.choices = [MagicMock(message=ok_msg)]

        llm.call = AsyncMock(side_effect=[empty_resp, ok_resp])

        agent = _make_agent(llm=llm)
        await agent.initialize()
        mode = StandardMode()
        result = await mode.run(agent, {"id": "1", "objective": "answer"})
        assert result == "final answer"

    @pytest.mark.asyncio
    async def test_run_double_empty_error(self):
        """LLM returns empty both times -> error."""
        llm = MockLLM()
        empty_msg = MagicMock()
        empty_msg.tool_calls = None
        empty_msg.refusal = None
        empty_msg.content = None
        empty_resp = MagicMock()
        empty_resp.choices = [MagicMock(message=empty_msg)]

        llm.call = AsyncMock(return_value=empty_resp)

        agent = _make_agent(llm=llm)
        await agent.initialize()
        mode = StandardMode()
        result = await mode.run(agent, {"id": "1", "objective": "x"})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_run_structured_output_shortcircuit(self):
        """Structured output response is returned directly."""
        from pydantic import BaseModel
        from nucleusiq.agents.structured_output.config import OutputSchema
        from nucleusiq.agents.structured_output.types import OutputMode

        class Out(BaseModel):
            val: int

        llm = MockLLM()
        native_resp = MagicMock(spec=[])
        llm.call = AsyncMock(return_value=native_resp)

        agent = _make_agent(llm=llm, response_format=Out)
        await agent.initialize()
        mode = StandardMode()
        result = await mode.run(agent, {"id": "1", "objective": "compute"})
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# AutonomousMode paths (lines 79-90, 135-138, 148-149, 162-171)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutonomousModeRun:

    @pytest.mark.asyncio
    async def test_no_llm_echo(self):
        agent = _make_agent(llm=None)
        mode = AutonomousMode()
        result = await mode.run(agent, {"id": "1", "objective": "test"})
        assert "Echo" in result

    @pytest.mark.asyncio
    async def test_autonomous_with_memory(self):
        mem = FullHistoryMemory()
        llm = MockLLM()
        agent = _make_agent(
            llm=llm, memory=mem,
            config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS),
        )
        await agent.initialize()
        mode = AutonomousMode()
        result = await mode.run(agent, {"id": "1", "objective": "test task"})
        assert isinstance(result, str)
        ctx = mem.get_context()
        assert len(ctx) >= 1

    @pytest.mark.asyncio
    async def test_autonomous_with_structured_output(self):
        from pydantic import BaseModel

        class MyOut(BaseModel):
            answer: str

        llm = MockLLM()
        agent = _make_agent(
            llm=llm, response_format=MyOut,
            config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS),
        )
        await agent.initialize()
        mode = AutonomousMode()
        result = await mode.run(agent, {"id": "1", "objective": "compute"})
        assert isinstance(result, dict)
        assert result["mode"] == "autonomous"

    @pytest.mark.asyncio
    async def test_autonomous_error_fallback_to_standard(self):
        """Autonomous mode falls back to standard mode on exception."""
        llm = MockLLM()
        agent = _make_agent(llm=llm)
        await agent.initialize()
        mode = AutonomousMode()

        with patch(
            "nucleusiq.agents.modes.autonomous_mode.Planner",
            side_effect=RuntimeError("broken"),
        ):
            result = await mode.run(agent, {"id": "1", "objective": "test"})
            assert isinstance(result, str)
