"""
Comprehensive Agent tests with full code coverage.

Tests cover:
- All execution modes (DIRECT, STANDARD, AUTONOMOUS)
- Planning (default, LLM-based, basic fallback)
- Plan execution (execute steps, tool steps, unknown actions, $step_N resolution)
- Structured output
- State management (save/load)
- Error handling and edge cases
- Memory integration
- Delegation
"""

import json
import os
import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan, PlanStep, PlanResponse, PlanStepResponse
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.planning.planner import Planner
from nucleusiq.agents.planning.plan_parser import PlanParser
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import BaseTool
from nucleusiq.llms.mock_llm import MockLLM


# --------------------------------------------------------------------------- #
# Mock Tools                                                                  #
# --------------------------------------------------------------------------- #


class MockAddTool(BaseTool):
    def __init__(self):
        super().__init__(name="add", description="Add two numbers")

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
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        }


class MockMultiplyTool(BaseTool):
    def __init__(self):
        super().__init__(name="multiply", description="Multiply two numbers")

    async def initialize(self) -> None:
        pass

    async def execute(self, a: int, b: int) -> int:
        return a * b

    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        }


# --------------------------------------------------------------------------- #
# Mock LLMs for specific scenarios                                           #
# --------------------------------------------------------------------------- #


class ContentOnlyMockLLM(MockLLM):
    """Always returns content, never tool calls."""

    async def call(self, **kwargs) -> Any:
        resp = await super().call(**kwargs)
        # Clear tool_calls so agent treats as content-only
        msg = resp.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg.tool_calls = None
        return resp


class PlanningMockLLM(MockLLM):
    """Returns create_plan tool call for autonomous planning."""

    def __init__(self, plan_steps: Optional[List[Dict[str, Any]]] = None, **kwargs):
        super().__init__(**kwargs)
        self._call_count = 0
        self.plan_steps = plan_steps or [
            {"step": 1, "action": "add", "args": {"a": 5, "b": 3}, "details": "5+3"},
            {"step": 2, "action": "multiply", "args": {"a": "$step_1", "b": 2}, "details": "8*2"},
        ]

    async def call(self, **kwargs) -> Any:
        self._call_count += 1
        tools = kwargs.get("tools") or []
        tool_names = []
        for t in tools:
            if isinstance(t, dict) and "function" in t:
                tool_names.append(t["function"].get("name", ""))
            else:
                tool_names.append(t.get("name", ""))

        # If create_plan tool is requested, return plan via tool call
        if "create_plan" in tool_names:
            plan_data = {"steps": self.plan_steps}
            tool_calls = [{
                "id": "call_plan_1",
                "type": "function",
                "function": {"name": "create_plan", "arguments": json.dumps(plan_data)},
            }]
            msg = self.Message(content=None, function_call=None, tool_calls=tool_calls)
            return self.LLMResponse([self.Choice(msg)])

        # Otherwise use parent behavior
        return await super().call(**kwargs)


class FailingMockLLM(MockLLM):
    """Raises error on call."""

    async def call(self, **kwargs) -> Any:
        raise RuntimeError("LLM call failed")


class EmptyResponseMockLLM(MockLLM):
    """Returns empty response (no content, no tool_calls)."""

    async def call(self, **kwargs) -> Any:
        msg = self.Message(content=None, function_call=None, tool_calls=None)
        return self.LLMResponse([self.Choice(msg)])


# --------------------------------------------------------------------------- #
# DIRECT Mode Coverage                                                        #
# --------------------------------------------------------------------------- #


class TestDirectModeCoverage:
    @pytest.mark.asyncio
    async def test_direct_with_llm_returns_content(self):
        llm = MockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hello"))
        assert result is not None
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_direct_no_llm_echo(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hello"))
        assert "Echo:" in result
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_direct_llm_error_fallback_echo(self):
        llm = FailingMockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hello"))
        assert "Echo:" in result
        assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_direct_empty_content_suggests_other_modes(self):
        llm = EmptyResponseMockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hello"))
        assert "No response" in result or "STANDARD" in result or "AUTONOMOUS" in result
        assert agent.state == AgentState.COMPLETED


# --------------------------------------------------------------------------- #
# STANDARD Mode Coverage                                                      #
# --------------------------------------------------------------------------- #


class TestStandardModeCoverage:
    @pytest.mark.asyncio
    async def test_standard_with_tools_executes_tool(self):
        llm = MockLLM()
        add_tool = MockAddTool()
        agent = Agent(
            name="Test",
            role="Calc",
            objective="Math",
            llm=llm,
            tools=[add_tool],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="What is 5 + 3?"))
        assert result is not None
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_standard_no_llm_echo(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            tools=[],
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hi"))
        assert "Echo:" in result

    @pytest.mark.asyncio
    async def test_standard_llm_error_returns_error_message(self):
        llm = FailingMockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hi"))
        assert "Error:" in result
        assert agent.state == AgentState.ERROR


# --------------------------------------------------------------------------- #
# AUTONOMOUS Mode Coverage                                                    #
# --------------------------------------------------------------------------- #


class TestAutonomousModeCoverage:
    @pytest.mark.asyncio
    async def test_autonomous_generates_and_executes_plan(self):
        llm = PlanningMockLLM(plan_steps=[
            {"step": 1, "action": "add", "args": {"a": 5, "b": 3}, "details": "5+3"},
            {"step": 2, "action": "multiply", "args": {"a": "$step_1", "b": 2}, "details": "8*2"},
        ])
        add_tool = MockAddTool()
        multiply_tool = MockMultiplyTool()
        agent = Agent(
            name="Test",
            role="Calc",
            objective="Math",
            llm=llm,
            tools=[add_tool, multiply_tool],
            config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Calculate (5+3)*2"))
        assert result == 16  # 5+3=8, 8*2=16
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_autonomous_no_llm_fallback_to_standard(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hi"))
        assert "Echo:" in result

    @pytest.mark.asyncio
    async def test_autonomous_planning_fails_fallback_to_basic_plan(self):
        llm = FailingMockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Hi"))
        # Falls back to standard mode on planning failure
        assert result is not None


# --------------------------------------------------------------------------- #
# Plan Execution Coverage                                                      #
# --------------------------------------------------------------------------- #


class TestPlanExecutionCoverage:
    @pytest.mark.asyncio
    async def test_execute_plan_execute_action(self):
        llm = MockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="Hello")
        plan = Plan(task=task, steps=[
            PlanStep(step=1, action="execute", task=task),
        ])
        result = await Planner(agent).execute_plan(task, plan)
        assert result is not None
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_plan_tool_action_with_args(self):
        llm = MockLLM()
        add_tool = MockAddTool()
        agent = Agent(
            name="Test",
            role="Calc",
            objective="Math",
            llm=llm,
            tools=[add_tool],
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="Add numbers")
        plan = Plan(task=task, steps=[
            PlanStep(step=1, action="add", args={"a": 10, "b": 20}, task=task),
        ])
        result = await Planner(agent).execute_plan(task, plan)
        assert result == 30
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_plan_step_resolution(self):
        llm = MockLLM()
        add_tool = MockAddTool()
        multiply_tool = MockMultiplyTool()
        agent = Agent(
            name="Test",
            role="Calc",
            objective="Math",
            llm=llm,
            tools=[add_tool, multiply_tool],
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="(5+3)*2")
        plan = Plan(task=task, steps=[
            PlanStep(step=1, action="add", args={"a": 5, "b": 3}),
            PlanStep(step=2, action="multiply", args={"a": "$step_1", "b": 2}),
        ])
        result = await Planner(agent).execute_plan(task, plan)
        assert result == 16
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_plan_unknown_action_skipped(self):
        llm = MockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            tools=[],
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="Test")
        plan = Plan(task=task, steps=[
            PlanStep(step=1, action="unknown_action", args={}),
        ])
        result = await Planner(agent).execute_plan(task, plan)
        assert "Skipped unknown action" in str(result)
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_plan_list_format(self):
        llm = MockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = {"id": "t1", "objective": "Test"}
        plan_list = [{"step": 1, "action": "execute", "task": task}]
        result = await Planner(agent).execute_plan(task, plan_list)
        assert result is not None


# --------------------------------------------------------------------------- #
# Planning Coverage                                                           #
# --------------------------------------------------------------------------- #


class TestPlanningCoverage:
    @pytest.mark.asyncio
    async def test_plan_default_single_step(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = {"id": "t1", "objective": "Do something"}
        plan = await agent.plan(task)
        assert isinstance(plan, Plan)
        assert len(plan) == 1
        assert plan.steps[0].action == "execute"

    @pytest.mark.asyncio
    async def test_create_basic_plan(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = {"id": "t1", "objective": "Test"}
        plan = await agent.plan(task)
        assert len(plan) == 1
        assert plan.steps[0].action == "execute"

    @pytest.mark.asyncio
    async def test_get_context_with_task_dict(self):
        agent = Agent(
            name="Test",
            role="Calculator",
            objective="Math",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = {"id": "t1", "objective": "5+3"}
        context = await Planner(agent).get_context(task)
        assert "task" in context
        assert context["agent_role"] == "Calculator"
        assert context["agent_objective"] == "Math"
        assert "timestamp" in context

    @pytest.mark.asyncio
    async def test_get_context_with_task_object(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="Test")
        context = await Planner(agent).get_context(task)
        assert context["task"]["id"] == "t1"
        assert context["task"]["objective"] == "Test"


# --------------------------------------------------------------------------- #
# Message Building & Formatting                                               #
# --------------------------------------------------------------------------- #


class TestMessageBuildingCoverage:
    @pytest.mark.asyncio
    async def test_build_messages_role_objective_no_prompt(self):
        agent = Agent(
            name="Test",
            role="Assistant",
            objective="Help users",
            llm=None,
            prompt=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="Hello")
        messages = MessageBuilder.build(
            task, prompt=agent.prompt, role=agent.role, objective=agent.objective,
        )
        assert any(m.role == "system" for m in messages)
        assert any("Assistant" in (m.content or "") for m in messages)
        assert any("Hello" in (m.content or "") for m in messages)

    @pytest.mark.asyncio
    async def test_build_messages_with_plan_multi_step(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="Task")
        plan = Plan(task=task, steps=[
            PlanStep(step=1, action="a"),
            PlanStep(step=2, action="b"),
        ])
        messages = MessageBuilder.build(
            task, plan, prompt=agent.prompt, role=agent.role, objective=agent.objective,
        )
        assert any("Execution Plan" in (m.content or "") for m in messages)

    @pytest.mark.asyncio
    async def test_format_plan(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        plan = Plan(
            task=Task(id="t1", objective="X"),
            steps=[
                PlanStep(step=1, action="execute", details="Step 1"),
                PlanStep(step=2, action="verify", details="Step 2"),
            ],
        )
        formatted = MessageBuilder.format_plan(plan)
        assert "Step 1" in formatted
        assert "execute" in formatted
        assert "Step 2" in formatted
        assert "verify" in formatted


# --------------------------------------------------------------------------- #
# Plan Parsing Coverage                                                        #
# --------------------------------------------------------------------------- #


class TestPlanParsingCoverage:
    def test_parse_plan_response_json(self):
        parser = PlanParser()
        response = '{"steps": [{"step": 1, "action": "add", "args": {"a": 1, "b": 2}, "details": "Add"}]}'
        plan_resp = parser.parse(response)
        assert len(plan_resp.steps) == 1
        assert plan_resp.steps[0].action == "add"
        assert plan_resp.steps[0].args == {"a": 1, "b": 2}

    def test_parse_plan_response_markdown_json(self):
        parser = PlanParser()
        response = '```json\n{"steps": [{"step": 1, "action": "execute", "args": {}, "details": ""}]}\n```'
        plan_resp = parser.parse(response)
        assert len(plan_resp.steps) == 1
        assert plan_resp.steps[0].action == "execute"

    def test_parse_plan_response_fallback_text(self):
        parser = PlanParser()
        response = "Step 1: add numbers\n  Use calculator\nStep 2: verify"
        plan_resp = parser.parse(response)
        assert len(plan_resp.steps) >= 1
        assert plan_resp.steps[0].step == 1


# --------------------------------------------------------------------------- #
# Execute & Validation                                                        #
# --------------------------------------------------------------------------- #


class TestExecuteValidationCoverage:
    @pytest.mark.asyncio
    async def test_execute_unknown_mode_raises(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        agent.config.execution_mode = "invalid_mode"  # type: ignore
        with pytest.raises(ValueError, match="Unknown execution mode"):
            await agent.execute(Task(id="t1", objective="Test"))

    @pytest.mark.asyncio
    async def test_execute_accepts_task_dict(self):
        llm = MockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        result = await agent.execute({"id": "t1", "objective": "Hi"})
        assert result is not None


# --------------------------------------------------------------------------- #
# State & Delegation                                                           #
# --------------------------------------------------------------------------- #


class TestStateDelegationCoverage:
    @pytest.mark.asyncio
    async def test_save_state(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        state = await agent.save_state()
        assert "id" in state
        assert "name" in state
        assert "state" in state
        assert "metrics" in state
        assert "timestamp" in state

    @pytest.mark.asyncio
    async def test_load_state(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        state = {
            "id": str(agent.id),
            "name": "Test",
            "state": AgentState.COMPLETED,
            "metrics": agent.metrics.model_dump(),
            "current_task": None,
            "timestamp": "2024-01-01T00:00:00",
        }
        await agent.load_state(state)
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_delegate_task(self):
        llm = MockLLM()
        agent1 = Agent(
            name="Agent1",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        agent2 = Agent(
            name="Agent2",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(verbose=False),
        )
        await agent1.initialize()
        await agent2.initialize()
        result = await agent1.delegate_task({"id": "t1", "objective": "Hi"}, agent2)
        assert result is not None


# --------------------------------------------------------------------------- #
# Helper Methods                                                               #
# --------------------------------------------------------------------------- #


class TestHelperMethodsCoverage:
    @pytest.mark.asyncio
    async def test_validate_task(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        assert agent._validate_task({"id": "1", "objective": "X"}) is True
        assert agent._validate_task({"id": "1"}) is False
        assert agent._validate_task({"objective": "X"}) is False
        assert agent._validate_task({}) is False

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        add_tool = MockAddTool()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            tools=[add_tool],
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        result = await agent._execute_tool("add", {"a": 3, "b": 4})
        assert result == 7

    @pytest.mark.asyncio
    async def test_execute_tool_not_found_raises(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            tools=[],
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        with pytest.raises(ValueError, match="Tool not found"):
            await agent._execute_tool("nonexistent", {})

    def test_get_plan_schema(self):
        schema = Planner.get_plan_schema()
        assert "steps" in schema.get("properties", {})
        assert "required" in schema

    def test_get_plan_function_spec(self):
        spec = Planner.get_plan_function_spec()
        assert "function" in spec
        assert spec["function"]["name"] == "create_plan"


# --------------------------------------------------------------------------- #
# Error Handling & Process Result                                             #
# --------------------------------------------------------------------------- #


class TestErrorHandlingCoverage:
    @pytest.mark.asyncio
    async def test_handle_error(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        await agent._handle_error(ValueError("Test error"), {"key": "value"})
        assert agent.state == AgentState.ERROR
        assert agent.metrics.error_count >= 1

    @pytest.mark.asyncio
    async def test_process_result_no_memory_no_prompt(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            memory=None,
            prompt=None,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        result = await agent._process_result("output")
        assert result == "output"

    @pytest.mark.asyncio
    async def test_process_result_with_prompt_process_result(self):
        # Create a prompt subclass with process_result (agent uses getattr)
        from nucleusiq.prompts.base import BasePrompt

        class PromptWithProcessResult(BasePrompt):
            @property
            def technique_name(self) -> str:
                return "test"
            def _construct_prompt(self, **kwargs) -> str:
                return kwargs.get("user", "")
            def process_result(self, x):
                return f"Processed: {x}"

        prompt = PromptWithProcessResult(template="x", system="You are helpful", user="Answer")
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            prompt=prompt,
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        result = await agent._process_result("raw")
        assert result == "Processed: raw"


# --------------------------------------------------------------------------- #
# Step Failure & Missing Args                                                 #
# --------------------------------------------------------------------------- #


class TestStepFailureCoverage:
    @pytest.mark.asyncio
    async def test_execute_plan_step_failure_sets_error(self):
        llm = MockLLM()
        add_tool = MockAddTool()
        agent = Agent(
            name="Test",
            role="Calc",
            objective="Math",
            llm=llm,
            tools=[add_tool],
            config=AgentConfig(verbose=False),
        )
        await agent.initialize()
        task = Task(id="t1", objective="Add")
        # Step with missing required args - will fail
        plan = Plan(task=task, steps=[
            PlanStep(step=1, action="add", args={}),  # Missing a, b
        ])
        result = await Planner(agent).execute_plan(task, plan)
        assert "Error:" in str(result)
        assert agent.state == AgentState.ERROR


# --------------------------------------------------------------------------- #
# Structured Output (basic - no OpenAI needed)                                 #
# --------------------------------------------------------------------------- #


class TestStructuredOutputCoverage:
    def test_resolve_response_format_none(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            response_format=None,
            config=AgentConfig(verbose=False),
        )
        config = agent._resolve_response_format()
        assert config is None

    def test_get_structured_output_kwargs_none_config(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        kwargs = agent._get_structured_output_kwargs(None)
        assert kwargs == {}

    def test_wrap_structured_output_result_none_config(self):
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=None,
            config=AgentConfig(verbose=False),
        )
        result = agent._wrap_structured_output_result("raw_response", None)
        assert result == "raw_response"


# --------------------------------------------------------------------------- #
# Deprecated Method                                                            #
# --------------------------------------------------------------------------- #


class TestDeprecatedCoverage:
    @pytest.mark.asyncio
    async def test_standard_mode_execution(self):
        """Test that standard mode execution works (replaces deprecated _execute_direct)."""
        from nucleusiq.agents.modes.standard_mode import StandardMode

        llm = MockLLM()
        agent = Agent(
            name="Test",
            role="A",
            objective="B",
            llm=llm,
            config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False),
        )
        await agent.initialize()
        result = await StandardMode().run(agent, Task(id="t1", objective="Hi"))
        assert result is not None
