"""
Comprehensive Agent tests with full code coverage.

Tests cover:
- All execution modes (DIRECT, STANDARD, AUTONOMOUS)
- Structured output
- State management (save/load)
- Error handling and edge cases
- Memory integration
- Delegation
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from typing import Any, Dict

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools import BaseTool

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
    async def test_autonomous_delegates_to_standard_mode(self):
        """Autonomous mode delegates to StandardMode for execution,
        then applies critique loop on the final result."""
        llm = MockLLM()
        agent = Agent(
            name="Test",
            role="Calc",
            objective="Math",
            llm=llm,
            config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, verbose=False),
        )
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="Calculate (5+3)*2"))
        assert isinstance(result, str)
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
# Planning Coverage (uses Agent.plan() — the built-in default planner)        #
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
            task,
            prompt=agent.prompt,
            role=agent.role,
            objective=agent.objective,
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
        plan = Plan(
            task=task,
            steps=[
                PlanStep(step=1, action="a"),
                PlanStep(step=2, action="b"),
            ],
        )
        messages = MessageBuilder.build(
            task,
            plan,
            prompt=agent.prompt,
            role=agent.role,
            objective=agent.objective,
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

        prompt = PromptWithProcessResult(
            template="x", system="You are helpful", user="Answer"
        )
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
