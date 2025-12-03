"""
Comprehensive tests for the Agent class.

Tests cover:
- Agent initialization
- Agent execution
- Agent planning
- State transitions
- Error handling
- Integration workflows
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, AgentState, AgentMetrics
from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.core.tools import BaseTool
from nucleusiq.llms.mock_llm import MockLLM


class MockCalculatorTool(BaseTool):
    """Mock calculator tool for testing."""
    
    def __init__(self):
        super().__init__(
            name="add",
            description="Add two numbers together",
        )
    
    async def initialize(self) -> None:
        pass
    
    async def execute(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def get_spec(self) -> Dict[str, Any]:
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
            }
        }


class TestAgentInitialization:
    """Test Agent initialization."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization_success(self):
        """Test successful agent initialization."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        assert agent.state == AgentState.INITIALIZING
        assert agent.name == "TestAgent"
        assert agent.role == "Assistant"
        assert agent.objective == "Help users"
        assert agent.llm == llm
    
    @pytest.mark.asyncio
    async def test_agent_initialization_with_tools(self):
        """Test agent initialization with tools."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        assert agent.state == AgentState.INITIALIZING
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "add"
    
    @pytest.mark.asyncio
    async def test_agent_initialization_with_prompt(self):
        """Test agent initialization with prompt."""
        llm = MockLLM()
        prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
        prompt.configure(
            system="You are a helpful assistant.",
            user="Answer the user's question."
        )
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            prompt=prompt,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        assert agent.state == AgentState.INITIALIZING
        assert agent.prompt == prompt
    
    @pytest.mark.asyncio
    async def test_agent_initialization_without_llm(self):
        """Test agent initialization without LLM (should still work)."""
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=None,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        assert agent.state == AgentState.INITIALIZING
        assert agent.llm is None
    
    @pytest.mark.asyncio
    async def test_agent_initialization_tool_initialization_error(self):
        """Test agent initialization when tool initialization fails."""
        llm = MockLLM()
        
        # Create a tool that fails to initialize
        class FailingTool(BaseTool):
            def __init__(self):
                super().__init__(
                    name="failing",
                    description="Fails"
                )
            
            async def initialize(self) -> None:
                raise RuntimeError("Tool initialization failed")
            
            async def execute(self, **kwargs):
                pass
            
            def get_spec(self):
                return {"name": "failing", "description": "Fails"}
        
        failing_tool = FailingTool()
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            tools=[failing_tool],
            config=AgentConfig(verbose=False)
        )
        
        with pytest.raises(RuntimeError):
            await agent.initialize()
        
        assert agent.state == AgentState.ERROR


class TestAgentPlan:
    """Test Agent planning functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_plan_default_implementation(self):
        """Test default plan() implementation."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Test task"}
        plan = await agent.plan(task)

        assert isinstance(plan, Plan)
        assert len(plan) == 1
        assert plan.steps[0].step == 1
        assert plan.steps[0].action == "execute"
        # Check that task is in the plan
        assert plan.task.id == task["id"]
        assert plan.task.objective == task["objective"]
    
    @pytest.mark.asyncio
    async def test_agent_plan_with_different_tasks(self):
        """Test plan() with different task formats."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # Test with minimal task
        task1 = {"id": "task1", "objective": "Simple task"}
        plan1 = await agent.plan(task1)
        assert isinstance(plan1, Plan)
        assert len(plan1) == 1

        # Test with additional fields
        task2 = {
            "id": "task2",
            "objective": "Complex task",
            "context": {"key": "value"}
        }
        plan2 = await agent.plan(task2)
        assert isinstance(plan2, Plan)
        assert len(plan2) == 1
        # Check that task is in the plan
        assert plan2.task.id == task2["id"]
        assert plan2.task.objective == task2["objective"]
    
    @pytest.mark.asyncio
    async def test_agent_create_basic_plan(self):
        """Test _create_basic_plan() method."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Test task"}
        plan = await agent._create_basic_plan(task)

        assert isinstance(plan, Plan)
        assert len(plan) == 1
        assert plan.steps[0].step == 1
        assert plan.steps[0].action == "execute"
        # Check that task is in the plan
        assert plan.task.id == task["id"]
        assert plan.task.objective == task["objective"]
    
    @pytest.mark.asyncio
    async def test_agent_get_context(self):
        """Test _get_context() method."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Calculate 5 + 3"}
        context = await agent._get_context(task)
        
        assert "task" in context
        assert "agent_role" in context
        assert "agent_objective" in context
        assert "timestamp" in context
        assert context["task"] == task
        assert context["agent_role"] == "Calculator"
        assert context["agent_objective"] == "Perform calculations"


class TestAgentExecute:
    """Test Agent execution functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_execute_without_llm(self):
        """Test execute() without LLM (echo mode)."""
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=None,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Test task"}
        result = await agent.execute(task)
        
        assert result == "Echo: Test task"
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_execute_with_llm_no_tools(self):
        """Test execute() with LLM but no tools."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Hello, world!"}
        result = await agent.execute(task)
        
        # MockLLM will echo the content
        assert "Echo:" in result or "Dummy Model" in result
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_execute_with_tools(self):
        """Test execute() with tools (function calling)."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "What is 5 + 3?"}
        result = await agent.execute(task)
        
        # MockLLM will call the tool and return final answer
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_execute_state_transitions(self):
        """Test state transitions during execution."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        assert agent.state == AgentState.INITIALIZING
        
        task = {"id": "task1", "objective": "Test task"}
        await agent.execute(task)
        
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_execute_with_prompt(self):
        """Test execute() with prompt."""
        llm = MockLLM()
        prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
        prompt.configure(
            system="You are a helpful assistant.",
            user="Answer the user's question."
        )
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            prompt=prompt,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Hello!"}
        result = await agent.execute(task)
        
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_execute_error_handling(self):
        """Test execute() error handling."""
        # Create a mock LLM that raises an error
        class FailingLLM(MockLLM):
            async def call(self, **kwargs):
                raise RuntimeError("LLM call failed")
        
        llm = FailingLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Test task"}
        result = await agent.execute(task)
        
        # Should fallback to echo mode on error
        assert "Echo:" in result
        assert agent.state == AgentState.ERROR


class TestAgentStateTransitions:
    """Test Agent state transitions."""
    
    @pytest.mark.asyncio
    async def test_agent_state_initialization(self):
        """Test state during initialization."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        # Before initialization
        assert agent.state == AgentState.INITIALIZING
        
        await agent.initialize()
        
        # After initialization
        assert agent.state == AgentState.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_agent_state_execution(self):
        """Test state during execution."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Test task"}
        
        # Execute should transition to EXECUTING, then COMPLETED
        await agent.execute(task)
        
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_state_error(self):
        """Test state on error."""
        # Create a tool that fails during execution
        class FailingTool(BaseTool):
            def __init__(self):
                super().__init__(
                    name="failing",
                    description="Fails"
                )
            
            async def initialize(self):
                pass
            
            async def execute(self, **kwargs):
                raise ValueError("Tool execution failed")
            
            def get_spec(self):
                return {"name": "failing", "description": "Fails"}
        
        llm = MockLLM()
        failing_tool = FailingTool()
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            tools=[failing_tool],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # Execution will fail when tool is called
        task = {"id": "task1", "objective": "Use failing tool"}
        result = await agent.execute(task)
        
        # Should handle error gracefully
        assert agent.state in [AgentState.ERROR, AgentState.COMPLETED]


class TestAgentErrorHandling:
    """Test Agent error handling."""
    
    @pytest.mark.asyncio
    async def test_agent_validate_task(self):
        """Test _validate_task() method."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # Valid task
        valid_task = {"id": "task1", "objective": "Test"}
        assert agent._validate_task(valid_task) is True
        
        # Missing 'id'
        invalid_task1 = {"objective": "Test"}
        assert agent._validate_task(invalid_task1) is False
        
        # Missing 'objective'
        invalid_task2 = {"id": "task1"}
        assert agent._validate_task(invalid_task2) is False
        
        # Missing both
        invalid_task3 = {}
        assert agent._validate_task(invalid_task3) is False
    
    @pytest.mark.asyncio
    async def test_agent_tool_not_found_error(self):
        """Test error when tool is not found."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # MockLLM will try to call a tool that doesn't exist
        # This should be handled gracefully
        task = {"id": "task1", "objective": "Use unknown tool"}
        result = await agent.execute(task)
        
        # Should handle error (either echo or error state)
        assert result is not None


class TestAgentPlanIntegration:
    """Test Agent plan integration."""
    
    @pytest.mark.asyncio
    async def test_agent_execute_with_planning_enabled(self):
        """Test execute() with use_planning=True."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(use_planning=True, verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "What is 5 + 3?"}
        result = await agent.execute(task)
        
        # Should execute successfully (plan will be created)
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_execute_with_planning_disabled(self):
        """Test execute() with use_planning=False (default)."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(use_planning=False, verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "What is 5 + 3?"}
        result = await agent.execute(task)
        
        # Should execute directly without planning
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_execute_plan_method(self):
        """Test _execute_plan() method."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "What is 5 + 3?"}
        plan = [
            {"step": 1, "action": "execute", "task": task},
        ]
        
        result = await agent._execute_plan(task, plan)
        
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_build_messages_with_plan(self):
        """Test _build_messages() with plan."""
        llm = MockLLM()
        prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
        prompt.configure(
            system="You are a helpful assistant.",
            user="Answer questions."
        )
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            prompt=prompt,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Test"}
        plan = [
            {"step": 1, "action": "execute", "task": task},
            {"step": 2, "action": "execute", "task": task},
        ]
        
        messages = agent._build_messages(task, plan)
        
        # Should include system, user template, plan, and task objective
        assert len(messages) >= 3
        assert any(m.get("role") == "system" for m in messages)
        assert any("Execution Plan" in m.get("content", "") for m in messages)
        assert any(task["objective"] in m.get("content", "") for m in messages)
    
    @pytest.mark.asyncio
    async def test_agent_build_messages_without_plan(self):
        """Test _build_messages() without plan."""
        llm = MockLLM()
        prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
        prompt.configure(
            system="You are a helpful assistant.",
            user="Answer questions."
        )
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            prompt=prompt,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        task = {"id": "task1", "objective": "Test"}
        messages = agent._build_messages(task)
        
        # Should include system, user template, and task objective (no plan)
        assert len(messages) >= 2
        assert any(m.get("role") == "system" for m in messages)
        assert any(task["objective"] in m.get("content", "") for m in messages)
        # Should not have plan
        assert not any("Execution Plan" in m.get("content", "") for m in messages)


class TestAgentIntegration:
    """Integration tests for Agent workflow."""
    
    @pytest.mark.asyncio
    async def test_agent_full_workflow(self):
        """Test complete agent workflow: initialize -> plan -> execute."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        # Step 1: Initialize
        await agent.initialize()
        assert agent.state == AgentState.INITIALIZING
        
        # Step 2: Plan
        task = {"id": "task1", "objective": "What is 5 + 3?"}
        plan = await agent.plan(task)
        assert isinstance(plan, Plan)
        assert len(plan) == 1
        assert plan.steps[0].step == 1
        
        # Step 3: Execute
        result = await agent.execute(task)
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_multiple_executions(self):
        """Test agent handling multiple task executions."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # Execute multiple tasks
        task1 = {"id": "task1", "objective": "What is 2 + 2?"}
        result1 = await agent.execute(task1)
        assert agent.state == AgentState.COMPLETED
        
        # Reset state for next execution
        agent.state = AgentState.INITIALIZING
        
        task2 = {"id": "task2", "objective": "What is 3 + 4?"}
        result2 = await agent.execute(task2)
        assert agent.state == AgentState.COMPLETED
        
        assert result1 is not None
        assert result2 is not None
    
    @pytest.mark.asyncio
    async def test_agent_metrics_tracking(self):
        """Test that agent tracks metrics."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # Initial metrics
        assert agent.metrics.error_count == 0
        
        # Execute task
        task = {"id": "task1", "objective": "Test task"}
        await agent.execute(task)
        
        # Metrics should be accessible
        assert isinstance(agent.metrics, AgentMetrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

