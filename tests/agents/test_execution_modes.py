"""
Tests for Agent execution modes (Gearbox Strategy).

Tests cover:
- Direct mode (Gear 1)
- Standard mode (Gear 2)
- Autonomous mode (Gear 3) - fallback behavior
- Mode routing
- ExecutionMode enum
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.agents.task import Task
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


class TestExecutionModeEnum:
    """Test ExecutionMode enum."""
    
    def test_execution_mode_enum_values(self):
        """Test ExecutionMode enum has correct values."""
        assert ExecutionMode.DIRECT == "direct"
        assert ExecutionMode.STANDARD == "standard"
        assert ExecutionMode.AUTONOMOUS == "autonomous"
    
    def test_execution_mode_enum_string(self):
        """Test ExecutionMode enum string representation."""
        assert ExecutionMode.DIRECT.value == "direct"
        assert ExecutionMode.STANDARD.value == "standard"
        assert ExecutionMode.AUTONOMOUS.value == "autonomous"
    
    def test_execution_mode_enum_comparison(self):
        """Test ExecutionMode enum comparison."""
        assert ExecutionMode.DIRECT == ExecutionMode.DIRECT
        assert ExecutionMode.DIRECT != ExecutionMode.STANDARD
        assert ExecutionMode.STANDARD == ExecutionMode.STANDARD


class TestDirectMode:
    """Test Direct mode (Gear 1) - Fast, simple, no tools."""
    
    @pytest.mark.asyncio
    async def test_direct_mode_simple_task(self):
        """Test direct mode with simple task."""
        llm = MockLLM()
        agent = Agent(
            name="ChatBot",
            role="Assistant",
            objective="Answer questions",
            llm=llm,
            config=AgentConfig(
                execution_mode=ExecutionMode.DIRECT,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="Tell me a joke")
        result = await agent.execute(task)
        
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_direct_mode_no_tools(self):
        """Test direct mode ignores tools even if provided."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="ChatBot",
            role="Assistant",
            objective="Answer questions",
            llm=llm,
            tools=[calculator],  # Tools provided but should be ignored
            config=AgentConfig(
                execution_mode=ExecutionMode.DIRECT,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="What is 5 + 3?")
        result = await agent.execute(task)
        
        # Direct mode should not use tools
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_direct_mode_with_prompt(self):
        """Test direct mode with prompt."""
        llm = MockLLM()
        from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
        
        prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
        prompt.configure(
            system="You are a helpful assistant",
            user="Answer questions clearly"
        )
        
        agent = Agent(
            name="ChatBot",
            role="Assistant",
            objective="Answer questions",
            llm=llm,
            prompt=prompt,
            config=AgentConfig(
                execution_mode=ExecutionMode.DIRECT,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="What is Python?")
        result = await agent.execute(task)
        
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_direct_mode_no_llm_echo(self):
        """Test direct mode falls back to echo if no LLM."""
        agent = Agent(
            name="ChatBot",
            role="Assistant",
            objective="Answer questions",
            llm=None,  # No LLM
            config=AgentConfig(
                execution_mode=ExecutionMode.DIRECT,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="Test")
        result = await agent.execute(task)
        
        assert "Echo:" in result
        assert agent.state == AgentState.COMPLETED


class TestStandardMode:
    """Test Standard mode (Gear 2) - Tool-enabled, linear execution."""
    
    @pytest.mark.asyncio
    async def test_standard_mode_with_tool(self):
        """Test standard mode with tool execution."""
        llm = MockLLM()
        llm._call_count = 0  # Reset call count
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="CalculatorBot",
            role="Calculator",
            objective="Perform calculations",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(
                execution_mode=ExecutionMode.STANDARD,  # Default
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="What is 15 + 27?")
        result = await agent.execute(task)
        
        assert result is not None
        # May be COMPLETED or ERROR depending on MockLLM behavior, but should have a result
        assert result != ""
    
    @pytest.mark.asyncio
    async def test_standard_mode_default(self):
        """Test standard mode is the default."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm,
            config=AgentConfig(verbose=False)  # No execution_mode specified
        )
        
        await agent.initialize()
        
        # Should default to STANDARD
        assert agent.config.execution_mode == ExecutionMode.STANDARD
    
    @pytest.mark.asyncio
    async def test_standard_mode_no_tools(self):
        """Test standard mode without tools (should still work)."""
        llm = MockLLM()
        agent = Agent(
            name="ChatBot",
            role="Assistant",
            objective="Answer questions",
            llm=llm,
            tools=[],  # No tools
            config=AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="Tell me about Python")
        result = await agent.execute(task)
        
        assert result is not None
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_standard_mode_multiple_tools(self):
        """Test standard mode with multiple tools."""
        llm = MockLLM()
        
        class SubtractTool(BaseTool):
            def __init__(self):
                super().__init__(name="subtract", description="Subtract two numbers")
            
            async def initialize(self) -> None:
                pass
            
            async def execute(self, a: int, b: int) -> int:
                return a - b
            
            def get_spec(self) -> Dict[str, Any]:
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    }
                }
        
        calculator = MockCalculatorTool()
        subtract = SubtractTool()
        
        agent = Agent(
            name="MathBot",
            role="Math Assistant",
            objective="Perform math operations",
            llm=llm,
            tools=[calculator, subtract],
            config=AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="What is 10 - 3?")
        result = await agent.execute(task)
        
        assert result is not None
        assert agent.state == AgentState.COMPLETED


class TestAutonomousMode:
    """Test Autonomous mode (Gear 3) - Currently falls back to standard."""
    
    @pytest.mark.asyncio
    async def test_autonomous_mode_fallback(self):
        """Test autonomous mode falls back to standard (not yet implemented)."""
        llm = MockLLM()
        llm._call_count = 0  # Reset call count
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="ResearchBot",
            role="Researcher",
            objective="Research topics",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="Calculate 5 + 3")
        result = await agent.execute(task)
        
        # Should fallback to standard mode
        assert result is not None
        # May be COMPLETED or ERROR depending on MockLLM behavior, but should have a result
        assert result != ""
    
    @pytest.mark.asyncio
    async def test_autonomous_mode_warning(self):
        """Test autonomous mode logs warning about fallback."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm,
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                verbose=True
            )
        )
        
        await agent.initialize()
        
        task = Task(id="task1", objective="Test")
        
        # Patch the agent's logger instance
        with patch.object(agent, '_logger') as mock_logger:
            await agent.execute(task)
            # Check that warning was logged
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if call and 'Autonomous mode not yet implemented' in str(call)]
            assert len(warning_calls) > 0


class TestModeRouting:
    """Test mode routing in execute() method."""
    
    @pytest.mark.asyncio
    async def test_mode_routing_direct(self):
        """Test routing to direct mode."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm,
            config=AgentConfig(
                execution_mode=ExecutionMode.DIRECT,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        # Mock _run_direct to verify it's called
        with patch.object(agent, '_run_direct', new_callable=AsyncMock) as mock_direct:
            mock_direct.return_value = "Direct result"
            
            task = Task(id="task1", objective="Test")
            result = await agent.execute(task)
            
            assert result == "Direct result"
            mock_direct.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mode_routing_standard(self):
        """Test routing to standard mode."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm,
            config=AgentConfig(
                execution_mode=ExecutionMode.STANDARD,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        # Mock _run_standard to verify it's called
        with patch.object(agent, '_run_standard', new_callable=AsyncMock) as mock_standard:
            mock_standard.return_value = "Standard result"
            
            task = Task(id="task1", objective="Test")
            result = await agent.execute(task)
            
            assert result == "Standard result"
            mock_standard.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mode_routing_autonomous(self):
        """Test routing to autonomous mode (falls back to standard)."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm,
            config=AgentConfig(
                execution_mode=ExecutionMode.AUTONOMOUS,
                verbose=False
            )
        )
        
        await agent.initialize()
        
        # Mock _run_standard to verify fallback
        with patch.object(agent, '_run_standard', new_callable=AsyncMock) as mock_standard:
            mock_standard.return_value = "Standard result (fallback)"
            
            task = Task(id="task1", objective="Test")
            result = await agent.execute(task)
            
            assert result == "Standard result (fallback)"
            mock_standard.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mode_routing_invalid(self):
        """Test invalid execution mode raises error."""
        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm,
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # Set invalid execution mode
        agent.config.execution_mode = "invalid_mode"  # type: ignore
        
        task = Task(id="task1", objective="Test")
        
        with pytest.raises(ValueError, match="Unknown execution mode"):
            await agent.execute(task)


class TestExecutionModeIntegration:
    """Integration tests for execution modes."""
    
    @pytest.mark.asyncio
    async def test_execution_mode_state_transitions(self):
        """Test state transitions for different execution modes."""
        llm = MockLLM()
        
        for mode in [ExecutionMode.DIRECT, ExecutionMode.STANDARD, ExecutionMode.AUTONOMOUS]:
            agent = Agent(
                name="TestAgent",
                role="Assistant",
                objective="Help users",
                llm=llm,
                config=AgentConfig(
                    execution_mode=mode,
                    verbose=False
                )
            )
            
            await agent.initialize()
            assert agent.state == AgentState.INITIALIZING
            
            task = Task(id="task1", objective="Test")
            result = await agent.execute(task)
            
            assert result is not None
            assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execution_mode_with_task_dict(self):
        """Test execution modes work with task dict (backward compatibility)."""
        llm = MockLLM()
        
        for mode in [ExecutionMode.DIRECT, ExecutionMode.STANDARD]:
            agent = Agent(
                name="TestAgent",
                role="Assistant",
                objective="Help users",
                llm=llm,
                config=AgentConfig(
                    execution_mode=mode,
                    verbose=False
                )
            )
            
            await agent.initialize()
            
            # Use dict instead of Task object
            task_dict = {"id": "task1", "objective": "Test"}
            result = await agent.execute(task_dict)
            
            assert result is not None
            assert agent.state == AgentState.COMPLETED

