"""
Tests for Executor component.

Tests cover:
- Executor initialization
- Tool execution
- Error handling
- Plan step execution
- Context passing
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import json
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

from nucleusiq.agents.components.executor import Executor
from nucleusiq.agents.plan import PlanStep
from nucleusiq.core.tools import BaseTool
from nucleusiq.core.llms.mock_llm import MockLLM


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


class MockFailingTool(BaseTool):
    """Mock tool that fails execution."""
    
    def __init__(self):
        super().__init__(
            name="failing_tool",
            description="A tool that always fails",
        )
    
    async def initialize(self) -> None:
        pass
    
    async def execute(self, **kwargs) -> Any:
        """Always raises an error."""
        raise ValueError("Tool execution failed")
    
    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }


class MockNativeTool(BaseTool):
    """Mock native tool (has is_native=True)."""
    
    def __init__(self):
        super().__init__(
            name="native_tool",
            description="A native tool",
        )
        self.is_native = True
    
    async def initialize(self) -> None:
        pass
    
    async def execute(self, **kwargs) -> Any:
        """Should not be called."""
        raise NotImplementedError("Native tools should not be executed")
    
    def get_spec(self) -> Dict[str, Any]:
        return {
            "type": "web_search",
            "name": self.name,
            "description": self.description,
        }


class TestExecutorInitialization:
    """Test Executor initialization."""
    
    def test_executor_init_with_tools(self):
        """Test Executor initialization with tools."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        executor = Executor(llm, [calculator])
        
        assert executor.llm == llm
        assert "add" in executor.tools
        assert executor.tools["add"] == calculator
    
    def test_executor_init_with_multiple_tools(self):
        """Test Executor initialization with multiple tools."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        class SubtractTool(BaseTool):
            def __init__(self):
                super().__init__(name="subtract", description="Subtract")
            
            async def initialize(self) -> None:
                pass
            
            async def execute(self, a: int, b: int) -> int:
                return a - b
            
            def get_spec(self) -> Dict[str, Any]:
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
        
        subtract = SubtractTool()
        executor = Executor(llm, [calculator, subtract])
        
        assert len(executor.tools) == 2
        assert "add" in executor.tools
        assert "subtract" in executor.tools
    
    def test_executor_init_with_empty_tools(self):
        """Test Executor initialization with no tools."""
        llm = MockLLM()
        executor = Executor(llm, [])
        
        assert executor.llm == llm
        assert len(executor.tools) == 0


class TestExecutorExecute:
    """Test Executor.execute() method."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        executor = Executor(llm, [calculator])
        
        fn_call = {
            "name": "add",
            "arguments": json.dumps({"a": 5, "b": 3})
        }
        
        result = await executor.execute(fn_call)
        
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_execute_with_dict_arguments(self):
        """Test execute with dict arguments (not JSON string)."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        executor = Executor(llm, [calculator])
        
        fn_call = {
            "name": "add",
            "arguments": {"a": 10, "b": 20}  # Dict, not JSON string
        }
        
        result = await executor.execute(fn_call)
        
        assert result == 30
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test execute with non-existent tool."""
        llm = MockLLM()
        executor = Executor(llm, [])
        
        fn_call = {
            "name": "nonexistent",
            "arguments": json.dumps({})
        }
        
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await executor.execute(fn_call)
    
    @pytest.mark.asyncio
    async def test_execute_missing_name(self):
        """Test execute with missing tool name."""
        llm = MockLLM()
        executor = Executor(llm, [])
        
        fn_call = {
            "arguments": json.dumps({})
        }
        
        with pytest.raises(ValueError, match="Function call missing 'name' field"):
            await executor.execute(fn_call)
    
    @pytest.mark.asyncio
    async def test_execute_invalid_json(self):
        """Test execute with invalid JSON arguments."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        executor = Executor(llm, [calculator])
        
        fn_call = {
            "name": "add",
            "arguments": "{invalid json}"  # Invalid JSON
        }
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            await executor.execute(fn_call)
    
    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """Test execute when tool raises error."""
        llm = MockLLM()
        failing_tool = MockFailingTool()
        executor = Executor(llm, [failing_tool])
        
        fn_call = {
            "name": "failing_tool",
            "arguments": json.dumps({})
        }
        
        with pytest.raises(ValueError, match="Tool execution failed"):
            await executor.execute(fn_call)
    
    @pytest.mark.asyncio
    async def test_execute_native_tool_error(self):
        """Test execute with native tool (should raise error)."""
        llm = MockLLM()
        native_tool = MockNativeTool()
        executor = Executor(llm, [native_tool])
        
        fn_call = {
            "name": "native_tool",
            "arguments": json.dumps({})
        }
        
        with pytest.raises(ValueError, match="native tool"):
            await executor.execute(fn_call)


class TestExecutorExecuteStep:
    """Test Executor.execute_step() method."""
    
    @pytest.mark.asyncio
    async def test_execute_step_success(self):
        """Test successful plan step execution."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        executor = Executor(llm, [calculator])
        
        step = PlanStep(
            step=1,
            action="add",
            args={"a": 5, "b": 3}
        )
        
        context = {}  # Empty context - step args should be used
        
        result = await executor.execute_step(step, context)
        
        # Should execute with step args
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_execute_step_with_context(self):
        """Test execute_step merges context into args."""
        llm = MockLLM()
        
        class ContextTool(BaseTool):
            def __init__(self):
                super().__init__(name="context_tool", description="Uses context")
            
            async def initialize(self) -> None:
                pass
            
            async def execute(self, value: int, multiplier: int = 1) -> int:
                return value * multiplier
            
            def get_spec(self) -> Dict[str, Any]:
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "integer"},
                            "multiplier": {"type": "integer"},
                        },
                        "required": ["value"],
                    }
                }
        
        context_tool = ContextTool()
        executor = Executor(llm, [context_tool])
        
        step = PlanStep(
            step=1,
            action="context_tool",
            args={"value": 5}  # multiplier from context
        )
        
        context = {"multiplier": 3}
        
        result = await executor.execute_step(step, context)
        
        # Should use multiplier from context
        assert result == 15
    
    @pytest.mark.asyncio
    async def test_execute_step_no_args(self):
        """Test execute_step with no step args."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        executor = Executor(llm, [calculator])
        
        step = PlanStep(
            step=1,
            action="add",
            args=None  # No args
        )
        
        context = {"a": 5, "b": 3}
        
        result = await executor.execute_step(step, context)
        
        # Should use context args
        assert result == 8


class TestExecutorGetTool:
    """Test Executor.get_tool() method."""
    
    def test_get_tool_success(self):
        """Test get_tool returns tool if exists."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        executor = Executor(llm, [calculator])
        
        tool = executor.get_tool("add")
        
        assert tool == calculator
    
    def test_get_tool_not_found(self):
        """Test get_tool returns None if tool not found."""
        llm = MockLLM()
        executor = Executor(llm, [])
        
        tool = executor.get_tool("nonexistent")
        
        assert tool is None

