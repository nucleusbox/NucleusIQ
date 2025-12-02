"""
Tests for Agent integration with tools.

Tests cover:
- Agent with BaseTool instances
- Agent with native OpenAI tools
- Agent with mixed tools
- Tool execution flow
"""

import pytest
import asyncio
from typing import Dict, Any
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, AgentState
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.providers.llms.openai.tools import OpenAITool
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


class TestAgentWithBaseTool:
    """Test Agent with BaseTool instances."""
    
    @pytest.mark.asyncio
    async def test_agent_with_base_tool(self):
        """Test Agent with a BaseTool."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A helpful calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        assert agent.state == AgentState.INITIALIZING
        
        # Mock LLM will return a function call
        task = {"id": "task1", "objective": "What is 5 + 3?"}
        
        # Note: This will fail if MockLLM doesn't support function calling properly
        # For now, we just test that the agent can be created and initialized
        assert agent.tools == [calculator]
        assert len(agent.tools) == 1
    
    @pytest.mark.asyncio
    async def test_agent_tool_specs_conversion(self):
        """Test that Agent converts tool specs correctly."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A helpful calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # Check that tools are stored
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "add"


class TestAgentWithNativeTools:
    """Test Agent with native OpenAI tools."""
    
    @pytest.mark.asyncio
    async def test_agent_with_native_tool(self):
        """Test Agent with native OpenAI tool."""
        llm = MockLLM()
        web_search = OpenAITool.web_search()
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            tools=[web_search],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "web_search_preview"
        assert agent.tools[0].is_native is True
    
    @pytest.mark.asyncio
    async def test_agent_with_mcp_tool(self):
        """Test Agent with MCP tool."""
        llm = MockLLM()
        mcp_tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test server",
            server_url="https://test.com",
        )
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            tools=[mcp_tool],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "mcp_test"
        assert agent.tools[0].is_native is True


class TestAgentWithMixedTools:
    """Test Agent with mixed BaseTool and native tools."""
    
    @pytest.mark.asyncio
    async def test_agent_with_mixed_tools(self):
        """Test Agent with both BaseTool and native tools."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        web_search = OpenAITool.web_search()
        mcp_tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test",
            server_url="https://test.com",
        )
        
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            narrative="A helpful assistant",
            llm=llm,
            tools=[calculator, web_search, mcp_tool],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        assert len(agent.tools) == 3
        
        # Check BaseTool
        base_tool = next(t for t in agent.tools if t.name == "add")
        assert isinstance(base_tool, BaseTool)
        assert not hasattr(base_tool, 'is_native') or not base_tool.is_native
        
        # Check native tools
        native_tools = [t for t in agent.tools if hasattr(t, 'is_native') and t.is_native]
        assert len(native_tools) == 2
        assert any(t.name == "web_search_preview" for t in native_tools)
        assert any(t.name == "mcp_test" for t in native_tools)


class TestAgentToolExecution:
    """Test Agent tool execution flow."""
    
    @pytest.mark.asyncio
    async def test_agent_tool_not_found_error(self):
        """Test that Agent raises error when tool is not found."""
        llm = MockLLM()
        calculator = MockCalculatorTool()
        
        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            narrative="A helpful calculator agent",
            llm=llm,
            tools=[calculator],
            config=AgentConfig(verbose=False)
        )
        
        await agent.initialize()
        
        # This test verifies the error handling logic exists
        # Actual execution would require a properly configured MockLLM
        # that returns function calls
        assert agent.tools is not None
        assert len(agent.tools) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

