"""
Tests for tool spec conversion in BaseOpenAI.

Tests cover:
- BaseTool to OpenAI function calling format conversion
- Native OpenAI tools (pass-through)
- Mixed tool lists
"""

import pytest
from typing import Dict, Any
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
from nucleusiq.providers.llms.openai.tools import OpenAITool
from nucleusiq.core.tools import BaseTool


class MockBaseTool(BaseTool):
    """Mock BaseTool for testing."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        super().__init__(name=name, description=description)
        self._parameters = parameters
    
    async def initialize(self) -> None:
        pass
    
    async def execute(self, **kwargs: Any) -> Any:
        return "result"
    
    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._parameters,
        }


class TestToolConversion:
    """Test tool spec conversion in BaseOpenAI."""
    
    def test_convert_base_tool_to_function_calling(self):
        """Test converting BaseTool spec to OpenAI function calling format."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        
        tool = MockBaseTool(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            }
        )
        
        converted = llm.convert_tool_specs([tool])
        
        assert len(converted) == 1
        assert converted[0] == {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,  # Added by conversion
                },
            },
        }
    
    def test_convert_native_tool_passthrough(self):
        """Test that native OpenAI tools pass through unchanged."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        
        web_search = OpenAITool.web_search()
        code_interpreter = OpenAITool.code_interpreter()
        
        converted = llm.convert_tool_specs([web_search, code_interpreter])
        
        assert len(converted) == 2
        assert converted[0] == {"type": "web_search_preview"}
        assert converted[1] == {"type": "code_interpreter"}
    
    def test_convert_mcp_tool_passthrough(self):
        """Test that MCP tools pass through unchanged."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        
        mcp_tool = OpenAITool.mcp(
            server_label="dmcp",
            server_description="D&D server",
            server_url="https://dmcp-server.deno.dev/sse",
            require_approval="never",
        )
        
        converted = llm.convert_tool_specs([mcp_tool])
        
        assert len(converted) == 1
        spec = converted[0]
        assert spec["type"] == "mcp"
        assert spec["server_label"] == "dmcp"
        assert spec["server_url"] == "https://dmcp-server.deno.dev/sse"
        assert spec["require_approval"] == "never"
    
    def test_convert_mixed_tools(self):
        """Test converting mixed BaseTool and native tools."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        
        # BaseTool
        calculator = MockBaseTool(
            name="calculate",
            description="Calculate",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            }
        )
        
        # Native tools
        web_search = OpenAITool.web_search()
        mcp_tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test",
            server_url="https://test.com",
        )
        
        converted = llm.convert_tool_specs([calculator, web_search, mcp_tool])
        
        assert len(converted) == 3
        
        # BaseTool converted to function calling
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "calculate"
        
        # Native tools pass through
        assert converted[1] == {"type": "web_search_preview"}
        assert converted[2]["type"] == "mcp"
    
    def test_convert_empty_list(self):
        """Test converting empty tool list."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        
        converted = llm.convert_tool_specs([])
        
        assert converted == []
    
    def test_additional_properties_added(self):
        """Test that additionalProperties: False is added to BaseTool parameters."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        
        tool = MockBaseTool(
            name="test",
            description="Test",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                },
            }
        )
        
        converted = llm.convert_tool_specs([tool])
        
        params = converted[0]["function"]["parameters"]
        assert params["additionalProperties"] is False
    
    def test_additional_properties_preserved(self):
        """Test that existing additionalProperties is preserved."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        
        tool = MockBaseTool(
            name="test",
            description="Test",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                },
                "additionalProperties": True,  # Already present
            }
        )
        
        converted = llm.convert_tool_specs([tool])
        
        params = converted[0]["function"]["parameters"]
        assert params["additionalProperties"] is True  # Preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

