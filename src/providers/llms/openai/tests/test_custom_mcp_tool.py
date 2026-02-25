"""
Tests for custom MCP tool (BaseTool implementation).

Tests cover:
- Custom MCP tool as BaseTool
- Tool spec generation
- Tool execution
- Integration with OpenAI LLM
"""
# ruff: noqa: E402

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from typing import Any

import pytest
from nucleusiq.tools import BaseTool

from nucleusiq_openai import BaseOpenAI


class CustomMCPTool(BaseTool):
    """Custom MCP client tool for testing."""

    def __init__(
        self,
        server_url: str,
        server_label: str = "custom_mcp",
        authorization: str | None = None,
    ):
        super().__init__(
            name=f"mcp_client_{server_label}",
            description=f"Custom MCP client for {server_url}",
        )
        self.server_url = server_url
        self.server_label = server_label
        self.authorization = authorization
        self._available_tools: list[dict[str, Any]] | None = None

    async def initialize(self) -> None:
        """Initialize by fetching available tools."""
        self._available_tools = [
            {
                "name": "roll",
                "description": "Roll dice",
                "input_schema": {
                    "type": "object",
                    "properties": {"diceRollExpression": {"type": "string"}},
                    "required": ["diceRollExpression"],
                },
            }
        ]

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute an MCP tool."""
        if tool_name == "roll":
            dice_expr = kwargs.get("diceRollExpression", "")
            return f"Dice roll result for '{dice_expr}': 7"
        raise ValueError(f"Unknown tool: {tool_name}")

    def get_spec(self) -> dict[str, Any]:
        """Return tool spec for function calling."""
        if not self._available_tools:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["tool_name", "arguments"],
                },
            }

        tool_descriptions = [
            f"- {t['name']}: {t['description']}" for t in self._available_tools
        ]

        return {
            "name": self.name,
            "description": f"{self.description}\n\nAvailable tools:\n"
            + "\n".join(tool_descriptions),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the MCP tool to call",
                        "enum": [t["name"] for t in self._available_tools],
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments for the tool",
                        "additionalProperties": True,
                    },
                },
                "required": ["tool_name", "arguments"],
            },
        }


class TestCustomMCPTool:
    """Test custom MCP tool implementation."""

    def test_custom_mcp_tool_creation(self):
        """Test creating custom MCP tool."""
        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
        )

        assert isinstance(tool, BaseTool)
        assert tool.name == "mcp_client_test"
        assert tool.server_url == "https://test.com"
        assert tool.server_label == "test"
        # Check if is_native attribute exists and is False (custom tools are not native)
        is_native = getattr(tool, "is_native", False)
        assert not is_native

    def test_custom_mcp_tool_spec_before_init(self):
        """Test tool spec before initialization."""
        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
        )

        spec = tool.get_spec()
        assert spec["name"] == "mcp_client_test"
        assert "tool_name" in spec["parameters"]["properties"]
        assert "arguments" in spec["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_custom_mcp_tool_initialization(self):
        """Test tool initialization."""
        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
        )

        await tool.initialize()

        assert tool._available_tools is not None
        assert len(tool._available_tools) == 1
        assert tool._available_tools[0]["name"] == "roll"

    @pytest.mark.asyncio
    async def test_custom_mcp_tool_spec_after_init(self):
        """Test tool spec after initialization."""
        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
        )

        await tool.initialize()
        spec = tool.get_spec()

        assert spec["name"] == "mcp_client_test"
        assert "Available tools:" in spec["description"]
        assert "roll" in spec["description"]
        assert spec["parameters"]["properties"]["tool_name"]["enum"] == ["roll"]

    @pytest.mark.asyncio
    async def test_custom_mcp_tool_execute(self):
        """Test tool execution."""
        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
        )

        await tool.initialize()

        result = await tool.execute(tool_name="roll", diceRollExpression="2d4+1")

        assert "Dice roll result" in result
        assert "2d4+1" in result

    @pytest.mark.asyncio
    async def test_custom_mcp_tool_execute_unknown_tool(self):
        """Test tool execution with unknown tool."""
        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
        )

        await tool.initialize()

        with pytest.raises(ValueError, match="Unknown tool"):
            await tool.execute(tool_name="unknown", x=1)

    def test_custom_mcp_tool_conversion_to_openai_format(self):
        """Test converting custom MCP tool to OpenAI function calling format."""
        llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")

        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
        )

        converted = llm.convert_tool_specs([tool])

        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "mcp_client_test"
        assert "parameters" in converted[0]["function"]

    @pytest.mark.asyncio
    async def test_custom_mcp_tool_with_authorization(self):
        """Test custom MCP tool with authorization."""
        tool = CustomMCPTool(
            server_url="https://test.com",
            server_label="test",
            authorization="token123",
        )

        assert tool.authorization == "token123"
        await tool.initialize()

        # Tool should work the same way
        result = await tool.execute(tool_name="roll", diceRollExpression="1d6")
        assert "Dice roll result" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
