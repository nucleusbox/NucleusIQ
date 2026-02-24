"""
Tests for tool spec conversion and API routing in BaseOpenAI.

Tests cover:
- BaseTool to OpenAI function calling format conversion
- Native OpenAI tools (pass-through)
- Mixed tool lists
- _has_native_tools() routing detection
- _messages_to_responses_input() format conversion
- _normalize_responses_output() response normalization
- _build_responses_text_config() structured output conversion
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from typing import Any, Dict

import pytest
from nucleusiq.tools import BaseTool
from nucleusiq_openai import BaseOpenAI, OpenAITool


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


# ======================================================================== #
# Tool Spec Conversion (existing tests, updated for new type names)        #
# ======================================================================== #


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
            },
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
                    "additionalProperties": False,
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

        calculator = MockBaseTool(
            name="calculate",
            description="Calculate",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        )

        web_search = OpenAITool.web_search()
        mcp_tool = OpenAITool.mcp(
            server_label="test",
            server_description="Test",
            server_url="https://test.com",
        )

        converted = llm.convert_tool_specs([calculator, web_search, mcp_tool])

        assert len(converted) == 3
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "calculate"
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
                "properties": {"x": {"type": "string"}},
            },
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
                "properties": {"x": {"type": "string"}},
                "additionalProperties": True,
            },
        )

        converted = llm.convert_tool_specs([tool])
        params = converted[0]["function"]["parameters"]
        assert params["additionalProperties"] is True


# ======================================================================== #
# API Routing Detection                                                    #
# ======================================================================== #


class TestHasNativeTools:
    """Test _has_native_tools() routing logic."""

    def setup_method(self):
        self.llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")

    def test_empty_tools(self):
        """No tools → False."""
        assert self.llm._has_native_tools(None) is False
        assert self.llm._has_native_tools([]) is False

    def test_function_only_tools(self):
        """Only function-calling tools → False."""
        tools = [
            {"type": "function", "function": {"name": "add", "parameters": {}}},
            {"type": "function", "function": {"name": "multiply", "parameters": {}}},
        ]
        assert self.llm._has_native_tools(tools) is False

    def test_native_only_tools(self):
        """Only native tools → True."""
        tools = [{"type": "web_search_preview"}, {"type": "code_interpreter"}]
        assert self.llm._has_native_tools(tools) is True

    def test_mixed_tools(self):
        """Mix of function + native tools → True."""
        tools = [
            {"type": "function", "function": {"name": "add", "parameters": {}}},
            {"type": "web_search_preview"},
        ]
        assert self.llm._has_native_tools(tools) is True

    def test_all_native_types(self):
        """Every registered native type is detected."""
        for tool_type in [
            "web_search_preview",
            "web_search_preview_2025_03_11",
            "code_interpreter",
            "file_search",
            "image_generation",
            "mcp",
            "computer_use_preview",
            "computer_use_preview_2025_03_11",
        ]:
            assert self.llm._has_native_tools([{"type": tool_type}]) is True

    def test_unknown_type_not_native(self):
        """Unknown tool type is not treated as native."""
        tools = [{"type": "custom_experimental_tool"}]
        assert self.llm._has_native_tools(tools) is False


# ======================================================================== #
# Message Conversion (Chat → Responses API format)                         #
# ======================================================================== #


class TestMessagesToResponsesInput:
    """Test _messages_to_responses_input() conversion."""

    def setup_method(self):
        self.llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
        self.llm._last_response_id = None  # Ensure clean state

    def test_simple_user_message(self):
        """User message converts to input item."""
        messages = [{"role": "user", "content": "Hello"}]
        instructions, items = self.llm._messages_to_responses_input(messages)

        assert instructions is None
        assert items == [{"role": "user", "content": "Hello"}]

    def test_system_message_becomes_instructions(self):
        """System message is extracted as instructions."""
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
        ]
        instructions, items = self.llm._messages_to_responses_input(messages)

        assert instructions == "You are a helper."
        assert len(items) == 1
        assert items[0] == {"role": "user", "content": "Hi"}

    def test_assistant_message_preserved(self):
        """Assistant messages pass through."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        instructions, items = self.llm._messages_to_responses_input(messages)

        assert instructions is None
        assert len(items) == 3
        assert items[1] == {"role": "assistant", "content": "Hello!"}

    def test_tool_results_on_first_call(self):
        """Tool results on first call convert to function_call_output."""
        messages = [
            {"role": "user", "content": "Add 2+3"},
            {"role": "tool", "tool_call_id": "call_abc", "content": "5"},
        ]
        instructions, items = self.llm._messages_to_responses_input(messages)

        assert len(items) == 2
        assert items[0] == {"role": "user", "content": "Add 2+3"}
        assert items[1] == {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "5",
        }

    def test_continuation_only_sends_tool_results(self):
        """When _last_response_id is set, only tool results are sent."""
        self.llm._last_response_id = "resp_previous_123"

        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": None, "tool_calls": [...]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result_1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result_2"},
        ]
        instructions, items = self.llm._messages_to_responses_input(messages)

        # Instructions and user/assistant messages are NOT sent (server-side state)
        assert instructions is None
        assert len(items) == 2
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_1"
        assert items[1]["call_id"] == "call_2"

    def test_multiple_system_messages_joined(self):
        """Multiple system messages are joined with newline."""
        messages = [
            {"role": "system", "content": "Line 1"},
            {"role": "system", "content": "Line 2"},
            {"role": "user", "content": "Go"},
        ]
        instructions, items = self.llm._messages_to_responses_input(messages)

        assert instructions == "Line 1\nLine 2"
        assert len(items) == 1


# ======================================================================== #
# Response Normalization (Responses API → _LLMResponse)                    #
# ======================================================================== #


class _MockContentBlock:
    """Simulate Responses API content block."""

    def __init__(self, type: str, text: str = ""):
        self.type = type
        self.text = text


class _MockOutputItem:
    """Simulate Responses API output item."""

    def __init__(self, type: str, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self):
        return {"type": self.type}


class _MockResponse:
    """Simulate Responses API response."""

    def __init__(self, id: str, output: list):
        self.id = id
        self.output = output


class TestNormalizeResponsesOutput:
    """Test _normalize_responses_output() normalization."""

    def setup_method(self):
        self.llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")

    def test_simple_text_response(self):
        """Message with text content normalizes correctly."""
        resp = _MockResponse(
            id="resp_1",
            output=[
                _MockOutputItem(
                    "message",
                    role="assistant",
                    content=[_MockContentBlock("output_text", "Hello world")],
                )
            ],
        )

        result = self.llm._normalize_responses_output(resp)

        assert len(result.choices) == 1
        msg = result.choices[0].message
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert "tool_calls" not in msg

    def test_function_call_in_response(self):
        """Function call items normalize to tool_calls format."""
        resp = _MockResponse(
            id="resp_2",
            output=[
                _MockOutputItem(
                    "function_call",
                    call_id="call_abc",
                    name="add",
                    arguments='{"a": 2, "b": 3}',
                ),
            ],
        )

        result = self.llm._normalize_responses_output(resp)

        msg = result.choices[0].message
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "add"
        assert tc["function"]["arguments"] == '{"a": 2, "b": 3}'

    def test_mixed_native_and_function_calls(self):
        """Native tool outputs + function calls + text all normalize."""
        resp = _MockResponse(
            id="resp_3",
            output=[
                _MockOutputItem("web_search_call", id="ws_1", status="completed"),
                _MockOutputItem(
                    "function_call",
                    call_id="call_xyz",
                    name="calculate",
                    arguments='{"expr": "2+3"}',
                ),
                _MockOutputItem(
                    "message",
                    role="assistant",
                    content=[
                        _MockContentBlock("output_text", "Based on the search,"),
                        _MockContentBlock("output_text", "the answer is 5."),
                    ],
                ),
            ],
        )

        result = self.llm._normalize_responses_output(resp)

        msg = result.choices[0].message
        assert msg["role"] == "assistant"
        assert "Based on the search," in msg["content"]
        assert "the answer is 5." in msg["content"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "calculate"
        # Native outputs stored as metadata
        assert "_native_outputs" in msg
        assert len(msg["_native_outputs"]) == 1

    def test_empty_response(self):
        """Empty output list produces None content."""
        resp = _MockResponse(id="resp_4", output=[])
        result = self.llm._normalize_responses_output(resp)

        msg = result.choices[0].message
        assert msg["content"] is None
        assert "tool_calls" not in msg


# ======================================================================== #
# Responses API text config conversion                                     #
# ======================================================================== #


class TestBuildResponsesTextConfig:
    """Test _build_responses_text_config() conversion."""

    def setup_method(self):
        self.llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")

    def test_json_schema_format(self):
        """json_schema response_format converts to text.format."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }

        result = self.llm._build_responses_text_config(response_format)

        assert result == {
            "format": {
                "type": "json_schema",
                "name": "Person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            }
        }

    def test_json_object_format(self):
        """json_object mode converts to text.format."""
        response_format = {"type": "json_object"}
        result = self.llm._build_responses_text_config(response_format)
        assert result == {"format": {"type": "json_object"}}

    def test_unknown_format_returns_none(self):
        """Unknown format type returns None."""
        result = self.llm._build_responses_text_config({"type": "text"})
        assert result is None

    def test_non_dict_returns_none(self):
        """Non-dict input returns None."""
        result = self.llm._build_responses_text_config("not a dict")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
