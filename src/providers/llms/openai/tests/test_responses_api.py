"""
Tests for Responses API integration in BaseOpenAI.

Covers the full pipeline for each native tool:
- Per-tool routing (call() selects the correct backend)
- Response normalization for each output type
- Conversation continuity (previous_response_id flow)
- SDK version fallback
- Mixed tools (native + custom function-calling)
- Error handling through the Responses API path

All API calls are mocked — no real OpenAI key is required.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


import pytest
from nucleusiq.tools import BaseTool
from nucleusiq_openai import BaseOpenAI, OpenAITool

# ======================================================================== #
# Helpers — mock OpenAI Responses API objects                              #
# ======================================================================== #


class _FakeContentBlock:
    """Simulate a Responses API content block."""

    def __init__(self, type: str, text: str = ""):
        self.type = type
        self.text = text


class _FakeOutputItem:
    """Simulate a Responses API output item."""

    def __init__(self, type: str, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self):
        d = {"type": self.type}
        for k, v in self.__dict__.items():
            if k != "type":
                d[k] = v
        return d


class _FakeResponse:
    """Simulate the full Responses API response object."""

    def __init__(self, id: str, output: list, status: str = "completed"):
        self.id = id
        self.output = output
        self.status = status


class _FakeResponses:
    """Simulate client.responses with a create() method."""

    def __init__(self, response: _FakeResponse):
        self._response = response

    async def create(self, **kwargs):
        return self._response


class _FakeSyncResponses:
    """Simulate sync client.responses with a create() method."""

    def __init__(self, response: _FakeResponse):
        self._response = response

    def create(self, **kwargs):
        return self._response


def _make_llm(**kwargs) -> BaseOpenAI:
    """Create BaseOpenAI instance with a test key."""
    defaults = {"model_name": "gpt-4o", "api_key": "test-key", "async_mode": True}
    defaults.update(kwargs)
    return BaseOpenAI(**defaults)


def _make_calculator_tool() -> BaseTool:
    """Create a simple custom function-calling tool."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return BaseTool.from_function(add, description="Add two integers")


# ======================================================================== #
# Per-tool routing tests                                                   #
# ======================================================================== #


class TestPerToolRouting:
    """Verify call() routes to Responses API for each native tool type."""

    def setup_method(self):
        self.llm = _make_llm()

    @pytest.mark.asyncio
    async def test_web_search_routes_to_responses_api(self):
        """web_search_preview triggers Responses API."""
        fake_resp = _FakeResponse(
            "resp_ws",
            [
                _FakeOutputItem("web_search_call", id="ws_1", status="completed"),
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[_FakeContentBlock("output_text", "Search result here.")],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        web_search = OpenAITool.web_search()
        tool_specs = self.llm.convert_tool_specs([web_search])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for NucleusIQ"}],
            tools=tool_specs,
        )

        msg = result.choices[0].message
        assert msg["content"] == "Search result here."
        assert "_native_outputs" in msg
        assert msg["_native_outputs"][0]["type"] == "web_search_call"

    @pytest.mark.asyncio
    async def test_code_interpreter_routes_to_responses_api(self):
        """code_interpreter triggers Responses API."""
        fake_resp = _FakeResponse(
            "resp_ci",
            [
                _FakeOutputItem("code_interpreter_call", id="ci_1", status="completed"),
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[_FakeContentBlock("output_text", "The result is 42.")],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        ci = OpenAITool.code_interpreter()
        tool_specs = self.llm.convert_tool_specs([ci])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Compute 6 * 7"}],
            tools=tool_specs,
        )

        msg = result.choices[0].message
        assert msg["content"] == "The result is 42."
        assert any(
            n["type"] == "code_interpreter_call" for n in msg.get("_native_outputs", [])
        )

    @pytest.mark.asyncio
    async def test_file_search_routes_to_responses_api(self):
        """file_search triggers Responses API."""
        fake_resp = _FakeResponse(
            "resp_fs",
            [
                _FakeOutputItem("file_search_call", id="fs_1", status="completed"),
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[
                        _FakeContentBlock(
                            "output_text", "Found in document: answer is X."
                        )
                    ],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        fs = OpenAITool.file_search(vector_store_ids=["vs_test"])
        tool_specs = self.llm.convert_tool_specs([fs])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What does the doc say?"}],
            tools=tool_specs,
        )

        msg = result.choices[0].message
        assert "Found in document" in msg["content"]

    @pytest.mark.asyncio
    async def test_image_generation_routes_to_responses_api(self):
        """image_generation triggers Responses API."""
        fake_resp = _FakeResponse(
            "resp_ig",
            [
                _FakeOutputItem("image_generation_call", id="ig_1", status="completed"),
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[
                        _FakeContentBlock("output_text", "Here is the generated image.")
                    ],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        ig = OpenAITool.image_generation()
        tool_specs = self.llm.convert_tool_specs([ig])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Generate a cat picture"}],
            tools=tool_specs,
        )

        msg = result.choices[0].message
        assert "generated image" in msg["content"]

    @pytest.mark.asyncio
    async def test_mcp_routes_to_responses_api(self):
        """MCP tool triggers Responses API."""
        fake_resp = _FakeResponse(
            "resp_mcp",
            [
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[_FakeContentBlock("output_text", "Rolled a 17.")],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        mcp = OpenAITool.mcp(
            server_label="dmcp",
            server_description="D&D dice server",
            server_url="https://dmcp-server.deno.dev/sse",
            require_approval="never",
        )
        tool_specs = self.llm.convert_tool_specs([mcp])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Roll a d20"}],
            tools=tool_specs,
        )

        assert "Rolled a 17" in result.choices[0].message["content"]

    @pytest.mark.asyncio
    async def test_computer_use_routes_to_responses_api(self):
        """computer_use_preview triggers Responses API."""
        fake_resp = _FakeResponse(
            "resp_cu",
            [
                _FakeOutputItem("computer_use_call", id="cu_1", status="completed"),
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[_FakeContentBlock("output_text", "Screenshot captured.")],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        cu = OpenAITool.computer_use()
        tool_specs = self.llm.convert_tool_specs([cu])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Take a screenshot"}],
            tools=tool_specs,
        )

        assert "Screenshot captured" in result.choices[0].message["content"]

    @pytest.mark.asyncio
    async def test_function_only_routes_to_chat_completions(self):
        """Custom function tools only → Chat Completions (not Responses API)."""
        # Mock chat.completions
        fake_msg = MagicMock()
        fake_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "15 + 27 = 42",
            "tool_calls": None,
        }
        fake_choice = MagicMock()
        fake_choice.message = fake_msg
        fake_chat_resp = MagicMock()
        fake_chat_resp.choices = [fake_choice]
        self.llm._client.chat = MagicMock()
        self.llm._client.chat.completions = MagicMock()
        self.llm._client.chat.completions.create = AsyncMock(
            return_value=fake_chat_resp
        )

        calc = _make_calculator_tool()
        tool_specs = self.llm.convert_tool_specs([calc])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Add 15 and 27"}],
            tools=tool_specs,
        )

        # Verify Chat Completions was called (not Responses)
        self.llm._client.chat.completions.create.assert_called_once()
        assert result.choices[0].message["content"] == "15 + 27 = 42"


# ======================================================================== #
# Mixed tools (native + custom)                                            #
# ======================================================================== #


class TestMixedToolRouting:
    """Verify that mixed native + custom tools route to Responses API."""

    def setup_method(self):
        self.llm = _make_llm()

    @pytest.mark.asyncio
    async def test_mixed_tools_route_to_responses_api(self):
        """Mix of function + native tools → Responses API."""
        fake_resp = _FakeResponse(
            "resp_mix",
            [
                _FakeOutputItem("web_search_call", id="ws_1", status="completed"),
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[
                        _FakeContentBlock("output_text", "Answer with search and calc.")
                    ],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        calc = _make_calculator_tool()
        web = OpenAITool.web_search()
        tool_specs = self.llm.convert_tool_specs([calc, web])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search and add"}],
            tools=tool_specs,
        )

        assert result.choices[0].message["content"] == "Answer with search and calc."

    @pytest.mark.asyncio
    async def test_mixed_tools_with_function_call_response(self):
        """Responses API may return function_call for custom tools."""
        fake_resp = _FakeResponse(
            "resp_mix_fn",
            [
                _FakeOutputItem("web_search_call", id="ws_1", status="completed"),
                _FakeOutputItem(
                    "function_call",
                    call_id="call_add_1",
                    name="add",
                    arguments='{"a": 10, "b": 20}',
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        calc = _make_calculator_tool()
        web = OpenAITool.web_search()
        tool_specs = self.llm.convert_tool_specs([calc, web])

        result = await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search then add 10+20"}],
            tools=tool_specs,
        )

        msg = result.choices[0].message
        # Function call should be normalized to tool_calls format
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_add_1"
        assert tc["function"]["name"] == "add"
        assert tc["function"]["arguments"] == '{"a": 10, "b": 20}'
        # Native output is also captured
        assert len(msg["_native_outputs"]) == 1


# ======================================================================== #
# Conversation continuity (previous_response_id)                           #
# ======================================================================== #


class TestConversationContinuity:
    """Test multi-turn tool execution with previous_response_id."""

    def setup_method(self):
        self.llm = _make_llm()

    @pytest.mark.asyncio
    async def test_function_call_sets_response_id(self):
        """When function_call is in response, _last_response_id is set."""
        fake_resp = _FakeResponse(
            "resp_turn1",
            [
                _FakeOutputItem(
                    "function_call",
                    call_id="call_1",
                    name="add",
                    arguments='{"a": 5, "b": 3}',
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)

        assert self.llm._last_response_id is None

        web = OpenAITool.web_search()
        calc = _make_calculator_tool()
        tool_specs = self.llm.convert_tool_specs([calc, web])

        await self.llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Add 5 and 3"}],
            tools=tool_specs,
        )

        # Response ID should be stored for continuation
        assert self.llm._last_response_id == "resp_turn1"

    @pytest.mark.asyncio
    async def test_final_answer_resets_response_id(self):
        """When no function calls, _last_response_id is reset."""
        fake_resp = _FakeResponse(
            "resp_final",
            [
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[_FakeContentBlock("output_text", "Done!")],
                ),
            ],
        )
        self.llm._client.responses = _FakeResponses(fake_resp)
        self.llm._last_response_id = "resp_previous"

        web = OpenAITool.web_search()
        tool_specs = self.llm.convert_tool_specs([web])

        # Simulate sending tool results back
        messages = [
            {"role": "user", "content": "Search X"},
            {"role": "tool", "tool_call_id": "call_1", "content": "8"},
        ]

        await self.llm.call(
            model="gpt-4o",
            messages=messages,
            tools=tool_specs,
        )

        # Response ID should be reset (conversation complete)
        assert self.llm._last_response_id is None

    @pytest.mark.asyncio
    async def test_continuation_sends_only_tool_results(self):
        """When _last_response_id is set, only tool results are sent as input."""
        captured_payload = {}

        class _CapturingResponses:
            async def create(self, **kwargs):
                captured_payload.update(kwargs)
                return _FakeResponse(
                    "resp_cont",
                    [
                        _FakeOutputItem(
                            "message",
                            role="assistant",
                            content=[_FakeContentBlock("output_text", "Got it.")],
                        ),
                    ],
                )

        self.llm._client.responses = _CapturingResponses()
        self.llm._last_response_id = "resp_turn1"

        web = OpenAITool.web_search()
        tool_specs = self.llm.convert_tool_specs([web])

        messages = [
            {"role": "system", "content": "You are a helper"},
            {"role": "user", "content": "Original question"},
            {"role": "assistant", "content": None, "tool_calls": [...]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "42"},
        ]

        await self.llm.call(
            model="gpt-4o",
            messages=messages,
            tools=tool_specs,
        )

        # Should have used previous_response_id
        assert captured_payload["previous_response_id"] == "resp_turn1"
        # Input should only contain function_call_output (not user/system/assistant)
        input_items = captured_payload["input"]
        assert len(input_items) == 1
        assert input_items[0]["type"] == "function_call_output"
        assert input_items[0]["call_id"] == "call_abc"
        assert input_items[0]["output"] == "42"


# ======================================================================== #
# SDK version fallback                                                     #
# ======================================================================== #


class TestSDKFallback:
    """Test graceful fallback when SDK lacks Responses API."""

    @pytest.mark.asyncio
    async def test_fallback_to_chat_completions_when_no_responses(self):
        """If client has no .responses attr, falls back to Chat Completions."""
        llm = _make_llm()

        # Replace client with a mock that has chat but NOT responses
        fake_msg = MagicMock()
        fake_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "Fallback answer",
            "tool_calls": None,
        }
        fake_choice = MagicMock()
        fake_choice.message = fake_msg
        fake_chat_resp = MagicMock()
        fake_chat_resp.choices = [fake_choice]

        mock_client = MagicMock(spec=["chat"])  # Only 'chat', no 'responses'
        mock_client.chat.completions.create = AsyncMock(return_value=fake_chat_resp)
        llm._client = mock_client

        web = OpenAITool.web_search()
        tool_specs = llm.convert_tool_specs([web])

        result = await llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search X"}],
            tools=tool_specs,
        )

        # Should have fallen back to Chat Completions
        mock_client.chat.completions.create.assert_called_once()
        assert result.choices[0].message["content"] == "Fallback answer"


# ======================================================================== #
# Structured output through Responses API                                  #
# ======================================================================== #


class TestStructuredOutputViaResponsesAPI:
    """Test that structured output (response_format) works through Responses API path."""

    @pytest.mark.asyncio
    async def test_json_schema_structured_output(self):
        """Pydantic model as response_format routes through Responses API correctly."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        fake_resp = _FakeResponse(
            "resp_so",
            [
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[
                        _FakeContentBlock("output_text", '{"name": "Alice", "age": 30}')
                    ],
                ),
            ],
        )

        captured = {}

        class _CapturingResponses:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return fake_resp

        llm = _make_llm()
        llm._client.responses = _CapturingResponses()

        web = OpenAITool.web_search()
        tool_specs = llm.convert_tool_specs([web])

        result = await llm.call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Extract: Alice is 30"}],
            tools=tool_specs,
            response_format=Person,
        )

        # Result should be a validated Pydantic instance
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30
        # Verify text config was passed to Responses API
        assert "text" in captured
        assert captured["text"]["format"]["type"] == "json_schema"
        assert captured["text"]["format"]["name"] == "Person"


# ======================================================================== #
# Normalization edge cases                                                 #
# ======================================================================== #


class TestNormalizationEdgeCases:
    """Test normalization for various response shapes."""

    def setup_method(self):
        self.llm = _make_llm()

    def test_multiple_text_blocks_joined(self):
        """Multiple output_text blocks are joined with double newline."""
        resp = _FakeResponse(
            "resp_multi",
            [
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[
                        _FakeContentBlock("output_text", "Paragraph 1."),
                        _FakeContentBlock("output_text", "Paragraph 2."),
                    ],
                ),
            ],
        )

        result = self.llm._normalize_responses_output(resp)
        assert result.choices[0].message["content"] == "Paragraph 1.\n\nParagraph 2."

    def test_multiple_function_calls(self):
        """Multiple function_call items all get normalized."""
        resp = _FakeResponse(
            "resp_multi_fn",
            [
                _FakeOutputItem(
                    "function_call", call_id="c1", name="add", arguments='{"a":1,"b":2}'
                ),
                _FakeOutputItem(
                    "function_call",
                    call_id="c2",
                    name="multiply",
                    arguments='{"a":3,"b":4}',
                ),
            ],
        )

        result = self.llm._normalize_responses_output(resp)
        msg = result.choices[0].message
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "add"
        assert msg["tool_calls"][1]["function"]["name"] == "multiply"

    def test_only_native_outputs_no_text(self):
        """Response with only native outputs (no message) → content is None."""
        resp = _FakeResponse(
            "resp_native_only",
            [
                _FakeOutputItem("web_search_call", id="ws_1", status="completed"),
            ],
        )

        result = self.llm._normalize_responses_output(resp)
        msg = result.choices[0].message
        assert msg["content"] is None
        assert "tool_calls" not in msg
        assert len(msg["_native_outputs"]) == 1

    def test_none_output_list(self):
        """Response with None output → empty choices message."""
        resp = _FakeResponse("resp_none", [])
        resp.output = None  # Simulate None

        result = self.llm._normalize_responses_output(resp)
        assert result.choices[0].message["content"] is None

    def test_content_block_without_text(self):
        """Content block with non-text type is ignored."""
        resp = _FakeResponse(
            "resp_img",
            [
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[
                        _FakeContentBlock("output_image", ""),  # Not text
                        _FakeContentBlock("output_text", "Caption here."),
                    ],
                ),
            ],
        )

        result = self.llm._normalize_responses_output(resp)
        assert result.choices[0].message["content"] == "Caption here."


# ======================================================================== #
# responses_call() direct access                                           #
# ======================================================================== #


class TestResponsesCallDirect:
    """Test the public responses_call() method for advanced users."""

    @pytest.mark.asyncio
    async def test_responses_call_returns_raw_response(self):
        """responses_call() returns the raw Responses API response."""
        llm = _make_llm()
        fake_resp = _FakeResponse(
            "resp_raw",
            [
                _FakeOutputItem(
                    "message",
                    role="assistant",
                    content=[_FakeContentBlock("output_text", "Raw response")],
                ),
            ],
        )
        llm._client.responses = _FakeResponses(fake_resp)

        result = await llm.responses_call(
            model="gpt-4o",
            input="Hello",
            tools=[{"type": "web_search_preview"}],
        )

        # Should be the raw response object, NOT _LLMResponse
        assert result.id == "resp_raw"
        assert result.output[0].type == "message"

    @pytest.mark.asyncio
    async def test_responses_call_with_previous_response_id(self):
        """responses_call() passes previous_response_id."""
        captured = {}

        class _CapturingResponses:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return _FakeResponse(
                    "resp_2",
                    [
                        _FakeOutputItem(
                            "message",
                            role="assistant",
                            content=[_FakeContentBlock("output_text", "Continued.")],
                        ),
                    ],
                )

        llm = _make_llm()
        llm._client.responses = _CapturingResponses()

        await llm.responses_call(
            model="gpt-4o",
            input=[{"type": "function_call_output", "call_id": "c1", "output": "42"}],
            previous_response_id="resp_1",
        )

        assert captured["previous_response_id"] == "resp_1"

    @pytest.mark.asyncio
    async def test_responses_call_with_include(self):
        """responses_call() passes include parameter."""
        captured = {}

        class _CapturingResponses:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return _FakeResponse(
                    "resp_inc",
                    [
                        _FakeOutputItem(
                            "message",
                            role="assistant",
                            content=[_FakeContentBlock("output_text", "With sources.")],
                        ),
                    ],
                )

        llm = _make_llm()
        llm._client.responses = _CapturingResponses()

        await llm.responses_call(
            model="gpt-4o",
            input="Search for X",
            tools=[{"type": "web_search_preview"}],
            include=["output[*].web_search_call.results"],
        )

        assert captured["include"] == ["output[*].web_search_call.results"]

    @pytest.mark.asyncio
    async def test_responses_call_raises_if_no_sdk_support(self):
        """responses_call() raises AttributeError if SDK is too old."""
        llm = _make_llm()
        # Replace client with a mock that lacks .responses
        llm._client = MagicMock(spec=["chat"])

        with pytest.raises(AttributeError, match="Responses API requires openai>=1.66"):
            await llm.responses_call(model="gpt-4o", input="Hi")


# ======================================================================== #
# Message conversion edge cases                                            #
# ======================================================================== #


class TestMessageConversionEdgeCases:
    """Test _messages_to_responses_input with edge cases."""

    def setup_method(self):
        self.llm = _make_llm()
        self.llm._last_response_id = None

    def test_empty_messages(self):
        """Empty messages → empty input."""
        instructions, items = self.llm._messages_to_responses_input([])
        assert instructions is None
        assert items == []

    def test_none_content_handled(self):
        """Messages with None content don't crash."""
        messages = [
            {"role": "system", "content": None},
            {"role": "user", "content": None},
        ]
        instructions, items = self.llm._messages_to_responses_input(messages)
        assert instructions == ""
        assert items[0] == {"role": "user", "content": ""}

    def test_mixed_messages_and_tool_results(self):
        """Full conversation history with system + user + tool results."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Do the thing."},
            {"role": "assistant", "content": "I'll use a tool."},
            {"role": "tool", "tool_call_id": "tc_1", "content": "done"},
            {"role": "user", "content": "Thanks!"},
        ]
        instructions, items = self.llm._messages_to_responses_input(messages)

        assert instructions == "Be helpful."
        assert len(items) == 4  # user + assistant + tool_result + user
        assert items[0]["role"] == "user"
        assert items[1]["role"] == "assistant"
        assert items[2]["type"] == "function_call_output"
        assert items[3]["role"] == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
