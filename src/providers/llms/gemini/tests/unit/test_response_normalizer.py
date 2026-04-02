"""Tests for nucleusiq_gemini.nb_gemini.response_normalizer."""

from nucleusiq_gemini.nb_gemini.response_normalizer import (
    messages_to_gemini_contents,
    normalize_response,
)

from tests.unit._mock_factories import (
    make_candidate,
    make_function_call_part,
    make_response,
    make_text_part,
)


class TestNormalizeResponse:
    def test_simple_text(self, simple_response):
        result = normalize_response(simple_response)
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].message.tool_calls is None

    def test_tool_call(self, tool_call_response):
        result = normalize_response(tool_call_response)
        msg = result.choices[0].message
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.id == "call_123"
        assert '"location"' in tc.function.arguments

    def test_multiple_tool_calls(self, multi_tool_response):
        result = normalize_response(multi_tool_response)
        msg = result.choices[0].message
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].function.name == "get_weather"
        assert msg.tool_calls[1].function.name == "get_time"

    def test_thinking_response(self, thinking_response):
        result = normalize_response(thinking_response)
        msg = result.choices[0].message
        assert msg.content == "The answer is 42."
        assert msg.native_outputs is not None
        assert any(o["type"] == "thinking" for o in msg.native_outputs)

    def test_code_execution(self, code_exec_response):
        result = normalize_response(code_exec_response)
        msg = result.choices[0].message
        assert msg.content == "The result is 4."
        assert msg.native_outputs is not None
        types = [o["type"] for o in msg.native_outputs]
        assert "code_execution" in types
        assert "code_execution_result" in types

    def test_empty_response(self, empty_response):
        result = normalize_response(empty_response)
        assert len(result.choices) == 1
        assert result.choices[0].message.content == ""

    def test_usage_extraction(self, simple_response):
        result = normalize_response(simple_response)
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    def test_model_version(self, simple_response):
        result = normalize_response(simple_response)
        assert result.model == "gemini-2.5-flash"

    def test_no_usage(self):
        resp = make_response(usage=None)
        resp.usage_metadata = None
        result = normalize_response(resp)
        assert result.usage is None

    def test_function_call_without_id(self):
        resp = make_response(
            candidates=[
                make_candidate([make_function_call_part("fn", {"x": 1}, call_id=None)])
            ]
        )
        result = normalize_response(resp)
        tc = result.choices[0].message.tool_calls[0]
        assert tc.id is not None  # should generate a UUID

    def test_multiple_text_parts_concatenated(self):
        resp = make_response(
            candidates=[
                make_candidate(
                    [
                        make_text_part("Hello "),
                        make_text_part("world!"),
                    ]
                )
            ]
        )
        result = normalize_response(resp)
        assert result.choices[0].message.content == "Hello world!"


class TestMessagesToGeminiContents:
    def test_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        system, contents = messages_to_gemini_contents(messages)
        assert system is None
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "Hello"}]

    def test_system_message_extracted(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        system, contents = messages_to_gemini_contents(messages)
        assert system == "Be helpful"
        assert len(contents) == 1

    def test_assistant_to_model(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        _, contents = messages_to_gemini_contents(messages)
        assert contents[1]["role"] == "model"

    def test_tool_result_message(self):
        messages = [
            {
                "role": "tool",
                "content": '{"temp": 72}',
                "tool_call_id": "call_123",
                "name": "get_weather",
            }
        ]
        _, contents = messages_to_gemini_contents(messages)
        assert len(contents) == 1
        part = contents[0]["parts"][0]
        assert "function_response" in part
        assert part["function_response"]["name"] == "get_weather"

    def test_tool_result_with_plain_text(self):
        messages = [
            {
                "role": "tool",
                "content": "The weather is sunny",
                "tool_call_id": "c1",
                "name": "get_weather",
            }
        ]
        _, contents = messages_to_gemini_contents(messages)
        part = contents[0]["parts"][0]
        assert part["function_response"]["response"] == {
            "result": "The weather is sunny"
        }

    def test_tool_result_json_string_payload_is_wrapped_for_gemini_sdk(self):
        """json.dumps(str) parses back to str; google-genai requires response: dict."""
        import json

        messages = [
            {
                "role": "tool",
                "content": json.dumps("line1\nline2"),
                "tool_call_id": "c1",
                "name": "list_files",
            }
        ]
        _, contents = messages_to_gemini_contents(messages)
        part = contents[0]["parts"][0]
        assert part["function_response"]["response"] == {"result": "line1\nline2"}

    def test_tool_result_infers_name_from_prior_assistant_when_name_omitted(self):
        """Older agents omitted tool ``name``; Gemini rejects empty function_response.name."""
        import json

        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-abc",
                        "type": "function",
                        "function": {
                            "name": "list_tcs_pdf_inventory",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-abc",
                "content": json.dumps("PDFs: a.pdf"),
            },
        ]
        _, contents = messages_to_gemini_contents(messages)
        fr = contents[2]["parts"][0]["function_response"]
        assert fr["name"] == "list_tcs_pdf_inventory"

    def test_tool_result_infers_name_single_prior_call_without_matching_id(self):
        import json

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "only_tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": json.dumps("ok")},
        ]
        _, contents = messages_to_gemini_contents(messages)
        assert len(contents) == 2
        fr = contents[1]["parts"][0]["function_response"]
        assert fr["name"] == "only_tool"

    def test_assistant_with_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                        },
                    }
                ],
            }
        ]
        _, contents = messages_to_gemini_contents(messages)
        parts = contents[0]["parts"]
        assert len(parts) == 1
        assert "function_call" in parts[0]
        assert parts[0]["function_call"]["name"] == "get_weather"
        assert parts[0]["function_call"]["args"] == {"location": "SF"}

    def test_multimodal_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            }
        ]
        _, contents = messages_to_gemini_contents(messages)
        parts = contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "What's in this image?"}
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        assert parts[1]["inline_data"]["data"] == "abc123"

    def test_empty_messages(self):
        system, contents = messages_to_gemini_contents([])
        assert system is None
        assert contents == []

    def test_empty_content(self):
        messages = [{"role": "user", "content": ""}]
        _, contents = messages_to_gemini_contents(messages)
        assert len(contents) == 0 or contents[0]["parts"] == []
