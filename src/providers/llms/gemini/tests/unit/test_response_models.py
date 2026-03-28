"""Tests for nucleusiq_gemini._shared.response_models."""

from nucleusiq_gemini._shared.response_models import (
    AssistantMessage,
    GeminiLLMResponse,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
    _Choice,
)


class TestToolCallFunction:
    def test_construction(self):
        f = ToolCallFunction(name="get_weather", arguments='{"city": "SF"}')
        assert f.name == "get_weather"
        assert f.arguments == '{"city": "SF"}'

    def test_serialization(self):
        f = ToolCallFunction(name="fn", arguments="{}")
        d = f.model_dump()
        assert d == {"name": "fn", "arguments": "{}"}


class TestToolCall:
    def test_defaults(self):
        tc = ToolCall(id="c1", function=ToolCallFunction(name="fn", arguments="{}"))
        assert tc.type == "function"
        assert tc.id == "c1"

    def test_custom_type(self):
        tc = ToolCall(
            id="c2",
            type="custom",
            function=ToolCallFunction(name="fn", arguments="{}"),
        )
        assert tc.type == "custom"


class TestAssistantMessage:
    def test_defaults(self):
        m = AssistantMessage()
        assert m.role == "assistant"
        assert m.content is None
        assert m.tool_calls is None
        assert m.native_outputs is None

    def test_with_content(self):
        m = AssistantMessage(content="Hello")
        assert m.content == "Hello"

    def test_with_tool_calls(self):
        tc = ToolCall(id="c1", function=ToolCallFunction(name="fn", arguments="{}"))
        m = AssistantMessage(tool_calls=[tc])
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].id == "c1"

    def test_to_dict_minimal(self):
        m = AssistantMessage(content="Hi")
        d = m.to_dict()
        assert d == {"role": "assistant", "content": "Hi"}
        assert "tool_calls" not in d

    def test_to_dict_with_tool_calls(self):
        tc = ToolCall(id="c1", function=ToolCallFunction(name="fn", arguments="{}"))
        m = AssistantMessage(content=None, tool_calls=[tc])
        d = m.to_dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["id"] == "c1"

    def test_to_dict_with_native_outputs(self):
        m = AssistantMessage(native_outputs=[{"type": "thinking", "text": "hmm"}])
        d = m.to_dict()
        assert "_native_outputs" in d

    def test_alias_population(self):
        m = AssistantMessage(**{"_native_outputs": [{"type": "test"}]})
        assert m.native_outputs == [{"type": "test"}]


class TestUsageInfo:
    def test_defaults(self):
        u = UsageInfo()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.thoughts_tokens == 0
        assert u.cached_tokens == 0

    def test_custom_values(self):
        u = UsageInfo(
            prompt_tokens=10, completion_tokens=20, total_tokens=30, thoughts_tokens=5
        )
        assert u.prompt_tokens == 10
        assert u.thoughts_tokens == 5


class TestGeminiLLMResponse:
    def test_minimal(self):
        msg = AssistantMessage(content="Hi")
        choice = _Choice(message=msg)
        r = GeminiLLMResponse(choices=[choice])
        assert len(r.choices) == 1
        assert r.choices[0].message.content == "Hi"
        assert r.usage is None

    def test_with_usage(self):
        msg = AssistantMessage(content="Hi")
        r = GeminiLLMResponse(
            choices=[_Choice(message=msg)],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        assert r.usage.total_tokens == 15

    def test_with_model(self):
        msg = AssistantMessage(content="Hi")
        r = GeminiLLMResponse(
            choices=[_Choice(message=msg)],
            model="gemini-2.5-flash",
        )
        assert r.model == "gemini-2.5-flash"
