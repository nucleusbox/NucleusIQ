"""Tests for chat models and execution context protocol."""

from __future__ import annotations

from types import SimpleNamespace

from nucleusiq.agents.chat_models import (
    ChatMessage,
    ToolCallRequest,
    messages_to_dicts,
)
from nucleusiq.agents.config import AgentConfig, AgentState
from nucleusiq.agents.execution_context import ExecutionContext


class _DummyCtx:
    llm = None
    tools = []
    memory = None
    prompt = None
    config = AgentConfig()
    role = "assistant"
    objective = "help"
    state = AgentState.INITIALIZING
    response_format = None

    @property
    def _logger(self):
        return None

    @property
    def _executor(self):
        return None

    @property
    def _current_llm_overrides(self):
        return {}

    def _resolve_response_format(self):
        return None

    def _get_structured_output_kwargs(self, output_config):
        return {}

    def _wrap_structured_output_result(self, response, output_config):
        return response


def test_execution_context_runtime_protocol():
    assert isinstance(_DummyCtx(), ExecutionContext)


def test_tool_call_request_from_raw_dict_and_object():
    tc1 = ToolCallRequest.from_raw(
        {
            "id": "call_1",
            "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
        }
    )
    assert tc1.id == "call_1"
    assert tc1.name == "add"
    assert tc1.arguments == '{"a":1,"b":2}'

    obj = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(name="mul", arguments='{"a":3,"b":4}'),
    )
    tc2 = ToolCallRequest.from_raw(obj)
    assert tc2.id == "call_2"
    assert tc2.name == "mul"


def test_tool_call_request_handles_missing_function_info():
    tc = ToolCallRequest.from_raw({"id": "x", "function": "not-a-dict"})
    assert tc.id == "x"
    assert tc.name == ""
    assert tc.arguments == "{}"

    tc_obj = ToolCallRequest.from_raw(SimpleNamespace(id="y", function=None))
    assert tc_obj.id == "y"
    assert tc_obj.name == ""


def test_chat_message_roundtrip_and_messages_to_dicts():
    msg = ChatMessage(
        role="assistant",
        content="ok",
        tool_calls=[ToolCallRequest(id="c1", name="add", arguments="{}")],
        name="assistant-name",
    )
    as_dict = msg.to_dict()
    assert as_dict["role"] == "assistant"
    assert as_dict["tool_calls"][0]["function"]["name"] == "add"

    parsed = ChatMessage.from_dict(as_dict)
    assert parsed.role == "assistant"
    assert parsed.tool_calls is not None
    assert parsed.tool_calls[0].name == "add"

    out = messages_to_dicts([parsed])
    assert out[0]["role"] == "assistant"


def test_tool_call_request_passthrough_instance():
    original = ToolCallRequest(id="x1", name="noop", arguments="{}")
    parsed = ToolCallRequest.from_raw(original)
    assert parsed is original


def test_chat_message_from_dict_defaults_role_and_handles_no_tool_calls():
    parsed = ChatMessage.from_dict({"content": "hello"})
    assert parsed.role == "user"
    assert parsed.content == "hello"
    assert parsed.tool_calls is None


def test_chat_message_to_dict_includes_tool_call_id_and_name():
    msg = ChatMessage(
        role="tool",
        content="done",
        tool_call_id="call_1",
        name="tool-name",
    )
    d = msg.to_dict()
    assert d["tool_call_id"] == "call_1"
    assert d["name"] == "tool-name"
