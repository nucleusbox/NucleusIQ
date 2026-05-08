"""Tests for Chat Completions request sanitization and payload building."""

from __future__ import annotations

import pytest
from nucleusiq.llms.errors import InvalidRequestError
from nucleusiq_groq._shared.wire import (
    build_chat_completion_payload,
    filter_unsupported_kwargs,
    sanitize_messages,
    validate_sampling_count,
)

from tests.shared.constants import GROQ_OPENAI_COMPAT_V1_BASE


def test_groq_openai_compat_base_reference() -> None:
    assert "api.groq.com" in GROQ_OPENAI_COMPAT_V1_BASE
    assert GROQ_OPENAI_COMPAT_V1_BASE.endswith("/openai/v1")


def test_sanitize_messages_strips_name() -> None:
    msgs = [{"role": "user", "content": "hi", "name": "alice"}]
    out = sanitize_messages(msgs)
    assert out == [{"role": "user", "content": "hi"}]
    assert msgs[0].get("name") == "alice"


def test_sanitize_messages_coerces_flat_tool_calls() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
            ],
        }
    ]
    out = sanitize_messages(msgs)
    tc0 = out[0]["tool_calls"][0]
    assert tc0["type"] == "function"
    assert tc0["id"] == "call_1"
    assert tc0["function"]["name"] == "get_weather"
    assert tc0["function"]["arguments"] == '{"city":"Paris"}'


def test_sanitize_messages_leaves_nested_tool_calls() -> None:
    nested = {
        "id": "x",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }
    msgs = [{"role": "assistant", "tool_calls": [nested]}]
    out = sanitize_messages(msgs)
    assert out[0]["tool_calls"][0] == nested


def test_filter_unsupported_kwargs() -> None:
    raw = {"logprobs": True, "temperature": 0.5, "foo": 1}
    assert filter_unsupported_kwargs(raw) == {"temperature": 0.5, "foo": 1}


def test_validate_sampling_count_accepts_none_and_one() -> None:
    validate_sampling_count(None)
    validate_sampling_count(1)


def test_validate_sampling_count_rejects_other_n() -> None:
    with pytest.raises(InvalidRequestError):
        validate_sampling_count(2)


def test_build_chat_completion_payload_minimal() -> None:
    p = build_chat_completion_payload(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=64,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        parallel_tool_calls=None,
        seed=None,
        user=None,
        extra={},
    )
    assert p["model"] == "m"
    assert p["max_tokens"] == 64
    assert p["messages"] == [{"role": "user", "content": "x"}]


def test_build_chat_completion_payload_optional_fields() -> None:
    fmt = {"type": "json_object"}
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    p = build_chat_completion_payload(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=10,
        temperature=None,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        stop=["\n"],
        tools=tools,
        tool_choice="auto",
        response_format=fmt,
        parallel_tool_calls=True,
        seed=42,
        user="u1",
        extra={"metadata": {"k": "v"}},
    )
    assert "temperature" not in p
    assert p["stop"] == ["\n"]
    assert p["tools"] == tools
    assert p["response_format"] == fmt
    assert p["parallel_tool_calls"] is True
    assert p["seed"] == 42
    assert p["user"] == "u1"
    assert p["metadata"] == {"k": "v"}


def test_build_chat_completion_payload_strips_unsupported_extra() -> None:
    p = build_chat_completion_payload(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        parallel_tool_calls=None,
        seed=None,
        user=None,
        extra={"logit_bias": {"a": 1}},
    )
    assert "logit_bias" not in p
