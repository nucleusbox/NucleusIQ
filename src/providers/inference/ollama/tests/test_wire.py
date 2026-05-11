"""Wire helpers for Ollama chat (messages + options)."""

from __future__ import annotations

from nucleusiq_ollama._shared.wire import (
    build_chat_kwargs,
    build_options,
    sanitize_messages,
    tool_arguments_to_json_string,
)


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


def test_build_options_maps_num_predict_and_stop_scalar() -> None:
    o = build_options(
        max_output_tokens=256,
        temperature=0.2,
        top_p=0.95,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        stop=["."],
        seed=42,
    )
    assert o["num_predict"] == 256
    assert o["temperature"] == 0.2
    assert o["stop"] == "."


def test_build_chat_kwargs_includes_think_and_format() -> None:
    fmt = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    kwargs = build_chat_kwargs(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        format_payload=fmt,
        options={},
        think=True,
        keep_alive="5m",
        stream=False,
        tool_choice="auto",
    )
    assert kwargs["model"] == "m"
    assert kwargs["format"] == fmt
    assert kwargs["think"] is True
    assert kwargs["keep_alive"] == "5m"


def test_tool_arguments_to_json_string() -> None:
    assert tool_arguments_to_json_string({"x": 1}) == '{"x": 1}'
    assert tool_arguments_to_json_string("{}") == "{}"


def test_sanitize_messages_non_dict_tool_call_entry() -> None:
    msgs = [{"role": "assistant", "tool_calls": [None, object()]}]
    out = sanitize_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["name"] == ""
