"""Tool spec conversion tests."""

from __future__ import annotations

from nucleusiq_groq.tools.converter import to_openai_function_tool


def test_to_openai_function_tool_wraps_plain_spec() -> None:
    spec = {
        "name": "add",
        "description": "add numbers",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
    }
    out = to_openai_function_tool(spec)
    assert out["type"] == "function"
    assert out["function"]["name"] == "add"
    assert out["function"]["description"] == "add numbers"


def test_to_openai_function_tool_passes_through_openai_shape() -> None:
    wrapped = {
        "type": "function",
        "function": {"name": "x", "description": "d", "parameters": {}},
    }
    assert to_openai_function_tool(wrapped) is wrapped


def test_to_openai_function_tool_default_parameters() -> None:
    out = to_openai_function_tool({"name": "n", "description": ""})
    assert out["function"]["parameters"]["type"] == "object"
