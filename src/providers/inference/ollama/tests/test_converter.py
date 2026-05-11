"""Tool spec conversion."""

from __future__ import annotations

from nucleusiq_ollama.tools.converter import to_ollama_function_tool


def test_wrap_flat_spec() -> None:
    spec = {
        "name": "add",
        "description": "add nums",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
    }
    out = to_ollama_function_tool(spec)
    assert out["type"] == "function"
    assert out["function"]["name"] == "add"
    assert out["function"]["parameters"]["type"] == "object"


def test_passthrough_openai_shape() -> None:
    spec = {
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
    }
    assert to_ollama_function_tool(spec) == spec
