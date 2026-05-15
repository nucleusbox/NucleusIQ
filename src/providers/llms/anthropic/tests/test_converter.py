"""Tools converter tests."""

from __future__ import annotations

from nucleusiq_anthropic.tools.converter import (
    spec_looks_native,
    to_anthropic_tool_definition,
)


def test_openai_tool_envelope() -> None:

    spec = {
        "type": "function",
        "function": {
            "name": "roll",
            "description": "Dice",
            "parameters": {
                "type": "object",
                "properties": {"faces": {"type": "integer"}},
                "required": ["faces"],
            },
        },
    }

    out = to_anthropic_tool_definition(spec)

    assert out["name"] == "roll"

    assert out["input_schema"]["type"] == "object"

    assert out["description"] == "Dice"


def test_non_object_parameters_wrapped():

    payload = {
        "name": "noop",
        "description": "",
        "parameters": {"type": "string"},
    }

    out = to_anthropic_tool_definition(payload)

    assert out["input_schema"]["type"] == "object"

    assert "value" in out["input_schema"]["properties"]


def test_input_schema_native_definition() -> None:

    spec = {
        "type": "function",
        "name": "alpha",
        "description": "Beta",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    }

    out = to_anthropic_tool_definition(spec)

    assert out["name"] == "alpha"

    assert out["input_schema"]["type"] == "object"


def test_server_tool_passthrough() -> None:

    spec = {"type": "web_search", "name": "stub"}

    assert to_anthropic_tool_definition(spec) == spec


def test_spec_looks_native_empty_registry_expectations() -> None:
    """``NATIVE_TOOL_TYPES`` is empty in Phase A — only explicit builtins match."""

    assert not spec_looks_native({"type": "web_search"})
    assert not spec_looks_native({"type": "function"})
    assert not spec_looks_native({"type": "anthropic_builtin", "name": "missing"})


def test_additional_properties_injected_when_absent() -> None:

    payload = {
        "type": "function",
        "function": {
            "name": "n",
            "description": "",
            "parameters": {"type": "object"},
        },
    }

    out = to_anthropic_tool_definition(payload)

    assert out["input_schema"]["additionalProperties"] is False
