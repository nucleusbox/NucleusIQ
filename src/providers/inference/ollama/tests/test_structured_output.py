"""Structured output builder/smoke."""

from __future__ import annotations

from nucleusiq_ollama.structured_output import build_ollama_format
from pydantic import BaseModel


class M(BaseModel):
    title: str
    count: int


def test_build_ollama_format_pydantic() -> None:
    fmt = build_ollama_format(M)
    assert isinstance(fmt, dict)
    assert fmt.get("type") == "object"
    assert "title" in (fmt.get("properties") or {})


def test_build_json_string() -> None:
    assert build_ollama_format("json") == "json"


def test_build_openai_style_json_schema_wrapper() -> None:
    wrapped = {
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }
    fmt = build_ollama_format(wrapped)
    assert isinstance(fmt, dict)
    assert "a" in (fmt.get("properties") or {})


def test_build_generic_json_envelope_with_schema() -> None:
    """Matches OutputSchema.for_provider for non-OpenAI providers (ollama, groq, …)."""
    wrapped = {
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {"b": {"type": "integer"}},
            "required": ["b"],
            "additionalProperties": False,
        },
    }
    fmt = build_ollama_format(wrapped)
    assert isinstance(fmt, dict)
    assert fmt.get("type") == "object"
    assert "b" in (fmt.get("properties") or {})


def test_build_generic_json_envelope_bare_json_mode() -> None:
    assert build_ollama_format({"type": "json"}) == "json"
