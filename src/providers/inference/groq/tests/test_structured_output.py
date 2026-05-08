"""Structured output builder, parser, and schema cleaner."""

from __future__ import annotations

import dataclasses

import pytest
from nucleusiq.agents.structured_output.errors import SchemaParseError
from nucleusiq_groq.structured_output.builder import build_response_format
from nucleusiq_groq.structured_output.cleaner import clean_schema_for_groq
from nucleusiq_groq.structured_output.parser import parse_response
from pydantic import BaseModel


class _M(BaseModel):
    answer: str
    n: int


def test_build_response_format_pydantic() -> None:
    rf = build_response_format(_M)
    assert rf is not None
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_M"
    assert rf["json_schema"]["strict"] is True


def test_build_response_format_dict_passthrough() -> None:
    d = {"type": "json_object"}
    assert build_response_format(d) == d


@dataclasses.dataclass
class _DC:
    x: str


def test_build_response_format_dataclass() -> None:
    rf = build_response_format(_DC)
    assert rf is not None
    assert rf["json_schema"]["schema"]["properties"]["x"]


def test_build_response_format_annotations_object() -> None:
    class _Ann:
        title: str

    rf = build_response_format(_Ann)
    assert rf is not None


def test_build_response_format_unknown_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    assert build_response_format(42) is None  # type: ignore[arg-type]
    assert "Unknown schema type" in caplog.text


def test_clean_schema_for_groq_sets_additional_properties() -> None:
    s = {"type": "object", "properties": {"a": {"type": "string", "title": "A"}}}
    out = clean_schema_for_groq(s)
    assert out["additionalProperties"] is False
    assert "required" in out


def test_parse_response_pydantic() -> None:
    msg = {"role": "assistant", "content": '{"answer": "yes", "n": 1}'}
    obj = parse_response(msg, _M)
    assert isinstance(obj, _M)
    assert obj.answer == "yes"


def test_parse_response_empty_content() -> None:
    from nucleusiq.agents.structured_output.errors import StructuredOutputError

    with pytest.raises(StructuredOutputError):
        parse_response({"content": ""}, _M)


def test_parse_response_invalid_json() -> None:
    with pytest.raises(SchemaParseError):
        parse_response({"content": "not json"}, _M)


def test_parse_response_dataclass() -> None:
    msg = {"content": '{"x": "hi"}'}
    obj = parse_response(msg, _DC)
    assert obj.x == "hi"


def test_parse_response_dict_schema_returns_raw_dict() -> None:
    msg = {"content": '{"a": 1}'}
    assert parse_response(msg, {"type": "object"}) == {"a": 1}


def test_clean_schema_inline_ref() -> None:
    schema = {
        "type": "object",
        "properties": {"r": {"$ref": "#/$defs/Inner"}},
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"k": {"type": "string"}},
            },
        },
    }
    out = clean_schema_for_groq(schema)
    assert "properties" in out
