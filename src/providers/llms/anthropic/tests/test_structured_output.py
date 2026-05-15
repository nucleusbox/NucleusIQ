"""Anthropic structured output builder + parser."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from nucleusiq_anthropic.structured_output import (
    build_anthropic_output_config,
    parse_anthropic_response,
)
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class _Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150)


def test_build_from_pydantic_wraps_json_schema_format() -> None:
    oc = build_anthropic_output_config(_Person)
    assert oc is not None
    fmt = oc.get("format")
    assert isinstance(fmt, dict)
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["type"] == "object"


def test_build_dict_with_format_pass_through() -> None:
    src = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    oc = build_anthropic_output_config(src)
    assert oc == src


def test_build_dict_effort_only_pass_through() -> None:
    oc = build_anthropic_output_config({"effort": "low"})
    assert oc == {"effort": "low"}


def test_build_dict_inner_json_schema_shape() -> None:
    inner = {"type": "json_schema", "schema": {"type": "object", "properties": {}}}
    oc = build_anthropic_output_config(inner)
    assert oc == {"format": inner}


def test_build_raw_json_schema_object() -> None:
    rooted = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    oc = build_anthropic_output_config(rooted)
    assert oc is not None
    assert oc["format"]["schema"] == rooted


def test_parser_pydantic() -> None:
    data = '{"name":"Ada","age":41}'
    out = parse_anthropic_response({"content": data}, _Person)
    assert isinstance(out, _Person)
    assert out.name == "Ada"


@dataclass
class _DC:
    a: str
    b: int


def test_parser_dataclass() -> None:
    out = parse_anthropic_response({"content": '{"a":"hi","b":3}'}, _DC)
    assert out == _DC("hi", 3)


class _NX(BaseModel):
    x: int


@pytest.mark.asyncio
async def test_tuple_response_format_merges_user_output_config(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from types import SimpleNamespace

    from nucleusiq_anthropic.nb_anthropic.base import BaseAnthropic

    llm = BaseAnthropic(async_mode=True)
    captured: dict = {}

    async def _spy(**kw):
        captured.update(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"x":42}')],
            usage=SimpleNamespace(
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="m",
            id="i",
        )

    llm._client.messages.create = _spy
    hint = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    parsed = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        output_config={"effort": "low"},
        response_format=(hint, _NX),
    )
    assert isinstance(parsed, _NX)
    oc = captured["output_config"]
    assert oc["effort"] == "low"
    assert oc["format"]["type"] == "json_schema"


class _Inner(BaseModel):
    code: str


class _Nested(BaseModel):
    label: str
    inner: _Inner


def test_build_nested_pydantic_inlines_defs() -> None:
    oc = build_anthropic_output_config(_Nested)
    assert oc is not None
    sch = oc["format"]["schema"]
    assert sch.get("type") == "object"


@dataclass
class _Tagged:
    items: list[str]


def test_build_simple_dataclass() -> None:
    oc = build_anthropic_output_config(_Tagged)
    assert oc is not None
    assert oc["format"]["schema"]["type"] == "object"


class _ScoreTd(TypedDict):
    hits: int


def test_build_and_parse_typed_dict_style() -> None:
    oc = build_anthropic_output_config(_ScoreTd)
    assert oc is not None
    blob = '{"hits":9}'
    out = parse_anthropic_response({"content": blob}, _ScoreTd)
    assert out == {"hits": 9}


class _Opt(BaseModel):
    score: float | None = None


def test_optional_field_triggers_anyof_clean_branch() -> None:
    oc = build_anthropic_output_config(_Opt)
    assert oc is not None
    props = oc["format"]["schema"].get("properties", {})
    assert "score" in props


def test_non_type_input_returns_none() -> None:
    assert build_anthropic_output_config(123) is None

    assert build_anthropic_output_config("x") is None


def test_parser_with_schema_mapping_returns_dict() -> None:
    schema_hint: dict[str, object] = {"note": "hint"}
    out = parse_anthropic_response({"content": '{"a":1}'}, schema_hint)
    assert out == {"a": 1}


def test_parser_fallback_non_schema_indicator_returns_raw() -> None:
    out = parse_anthropic_response({"content": '{"v":true}'}, 999)
    assert out == {"v": True}


def test_merge_output_config_wire_merges_format_and_top_level_keys() -> None:
    from nucleusiq_anthropic.nb_anthropic.base import _merge_output_config_wire

    a = {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    b = {
        "format": {"extra": True},
        "effort": "high",
    }
    merged = _merge_output_config_wire(a, b)
    assert merged["effort"] == "high"
    assert merged["format"]["type"] == "json_schema"
    assert merged["format"]["extra"] is True


def test_parser_invalid_json_raises() -> None:
    from nucleusiq.agents.structured_output.errors import SchemaParseError

    with pytest.raises(SchemaParseError):
        parse_anthropic_response({"content": "{{"}, _Person)


def test_parser_requires_non_empty_body() -> None:
    from nucleusiq.agents.structured_output.errors import StructuredOutputError

    with pytest.raises(StructuredOutputError):
        parse_anthropic_response({"content": ""}, _Person)
