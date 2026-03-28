"""Tests for nucleusiq_gemini.structured_output — builder and parser."""

import dataclasses
from typing import Optional

import pytest
from nucleusiq_gemini.structured_output.builder import build_gemini_response_format
from nucleusiq_gemini.structured_output.parser import parse_gemini_response
from pydantic import BaseModel

# ====================================================================== #
# Test schema types                                                        #
# ====================================================================== #


class WeatherResponse(BaseModel):
    temperature: float
    conditions: str
    humidity: Optional[float] = None


@dataclasses.dataclass
class SimpleData:
    name: str
    value: int


class TypedDictLike:
    name: str
    age: int


# ====================================================================== #
# Builder tests                                                            #
# ====================================================================== #


class TestBuildGeminiResponseFormat:
    def test_pydantic_model(self):
        result = build_gemini_response_format(WeatherResponse)
        assert result["response_mime_type"] == "application/json"
        assert "response_json_schema" in result
        schema = result["response_json_schema"]
        assert "properties" in schema
        assert "temperature" in schema["properties"]

    def test_pydantic_schema_cleaned(self):
        result = build_gemini_response_format(WeatherResponse)
        schema = result["response_json_schema"]
        assert "title" not in schema
        assert "$defs" not in schema
        assert "additionalProperties" not in schema

    def test_nullable_field(self):
        result = build_gemini_response_format(WeatherResponse)
        schema = result["response_json_schema"]
        humidity = schema["properties"]["humidity"]
        assert humidity.get("nullable") is True or "anyOf" in humidity

    def test_dataclass(self):
        result = build_gemini_response_format(SimpleData)
        assert result["response_mime_type"] == "application/json"
        schema = result["response_json_schema"]
        assert "name" in schema["properties"]
        assert "value" in schema["properties"]
        assert "name" in schema["required"]

    def test_dict_passthrough(self):
        raw = {
            "response_mime_type": "application/json",
            "response_json_schema": {"type": "object"},
        }
        result = build_gemini_response_format(raw)
        assert result == raw

    def test_annotated_class(self):
        result = build_gemini_response_format(TypedDictLike)
        assert result is not None
        assert result["response_mime_type"] == "application/json"

    def test_unknown_type(self):
        result = build_gemini_response_format(42)
        assert result is None


# ====================================================================== #
# Parser tests                                                             #
# ====================================================================== #


class TestParseGeminiResponse:
    def test_pydantic_model(self):
        msg = {"content": '{"temperature": 72.5, "conditions": "sunny"}'}
        result = parse_gemini_response(msg, WeatherResponse)
        assert isinstance(result, WeatherResponse)
        assert result.temperature == 72.5
        assert result.conditions == "sunny"

    def test_pydantic_with_optional(self):
        msg = {
            "content": '{"temperature": 72.5, "conditions": "sunny", "humidity": 45.0}'
        }
        result = parse_gemini_response(msg, WeatherResponse)
        assert result.humidity == 45.0

    def test_dataclass(self):
        msg = {"content": '{"name": "test", "value": 42}'}
        result = parse_gemini_response(msg, SimpleData)
        assert isinstance(result, SimpleData)
        assert result.name == "test"
        assert result.value == 42

    def test_dict_returns_dict(self):
        msg = {"content": '{"key": "value"}'}
        result = parse_gemini_response(msg, {"type": "object"})
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="empty content"):
            parse_gemini_response({"content": ""}, WeatherResponse)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_gemini_response({"content": "not json"}, WeatherResponse)

    def test_no_content_key_raises(self):
        with pytest.raises(ValueError, match="empty content"):
            parse_gemini_response({}, WeatherResponse)
