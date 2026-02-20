"""Tests for tools/base_tool.py: BaseTool, FunctionTool, schema helpers."""

import pytest
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from nucleusiq.tools.base_tool import (
    BaseTool,
    _parse_annotation,
    _pydantic_model_to_json_schema,
)


# ── Fixture models ───────────────────────────────────────────────────────────

class CalcArgs(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_annotation
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseAnnotation:

    def test_int(self):
        assert _parse_annotation(int) == "integer"

    def test_float(self):
        assert _parse_annotation(float) == "number"

    def test_bool(self):
        assert _parse_annotation(bool) == "boolean"

    def test_str(self):
        assert _parse_annotation(str) == "string"

    def test_dict(self):
        assert _parse_annotation(dict) == "object"

    def test_list(self):
        assert _parse_annotation(list) == "array"

    def test_unknown_defaults_to_string(self):
        assert _parse_annotation(bytes) == "string"

    def test_generic_list(self):
        assert _parse_annotation(List[int]) == "array"


# ═══════════════════════════════════════════════════════════════════════════════
# _pydantic_model_to_json_schema
# ═══════════════════════════════════════════════════════════════════════════════


class TestPydanticModelToJsonSchema:

    def test_valid_model(self):
        schema = _pydantic_model_to_json_schema(CalcArgs)
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert "a" in schema["required"]

    def test_not_a_model_raises(self):
        with pytest.raises(TypeError, match="Expected Pydantic BaseModel"):
            _pydantic_model_to_json_schema(dict)


# ═══════════════════════════════════════════════════════════════════════════════
# BaseTool.from_function (FunctionTool)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFromFunction:

    def test_basic(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = BaseTool.from_function(add)
        assert tool.name == "add"
        assert tool.description == "Add two numbers."
        spec = tool.get_spec()
        assert spec["name"] == "add"
        assert "a" in spec["parameters"]["properties"]
        assert "a" in spec["parameters"]["required"]

    def test_custom_name_desc(self):
        def my_fn(x: str):
            pass

        tool = BaseTool.from_function(my_fn, name="custom", description="Custom desc")
        assert tool.name == "custom"
        assert tool.description == "Custom desc"

    @pytest.mark.asyncio
    async def test_execute_plain(self):
        def multiply(a: int, b: int) -> int:
            return a * b

        tool = BaseTool.from_function(multiply)
        result = await tool.execute(a=3, b=4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_execute_with_schema(self):
        def calc(a: int, b: int) -> int:
            return a + b

        tool = BaseTool.from_function(calc, args_schema=CalcArgs)
        result = await tool.execute(a=10, b=20)
        assert result == 30

    @pytest.mark.asyncio
    async def test_initialize(self):
        tool = BaseTool.from_function(lambda: None)
        await tool.initialize()

    def test_spec_with_schema(self):
        def fn(a: int, b: int):
            pass

        tool = BaseTool.from_function(fn, args_schema=CalcArgs)
        spec = tool.get_spec()
        assert spec["parameters"]["type"] == "object"
        assert "a" in spec["parameters"]["properties"]

    def test_no_annotation_defaults_str(self):
        def fn(x):
            pass

        tool = BaseTool.from_function(fn)
        spec = tool.get_spec()
        assert spec["parameters"]["properties"]["x"]["type"] == "string"

    def test_optional_param(self):
        def fn(name: str, label: str = "default"):
            pass

        tool = BaseTool.from_function(fn)
        spec = tool.get_spec()
        assert "name" in spec["parameters"]["required"]
        assert "label" not in spec["parameters"]["required"]


# ═══════════════════════════════════════════════════════════════════════════════
# BaseTool instance methods
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseToolMethods:

    def _make_tool(self):
        def noop():
            pass
        return BaseTool.from_function(noop, name="test_tool", description="Test")

    def test_to_dict_without_version(self):
        tool = self._make_tool()
        d = tool.to_dict()
        assert d["name"] == "test_tool"
        assert "version" not in d

    def test_to_dict_with_version(self):
        tool = self._make_tool()
        tool.version = "1.0"
        d = tool.to_dict()
        assert d["version"] == "1.0"

    def test_shutdown(self):
        tool = self._make_tool()
        tool.shutdown()
