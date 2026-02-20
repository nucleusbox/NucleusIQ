"""Additional tests for base_tool.py — Pydantic schema edge cases."""

import pytest
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

from nucleusiq.tools.base_tool import _pydantic_model_to_json_schema, BaseTool


class ToolWithEnum(BaseModel):
    color: Literal["red", "green", "blue"] = Field(description="Pick a color")
    size: int = Field(default=10, description="Size", ge=1, le=100)


class ToolWithDefault(BaseModel):
    name: str = Field(description="Name")
    optional_field: Optional[str] = Field(default=None, description="Optional")


class TestPydanticSchemaEdgeCases:

    def test_enum_field(self):
        schema = _pydantic_model_to_json_schema(ToolWithEnum)
        color_prop = schema["properties"]["color"]
        assert "enum" in color_prop
        assert set(color_prop["enum"]) == {"red", "green", "blue"}

    def test_default_value(self):
        schema = _pydantic_model_to_json_schema(ToolWithDefault)
        assert "name" in schema["properties"]

    def test_constraints(self):
        schema = _pydantic_model_to_json_schema(ToolWithEnum)
        size_prop = schema["properties"]["size"]
        assert "default" in size_prop or "type" in size_prop

    def test_from_function_with_enum_schema(self):
        def my_fn(color: str, size: int):
            return f"{color}:{size}"

        tool = BaseTool.from_function(my_fn, args_schema=ToolWithEnum)
        spec = tool.get_spec()
        assert "color" in spec["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_shutdown_and_initialize(self):
        tool = BaseTool.from_function(lambda: None, name="t", description="d")
        tool.shutdown()
        await tool.initialize()
