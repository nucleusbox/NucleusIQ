"""Tests for nucleusiq_gemini.tools.tool_converter."""

from nucleusiq_gemini.tools.tool_converter import (
    build_tools_payload,
    convert_tool_spec,
)


class TestConvertToolSpec:
    def test_basic_function_tool(self):
        spec = {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        }
        result = convert_tool_spec(spec)
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"
        assert "parameters" in result

    def test_native_tool_passthrough(self):
        spec = {"type": "google_search", "google_search": {}}
        result = convert_tool_spec(spec)
        assert result == spec

    def test_removes_additional_properties(self):
        spec = {
            "name": "fn",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": False,
            },
        }
        result = convert_tool_spec(spec)
        assert "additionalProperties" not in result["parameters"]

    def test_removes_defs(self):
        spec = {
            "name": "fn",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {},
                "$defs": {"SomeType": {}},
            },
        }
        result = convert_tool_spec(spec)
        assert "$defs" not in result["parameters"]

    def test_cleans_property_titles(self):
        spec = {
            "name": "fn",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "title": "Name"},
                },
            },
        }
        result = convert_tool_spec(spec)
        assert "title" not in result["parameters"]["properties"]["name"]

    def test_handles_anyof_nullable(self):
        spec = {
            "name": "fn",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                },
            },
        }
        result = convert_tool_spec(spec)
        prop = result["parameters"]["properties"]["name"]
        assert prop.get("nullable") is True
        assert prop.get("type") == "string"

    def test_empty_parameters(self):
        spec = {"name": "fn", "description": "desc"}
        result = convert_tool_spec(spec)
        assert result["parameters"] == {}

    def test_missing_description(self):
        spec = {"name": "fn", "parameters": {"type": "object"}}
        result = convert_tool_spec(spec)
        assert result["description"] == ""

    def test_nested_properties_cleaned(self):
        spec = {
            "name": "fn",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "inner": {
                                "type": "string",
                                "title": "Inner",
                                "default": "x",
                            },
                        },
                    },
                },
            },
        }
        result = convert_tool_spec(spec)
        inner = result["parameters"]["properties"]["data"]["properties"]["inner"]
        assert "title" not in inner
        assert "default" not in inner

    def test_array_items_cleaned(self):
        spec = {
            "name": "fn",
            "description": "desc",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "title": "Tag"},
                    },
                },
            },
        }
        result = convert_tool_spec(spec)
        assert "title" not in result["parameters"]["properties"]["tags"]["items"]


class TestBuildToolsPayload:
    def test_function_declarations(self):
        decls = [
            {"name": "fn1", "description": "d1", "parameters": {}},
            {"name": "fn2", "description": "d2", "parameters": {}},
        ]
        result = build_tools_payload(decls)
        assert len(result) == 1
        assert "function_declarations" in result[0]
        assert len(result[0]["function_declarations"]) == 2

    def test_native_tools_separate(self):
        decls = [
            {"name": "fn1", "description": "d1", "parameters": {}},
            {"type": "google_search", "google_search": {}},
        ]
        result = build_tools_payload(decls)
        assert len(result) == 2
        assert "function_declarations" in result[0]
        assert result[1]["type"] == "google_search"

    def test_only_native_tools(self):
        decls = [
            {"type": "google_search", "google_search": {}},
            {"type": "code_execution", "code_execution": {}},
        ]
        result = build_tools_payload(decls)
        assert len(result) == 2
        assert all("function_declarations" not in r for r in result)

    def test_empty_list(self):
        result = build_tools_payload([])
        assert result == []
