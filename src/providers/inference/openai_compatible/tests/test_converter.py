"""Tool-spec conversion to the OpenAI function-tool shape."""

from __future__ import annotations

import pytest
from nucleusiq_openai_compatible.tools.converter import (
    STRIPPED_SCHEMA_KEYS,
    convert_tool_spec,
    convert_tool_specs,
)

PARAMS = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
}


class TestInputShapes:
    def test_bare_spec(self) -> None:
        result = convert_tool_spec(
            {"name": "search", "description": "Find things", "parameters": PARAMS}
        )
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Find things"
        assert result["function"]["parameters"] == PARAMS

    def test_already_wrapped_spec(self) -> None:
        wrapped = {
            "type": "function",
            "function": {"name": "search", "parameters": PARAMS},
        }
        assert convert_tool_spec(wrapped)["function"]["name"] == "search"

    def test_conversion_is_idempotent(self) -> None:
        once = convert_tool_spec({"name": "search", "parameters": PARAMS})
        assert convert_tool_spec(once) == once

    def test_unknown_top_level_keys_dropped(self) -> None:
        result = convert_tool_spec(
            {"name": "search", "parameters": PARAMS, "executor": object()}
        )
        assert set(result["function"]) <= {
            "name",
            "description",
            "parameters",
            "strict",
        }

    def test_type_function_without_function_dict(self) -> None:
        result = convert_tool_spec({"type": "function", "name": "search"})
        assert result["function"]["name"] == "search"

    def test_input_not_mutated(self) -> None:
        spec = {"name": "search", "parameters": {"type": "object", "title": "X"}}
        convert_tool_spec(spec)
        assert spec["parameters"] == {"type": "object", "title": "X"}


class TestName:
    def test_whitespace_stripped(self) -> None:
        assert convert_tool_spec({"name": "  search  "})["function"]["name"] == "search"

    @pytest.mark.parametrize("name", [None, "", "   ", 42, ["search"]])
    def test_missing_name_rejected(self, name: object) -> None:
        with pytest.raises(ValueError, match="missing a function name"):
            convert_tool_spec({"name": name})

    def test_error_shows_the_offending_spec(self) -> None:
        with pytest.raises(ValueError, match="broken"):
            convert_tool_spec({"description": "broken"})


class TestDescription:
    def test_stripped(self) -> None:
        result = convert_tool_spec({"name": "s", "description": "  Find  "})
        assert result["function"]["description"] == "Find"

    @pytest.mark.parametrize("description", [None, "", "   ", 42])
    def test_unusable_description_omitted(self, description: object) -> None:
        result = convert_tool_spec({"name": "s", "description": description})
        assert "description" not in result["function"]


class TestParameters:
    def test_missing_parameters_get_an_empty_object_schema(self) -> None:
        # A function with no schema at all is rejected by strict validators.
        assert convert_tool_spec({"name": "ping"})["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    @pytest.mark.parametrize("parameters", [None, {}, "not-a-dict", []])
    def test_unusable_parameters_replaced(self, parameters: object) -> None:
        result = convert_tool_spec({"name": "ping", "parameters": parameters})
        assert result["function"]["parameters"]["type"] == "object"

    @pytest.mark.parametrize("key", sorted(STRIPPED_SCHEMA_KEYS))
    def test_non_semantic_keys_stripped(self, key: str) -> None:
        result = convert_tool_spec(
            {"name": "s", "parameters": {"type": "object", key: "x"}}
        )
        assert key not in result["function"]["parameters"]

    def test_stripping_is_recursive(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "nested": {"type": "object", "title": "Nested", "$defs": {}}
            },
        }
        cleaned = convert_tool_spec({"name": "s", "parameters": schema})
        nested = cleaned["function"]["parameters"]["properties"]["nested"]
        assert "title" not in nested and "$defs" not in nested

    def test_stripping_descends_into_lists(self) -> None:
        schema = {
            "type": "object",
            "anyOf": [{"type": "string", "title": "A"}, {"type": "integer"}],
        }
        cleaned = convert_tool_spec({"name": "s", "parameters": schema})
        assert "title" not in cleaned["function"]["parameters"]["anyOf"][0]

    def test_semantic_keys_survive(self) -> None:
        schema = {
            "type": "object",
            "properties": {"q": {"type": "string", "enum": ["a", "b"]}},
            "required": ["q"],
            "additionalProperties": False,
        }
        cleaned = convert_tool_spec({"name": "s", "parameters": schema})["function"][
            "parameters"
        ]
        assert cleaned["required"] == ["q"]
        assert cleaned["additionalProperties"] is False
        assert cleaned["properties"]["q"]["enum"] == ["a", "b"]

    def test_scalar_schema_passes_through(self) -> None:
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        assert (
            convert_tool_spec({"name": "s", "parameters": schema})["function"][
                "parameters"
            ]
            == schema
        )


class TestStrict:
    def test_forwarded_when_true(self) -> None:
        assert convert_tool_spec({"name": "s", "strict": True})["function"]["strict"]

    @pytest.mark.parametrize("value", [False, None, "true", 1])
    def test_omitted_otherwise(self, value: object) -> None:
        result = convert_tool_spec({"name": "s", "strict": value})
        assert "strict" not in result["function"], (
            "only a literal True should set strict; many servers reject the key"
        )


class TestBatch:
    def test_order_preserved(self) -> None:
        specs = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert [t["function"]["name"] for t in convert_tool_specs(specs)] == [
            "a",
            "b",
            "c",
        ]

    def test_empty_list(self) -> None:
        assert convert_tool_specs([]) == []

    def test_one_bad_spec_fails_the_batch(self) -> None:
        with pytest.raises(ValueError):
            convert_tool_specs([{"name": "ok"}, {"description": "no name"}])
