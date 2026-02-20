"""Additional parser.py coverage: TypedDict, Literal, code-block edge cases."""

import json
import sys
import pytest
from typing import Any, Dict, List, Optional, Union, TypedDict

from pydantic import BaseModel

from nucleusiq.agents.structured_output.parser import (
    is_typed_dict,
    schema_to_json,
    extract_json,
    parse_schema,
    validate_output,
    _type_to_json,
    _typeddict_to_json,
    _clean_schema,
    _dataclass_to_json,
)
from nucleusiq.agents.structured_output.errors import SchemaParseError, SchemaValidationError


# ── TypedDict fixture (must use typing.TypedDict for detection) ──────────────

class UserTD(TypedDict, total=False):
    name: str
    age: int


class StrictTD(TypedDict):
    name: str
    value: int


# ═══════════════════════════════════════════════════════════════════════════════
# TypedDict detection and conversion (lines 153-172)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypedDict:

    def test_is_typed_dict_true(self):
        assert is_typed_dict(UserTD) is True

    def test_is_typed_dict_false(self):
        assert is_typed_dict(dict) is False

    def test_typeddict_to_json(self):
        result = _typeddict_to_json(StrictTD)
        assert result["type"] == "object"
        assert "name" in result["properties"]
        assert "value" in result["properties"]

    def test_typeddict_required_keys(self):
        result = schema_to_json(StrictTD)
        assert "required" in result

    def test_typeddict_optional_keys(self):
        result = schema_to_json(UserTD)
        assert result["type"] == "object"


# ═══════════════════════════════════════════════════════════════════════════════
# Literal type (line 225-227)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiteralType:

    def test_literal_enum(self):
        from typing import Literal
        result = _type_to_json(Literal["red", "green", "blue"])
        assert result["type"] == "string"
        assert set(result["enum"]) == {"red", "green", "blue"}


# ═══════════════════════════════════════════════════════════════════════════════
# Union types (line 210)
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnionTypes:

    def test_union_multiple(self):
        result = _type_to_json(Union[str, int, float])
        assert "anyOf" in result
        assert len(result["anyOf"]) == 3

    def test_bare_list(self):
        result = _type_to_json(List[str])
        assert result["type"] == "array"
        assert result["items"]["type"] == "string"

    def test_bytes_type(self):
        result = _type_to_json(bytes)
        assert result["type"] == "string"
        assert result.get("format") == "binary"


# ═══════════════════════════════════════════════════════════════════════════════
# extract_json edge cases (lines 343-344, 352-353, 358)
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractJsonEdgeCases:

    def test_code_block_invalid_then_valid(self):
        text = '```json\n{invalid}\n```\n```json\n{"valid": true}\n```'
        result = extract_json(text)
        assert json.loads(result) == {"valid": True}

    def test_raw_json_object_among_text(self):
        text = 'Result is {"answer": 42} and that is final.'
        result = extract_json(text)
        assert json.loads(result)["answer"] == 42

    def test_raw_json_array(self):
        text = "Results: [1, 2, 3] done."
        result = extract_json(text)
        assert json.loads(result) == [1, 2, 3]

    def test_invalid_code_block_non_json_start(self):
        text = "```json\nnot json\n```"
        with pytest.raises(SchemaParseError):
            extract_json(text)

    def test_invalid_raw_json(self):
        text = 'The result is {bad json} here.'
        with pytest.raises(SchemaParseError):
            extract_json(text)


# ═══════════════════════════════════════════════════════════════════════════════
# parse_schema with bad JSON (lines 383-384)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseSchemaErrors:

    def test_bad_json_in_code_block(self):
        with pytest.raises(SchemaParseError):
            parse_schema("not json at all!")


# ═══════════════════════════════════════════════════════════════════════════════
# validate_output edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateOutputEdgeCases:

    def test_schema_name_from_dict(self):
        data = {"x": 1}
        result = validate_output(data, {"title": "Custom", "type": "object"})
        assert result == data

    def test_json_string_input(self):
        result = validate_output('{"a": 1}', {"type": "object"})
        assert result == {"a": 1}


# ═══════════════════════════════════════════════════════════════════════════════
# _clean_schema with $defs (line 260)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanSchemaWithDefs:

    def test_inline_defs(self):
        schema = {
            "type": "object",
            "$defs": {"Addr": {"type": "object", "properties": {"city": {"type": "string"}}}},
            "properties": {"address": {"$ref": "#/$defs/Addr"}},
        }
        result = _clean_schema(schema)
        assert "$defs" not in result
        assert result["properties"]["address"]["type"] == "object"


# ═══════════════════════════════════════════════════════════════════════════════
# config.py schema_name fallback (line 90)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigSchemaNameEdge:

    def test_non_type_non_dict_schema(self):
        from nucleusiq.agents.structured_output.config import OutputSchema
        cfg = OutputSchema(schema={"type": "object"})
        assert cfg.schema_name == "Schema"
