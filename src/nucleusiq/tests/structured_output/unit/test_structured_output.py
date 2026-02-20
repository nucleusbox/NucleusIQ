"""Tests for the structured_output package: types, errors, config, resolver, parser."""

import json
import dataclasses
import pytest
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from nucleusiq.agents.structured_output.types import (
    OutputMode, ErrorHandling, SchemaType,
)
from nucleusiq.agents.structured_output.errors import (
    StructuredOutputError,
    SchemaValidationError,
    SchemaParseError,
    MultipleOutputError,
)
from nucleusiq.agents.structured_output.config import OutputSchema, output_schema
from nucleusiq.agents.structured_output.resolver import (
    supports_native_output,
    resolve_output_config,
    get_provider_from_llm,
    _auto_select_mode,
)
from nucleusiq.agents.structured_output.parser import (
    is_pydantic,
    is_dataclass,
    is_json_schema,
    schema_to_json,
    extract_json,
    parse_schema,
    validate_output,
    _type_to_json,
    _clean_schema,
    _clean_property,
    _inline_refs,
)


# ── Fixture models ───────────────────────────────────────────────────────────

class PersonModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


@dataclasses.dataclass
class PointDC:
    x: float
    y: float
    label: str = "origin"


# ═══════════════════════════════════════════════════════════════════════════════
# types.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputMode:

    def test_all_values(self):
        assert set(m.value for m in OutputMode) == {"auto", "native", "tool", "prompt"}

    def test_implemented_modes(self):
        impl = OutputMode.implemented_modes()
        assert OutputMode.AUTO in impl
        assert OutputMode.NATIVE in impl
        assert OutputMode.TOOL not in impl

    def test_is_implemented(self):
        assert OutputMode.is_implemented(OutputMode.NATIVE) is True
        assert OutputMode.is_implemented(OutputMode.TOOL) is False

    def test_validate_mode_ok(self):
        OutputMode.validate_mode(OutputMode.NATIVE)

    def test_validate_mode_not_impl(self):
        with pytest.raises(NotImplementedError, match="TOOL"):
            OutputMode.validate_mode(OutputMode.TOOL)


class TestErrorHandling:

    def test_values(self):
        assert set(e.value for e in ErrorHandling) == {"retry", "raise", "fallback"}


# ═══════════════════════════════════════════════════════════════════════════════
# errors.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructuredOutputError:

    def test_basic(self):
        e = StructuredOutputError("bad")
        assert str(e) == "bad"
        assert e.retryable is True
        assert e.raw_output is None

    def test_format_for_retry(self):
        e = StructuredOutputError("oops", schema_name="X")
        assert "oops" in e.format_for_retry()

    def test_non_retryable(self):
        e = StructuredOutputError("fatal", retryable=False)
        assert e.retryable is False


class TestSchemaValidationError:

    def test_with_field_errors(self):
        errs = [{"field": "age", "error": "not an int"}]
        e = SchemaValidationError("bad", schema_name="Person", field_errors=errs)
        retry_msg = e.format_for_retry()
        assert "age" in retry_msg
        assert "Person" in retry_msg

    def test_without_field_errors(self):
        e = SchemaValidationError("bad", schema_name="X")
        assert "bad" in e.format_for_retry()


class TestSchemaParseError:

    def test_basic(self):
        e = SchemaParseError("bad json", raw_output="{", parse_position=1)
        assert e.parse_position == 1
        assert "JSON" in e.format_for_retry()


class TestMultipleOutputError:

    def test_basic(self):
        e = MultipleOutputError(
            "too many", output_names=["A", "B"], outputs=[1, 2]
        )
        retry = e.format_for_retry()
        assert "2" in retry
        assert "A, B" in retry

    def test_empty_names(self):
        e = MultipleOutputError("x")
        assert "multiple" in e.format_for_retry()


# ═══════════════════════════════════════════════════════════════════════════════
# config.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputSchema:

    def test_defaults(self):
        cfg = OutputSchema(schema=PersonModel)
        assert cfg.mode == OutputMode.AUTO
        assert cfg.strict is True
        assert cfg.retry_on_error is True
        assert cfg.max_retries == 2

    def test_schema_name_pydantic(self):
        cfg = OutputSchema(schema=PersonModel)
        assert cfg.schema_name == "PersonModel"

    def test_schema_name_dict_title(self):
        cfg = OutputSchema(schema={"title": "MyThing", "type": "object"})
        assert cfg.schema_name == "MyThing"

    def test_schema_name_dict_name(self):
        cfg = OutputSchema(schema={"name": "Foo", "type": "object"})
        assert cfg.schema_name == "Foo"

    def test_schema_name_fallback(self):
        cfg = OutputSchema(schema={"type": "object"})
        assert cfg.schema_name == "Schema"

    def test_max_retries_negative(self):
        with pytest.raises(ValueError, match="max_retries"):
            OutputSchema(schema=PersonModel, max_retries=-1)

    def test_validate_unimplemented_mode(self):
        with pytest.raises(NotImplementedError):
            OutputSchema(schema=PersonModel, mode=OutputMode.TOOL)

    def test_get_error_message_custom_handler(self):
        cfg = OutputSchema(
            schema=PersonModel,
            error_handler=lambda e: f"CUSTOM: {e}",
        )
        assert "CUSTOM" in cfg.get_error_message(ValueError("x"))

    def test_get_error_message_format_for_retry(self):
        err = SchemaValidationError("bad", schema_name="S")
        cfg = OutputSchema(schema=PersonModel)
        msg = cfg.get_error_message(err)
        assert "bad" in msg

    def test_get_error_message_plain(self):
        cfg = OutputSchema(schema=PersonModel)
        msg = cfg.get_error_message(RuntimeError("boom"))
        assert "boom" in msg

    def test_should_retry_true(self):
        cfg = OutputSchema(schema=PersonModel, retry_on_error=True, max_retries=3)
        assert cfg.should_retry(ValueError("x"), attempt=0) is True

    def test_should_retry_disabled(self):
        cfg = OutputSchema(schema=PersonModel, retry_on_error=False)
        assert cfg.should_retry(ValueError("x"), attempt=0) is False

    def test_should_retry_max_reached(self):
        cfg = OutputSchema(schema=PersonModel, max_retries=2)
        assert cfg.should_retry(ValueError("x"), attempt=2) is False

    def test_should_retry_non_retryable(self):
        cfg = OutputSchema(schema=PersonModel)
        err = StructuredOutputError("fatal", retryable=False)
        assert cfg.should_retry(err, attempt=0) is False

    def test_for_provider_openai(self):
        cfg = OutputSchema(schema=PersonModel)
        result = cfg.for_provider("openai")
        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["strict"] is True

    def test_for_provider_generic(self):
        cfg = OutputSchema(schema=PersonModel)
        result = cfg.for_provider("anthropic")
        assert result["type"] == "json"
        assert "schema" in result


class TestOutputSchemaFunction:

    def test_convenience(self):
        cfg = output_schema(PersonModel, strict=False, max_retries=5)
        assert isinstance(cfg, OutputSchema)
        assert cfg.strict is False
        assert cfg.max_retries == 5


# ═══════════════════════════════════════════════════════════════════════════════
# resolver.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupportsNativeOutput:

    def test_openai_models(self):
        assert supports_native_output("gpt-4o-mini") is True
        assert supports_native_output("gpt-5-turbo") is True

    def test_anthropic(self):
        assert supports_native_output("claude-3-opus") is True

    def test_unknown(self):
        assert supports_native_output("llama-3") is False

    def test_with_provider_filter(self):
        assert supports_native_output("gpt-4o", provider="openai") is True
        assert supports_native_output("gpt-4o", provider="anthropic") is False


class TestResolveOutputConfig:

    def test_none(self):
        assert resolve_output_config(None) is None

    def test_output_schema(self):
        cfg = OutputSchema(schema=PersonModel)
        result = resolve_output_config(cfg, model_name="gpt-4o")
        assert result is cfg
        assert result._resolved_mode == OutputMode.NATIVE

    def test_pydantic_type(self):
        result = resolve_output_config(PersonModel, model_name="gpt-4o")
        assert isinstance(result, OutputSchema)

    def test_dict_schema(self):
        result = resolve_output_config(
            {"type": "object", "properties": {}}, model_name="gpt-4o"
        )
        assert isinstance(result, OutputSchema)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid response_format"):
            resolve_output_config(42)

    def test_explicit_mode_not_auto(self):
        cfg = OutputSchema(schema=PersonModel, mode=OutputMode.NATIVE)
        result = resolve_output_config(cfg, model_name="gpt-4o")
        assert result._resolved_mode == OutputMode.NATIVE


class TestAutoSelectMode:

    def test_native_model(self):
        assert _auto_select_mode(model_name="gpt-4o") == OutputMode.NATIVE

    def test_unknown_model(self):
        assert _auto_select_mode(model_name="llama3") == OutputMode.NATIVE

    def test_no_model(self):
        assert _auto_select_mode() == OutputMode.PROMPT


class TestGetProviderFromLLM:

    def test_none(self):
        assert get_provider_from_llm(None) is None

    def test_openai(self):
        class FakeOpenAILLM:
            pass
        assert get_provider_from_llm(FakeOpenAILLM()) == "openai"

    def test_anthropic(self):
        class FakeAnthropicLLM:
            pass
        assert get_provider_from_llm(FakeAnthropicLLM()) == "anthropic"

    def test_claude(self):
        class FakeClaudeLLM:
            pass
        assert get_provider_from_llm(FakeClaudeLLM()) == "anthropic"

    def test_google(self):
        class FakeGoogleLLM:
            pass
        assert get_provider_from_llm(FakeGoogleLLM()) == "google"

    def test_unknown(self):
        class FakeLlamaCpp:
            pass
        assert get_provider_from_llm(FakeLlamaCpp()) is None


# ═══════════════════════════════════════════════════════════════════════════════
# parser.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestTypeDetection:

    def test_is_pydantic(self):
        assert is_pydantic(PersonModel) is True
        assert is_pydantic(dict) is False
        assert is_pydantic("string") is False

    def test_is_dataclass(self):
        assert is_dataclass(PointDC) is True
        assert is_dataclass(PersonModel) is False

    def test_is_json_schema(self):
        assert is_json_schema({"type": "object"}) is True
        assert is_json_schema({"foo": "bar"}) is False
        assert is_json_schema("nope") is False


class TestSchemaToJson:

    def test_pydantic(self):
        result = schema_to_json(PersonModel)
        assert result["type"] == "object"
        assert "name" in result.get("properties", {})

    def test_dataclass(self):
        result = schema_to_json(PointDC)
        assert result["type"] == "object"
        assert "x" in result.get("properties", {})

    def test_dict_passthrough(self):
        raw = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = schema_to_json(raw)
        assert result["type"] == "object"

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            schema_to_json(42)

    def test_strict_mode(self):
        result = schema_to_json(PersonModel, strict=True)
        assert result.get("additionalProperties") is False

    def test_for_openai(self):
        result = schema_to_json(PersonModel, for_provider="openai")
        assert result.get("additionalProperties") is False
        assert "required" in result

    def test_annotations_class(self):
        class Simple:
            name: str
            value: int
        result = schema_to_json(Simple)
        assert "name" in result["properties"]


class TestTypeToJson:

    def test_none_type(self):
        assert _type_to_json(type(None)) == {"type": "null"}

    def test_optional(self):
        result = _type_to_json(Optional[str])
        assert "anyOf" in result

    def test_list(self):
        result = _type_to_json(List[int])
        assert result["type"] == "array"

    def test_dict(self):
        result = _type_to_json(Dict[str, Any])
        assert result["type"] == "object"

    def test_basic_types(self):
        assert _type_to_json(str)["type"] == "string"
        assert _type_to_json(int)["type"] == "integer"
        assert _type_to_json(float)["type"] == "number"
        assert _type_to_json(bool)["type"] == "boolean"

    def test_nested_pydantic(self):
        result = _type_to_json(PersonModel)
        assert "properties" in result

    def test_nested_dataclass(self):
        result = _type_to_json(PointDC)
        assert result["type"] == "object"


class TestCleanProperty:

    def test_removes_metadata(self):
        prop = {"type": "string", "title": "Name", "default": "x", "description": "d"}
        result = _clean_property(prop)
        assert "title" not in result
        assert "default" not in result

    def test_nested_object(self):
        prop = {
            "type": "object",
            "properties": {"x": {"type": "string", "title": "X"}},
        }
        result = _clean_property(prop, for_provider="openai")
        assert result.get("additionalProperties") is False

    def test_anyof(self):
        prop = {"anyOf": [{"type": "string", "title": "S"}, {"type": "null"}]}
        result = _clean_property(prop)
        assert "title" not in result["anyOf"][0]

    def test_array_items(self):
        prop = {"type": "array", "items": {"type": "string", "title": "I"}}
        result = _clean_property(prop)
        assert "title" not in result["items"]


class TestInlineRefs:

    def test_resolves_refs(self):
        defs = {"Address": {"type": "object", "properties": {"city": {"type": "string"}}}}
        obj = {"type": "object", "properties": {"addr": {"$ref": "#/$defs/Address"}}}
        result = _inline_refs(obj, defs)
        assert result["properties"]["addr"]["type"] == "object"

    def test_list_handling(self):
        result = _inline_refs([1, {"$ref": "#/$defs/X"}], {"X": {"type": "string"}})
        assert result[1]["type"] == "string"


class TestExtractJson:

    def test_code_block(self):
        text = '```json\n{"a": 1}\n```'
        assert json.loads(extract_json(text)) == {"a": 1}

    def test_raw_json(self):
        text = 'Here is the result: {"b": 2} done.'
        assert json.loads(extract_json(text)) == {"b": 2}

    def test_plain_json(self):
        assert json.loads(extract_json('{"c": 3}')) == {"c": 3}

    def test_no_json(self):
        with pytest.raises(SchemaParseError):
            extract_json("no json here")

    def test_array(self):
        text = "[1, 2, 3]"
        assert json.loads(extract_json(text)) == [1, 2, 3]


class TestParseSchema:

    def test_valid(self):
        result = parse_schema('{"x": 1}')
        assert result == {"x": 1}

    def test_code_block(self):
        result = parse_schema('```json\n{"y": 2}\n```')
        assert result == {"y": 2}


class TestValidateOutput:

    def test_pydantic_valid(self):
        result = validate_output({"name": "Alice", "age": 30}, PersonModel)
        assert isinstance(result, PersonModel)
        assert result.name == "Alice"

    def test_pydantic_invalid(self):
        with pytest.raises(SchemaValidationError, match="age"):
            validate_output({"name": "Bob"}, PersonModel)

    def test_dataclass_valid(self):
        result = validate_output({"x": 1.0, "y": 2.0}, PointDC)
        assert isinstance(result, PointDC)

    def test_dataclass_invalid(self):
        with pytest.raises(SchemaValidationError):
            validate_output({"x": 1.0}, PointDC)

    def test_dict_schema_passthrough(self):
        result = validate_output({"a": 1}, {"type": "object"})
        assert result == {"a": 1}

    def test_string_input(self):
        result = validate_output('{"name": "Eve", "age": 25}', PersonModel)
        assert isinstance(result, PersonModel)

    def test_general_exception(self):
        class BrokenModel(BaseModel):
            x: int

            def __init__(self, **data):
                raise RuntimeError("boom")

        with pytest.raises(SchemaValidationError, match="boom"):
            validate_output({"x": 1}, BrokenModel)
