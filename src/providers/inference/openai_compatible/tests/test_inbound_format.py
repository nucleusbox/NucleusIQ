"""Interpreting a ``response_format`` kwarg handed over by the core agent."""

from __future__ import annotations

import pytest
from nucleusiq_openai_compatible.structured_output import (
    InboundFormat,
    normalize_response_format,
)
from pydantic import BaseModel

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class Person(BaseModel):
    name: str
    age: int


class TestEmpty:
    @pytest.mark.parametrize("value", [None, {}, "", 0])
    def test_nothing_to_do(self, value: object) -> None:
        assert normalize_response_format(value) is None


class TestOpenAIWireForms:
    def test_json_schema_block(self) -> None:
        result = normalize_response_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "Answer", "schema": SCHEMA, "strict": True},
            }
        )
        assert result.schema == SCHEMA
        assert result.name == "Answer"

    def test_json_schema_without_a_name(self) -> None:
        result = normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": SCHEMA}}
        )
        assert result.name == "response"

    def test_malformed_json_schema_block_is_forwarded(self) -> None:
        value = {"type": "json_schema", "json_schema": "oops"}
        result = normalize_response_format(value)
        assert result.schema is None
        assert result.passthrough == value

    def test_json_object_is_passthrough(self) -> None:
        result = normalize_response_format({"type": "json_object"})
        assert result.schema is None
        assert result.passthrough == {"type": "json_object"}


class TestNucleusIQGenericForm:
    def test_type_json_is_unwrapped(self) -> None:
        # for_provider() emits this for every non-OpenAI provider, and no
        # OpenAI-compatible server understands type="json".
        result = normalize_response_format({"type": "json", "schema": SCHEMA})
        assert result.schema == SCHEMA
        assert result.passthrough is None

    def test_type_json_without_schema_is_dropped(self) -> None:
        assert normalize_response_format({"type": "json"}) is None


class TestBareSchema:
    def test_object_schema(self) -> None:
        assert normalize_response_format(SCHEMA).schema == SCHEMA

    def test_title_becomes_the_name(self) -> None:
        result = normalize_response_format({**SCHEMA, "title": "Answer"})
        assert result.name == "Answer"

    def test_ref_schema(self) -> None:
        value = {"$ref": "#/$defs/Person"}
        assert normalize_response_format(value).schema == value

    def test_anyof_schema(self) -> None:
        value = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert normalize_response_format(value).schema == value

    def test_unrecognized_dict_is_forwarded_untouched(self) -> None:
        value = {"type": "text"}
        assert normalize_response_format(value).passthrough == value


class TestPydanticModel:
    def test_model_class(self) -> None:
        result = normalize_response_format(Person)
        assert result.schema["properties"].keys() == {"name", "age"}
        assert result.name == "Person"

    def test_model_whose_schema_generation_fails(self) -> None:
        class Broken:
            __name__ = "Broken"

            @staticmethod
            def model_json_schema():
                raise RuntimeError("cannot build schema")

        assert normalize_response_format(Broken) is None

    def test_model_returning_a_non_dict(self) -> None:
        class Odd:
            @staticmethod
            def model_json_schema():
                return "not a dict"

        assert normalize_response_format(Odd) is None


class TestHandlerTuple:
    """``get_call_kwargs`` emits ``(provider_format, schema)`` for OutputSchema."""

    def test_openai_shaped_pair(self) -> None:
        provider_format = {
            "type": "json_schema",
            "json_schema": {"name": "Answer", "schema": SCHEMA},
        }
        result = normalize_response_format((provider_format, SCHEMA))
        assert result.schema == SCHEMA

    def test_generic_shaped_pair(self) -> None:
        result = normalize_response_format(({"type": "json", "schema": SCHEMA}, Person))
        assert result.schema is not None, (
            "whichever element carries the schema must win over a passthrough"
        )

    def test_pair_with_a_pydantic_model(self) -> None:
        result = normalize_response_format(({"type": "json_object"}, Person))
        assert result.schema["properties"].keys() == {"name", "age"}

    def test_pair_with_nothing_usable(self) -> None:
        assert normalize_response_format((None, None)) is None

    def test_pair_falling_back_to_passthrough(self) -> None:
        result = normalize_response_format(({"type": "json_object"}, None))
        assert result.passthrough == {"type": "json_object"}


class TestRepr:
    def test_schema_repr(self) -> None:
        assert "schema=" in repr(InboundFormat(schema=SCHEMA, name="A"))

    def test_passthrough_repr(self) -> None:
        assert "passthrough=" in repr(
            InboundFormat(passthrough={"type": "json_object"})
        )
