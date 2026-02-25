"""Build OpenAI ``response_format`` from Python types.

Supports:
- Pydantic ``BaseModel`` subclasses (recommended)
- Python ``dataclasses``
- ``TypedDict`` and annotated classes
- Raw ``dict`` pass-through (e.g. ``{"type": "json_object"}``)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, get_type_hints

from pydantic import BaseModel

from nucleusiq_openai.structured_output.cleaner import clean_schema_for_openai

logger = logging.getLogger(__name__)


def build_response_format(
    schema: type[BaseModel] | type | dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a schema type to OpenAI's ``response_format`` parameter.

    Args:
        schema: A Pydantic model class, dataclass, TypedDict, or raw dict.

    Returns:
        OpenAI-compatible ``response_format`` dict, or ``None`` if the
        schema type is not recognised.
    """
    if isinstance(schema, dict):
        return schema

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        clean = clean_schema_for_openai(json_schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": clean,
            },
        }

    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        json_schema = _dataclass_to_schema(schema)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": json_schema,
            },
        }

    if hasattr(schema, "__annotations__"):
        json_schema = _annotations_to_schema(schema)
        name = getattr(schema, "__name__", "response")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": json_schema,
            },
        }

    logger.warning("Unknown schema type: %s, skipping response_format", type(schema))
    return None


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #


def _dataclass_to_schema(cls: type) -> dict[str, Any]:
    """Convert a dataclass to JSON Schema."""
    hints = get_type_hints(cls)
    fields = dataclasses.fields(cls)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in fields:
        prop_schema = _type_to_schema(hints.get(field.name, str))
        properties[field.name] = prop_schema
        if (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        ):
            required.append(field.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _annotations_to_schema(cls: type) -> dict[str, Any]:
    """Convert type annotations (e.g. TypedDict) to JSON Schema."""
    hints = get_type_hints(cls)
    properties: dict[str, Any] = {}
    required = list(hints.keys())

    for name, hint in hints.items():
        properties[name] = _type_to_schema(hint)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _type_to_schema(type_hint: type) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    from typing import Union as UnionType
    from typing import get_args, get_origin

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = _type_to_schema(non_none[0])
            return {"anyOf": [inner, {"type": "null"}]}

    if origin is list:
        item_schema = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    return type_map.get(type_hint, {"type": "string"})
