"""Build Gemini generation config for structured output from Python types.

Gemini uses ``response_mime_type`` and ``response_json_schema`` instead of
OpenAI's ``response_format``.  This module converts Pydantic models,
dataclasses, and TypedDict into the Gemini-native format.

Supports:
- Pydantic ``BaseModel`` subclasses (recommended)
- Python ``dataclasses``
- ``TypedDict`` and annotated classes
- Raw ``dict`` pass-through
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, get_type_hints

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def build_gemini_response_format(
    schema: type[BaseModel] | type | dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a schema type to Gemini's structured output config.

    Args:
        schema: A Pydantic model class, dataclass, TypedDict, or raw dict.

    Returns:
        Dict with ``response_mime_type`` and ``response_json_schema``,
        or ``None`` if the schema type is not recognised.
    """
    if isinstance(schema, dict):
        return schema

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        cleaned = _clean_schema(json_schema)
        return {
            "response_mime_type": "application/json",
            "response_json_schema": cleaned,
        }

    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        json_schema = _dataclass_to_schema(schema)
        return {
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
        }

    if hasattr(schema, "__annotations__"):
        json_schema = _annotations_to_schema(schema)
        return {
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
        }

    logger.warning("Unknown schema type: %s, skipping structured output", type(schema))
    return None


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean a Pydantic JSON schema for Gemini compatibility.

    Gemini doesn't support some JSON Schema keywords that Pydantic emits.
    ``$ref`` pointers are inlined (similar to OpenAI's cleaner) so nested
    Pydantic models work correctly.
    """
    import copy

    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})
    defs.update(schema.pop("definitions", {}))

    if defs:
        schema = _inline_refs(schema, defs)

    schema.pop("title", None)
    schema.pop("additionalProperties", None)

    if "properties" in schema:
        schema["properties"] = {
            k: _clean_property(v) for k, v in schema["properties"].items()
        }

    return schema


def _inline_refs(obj: Any, defs: dict[str, Any]) -> Any:
    """Recursively inline ``$ref`` references using the ``$defs`` mapping."""
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_path = obj["$ref"]
            for prefix in ("#/$defs/", "#/definitions/"):
                if ref_path.startswith(prefix):
                    def_name = ref_path[len(prefix) :]
                    if def_name in defs:
                        import copy

                        resolved = copy.deepcopy(defs[def_name])
                        return _inline_refs(_clean_schema_inner(resolved), defs)
            return obj
        return {k: _inline_refs(v, defs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_inline_refs(item, defs) for item in obj]
    return obj


def _clean_schema_inner(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean an inlined sub-schema (no $defs extraction needed)."""
    schema.pop("title", None)
    schema.pop("additionalProperties", None)
    schema.pop("$defs", None)
    schema.pop("definitions", None)

    if "properties" in schema:
        schema["properties"] = {
            k: _clean_property(v) for k, v in schema["properties"].items()
        }
    return schema


def _clean_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Recursively clean a single property schema for Gemini."""
    cleaned = dict(prop)
    cleaned.pop("title", None)
    cleaned.pop("default", None)
    cleaned.pop("additionalProperties", None)
    cleaned.pop("$defs", None)

    if "anyOf" in cleaned:
        non_null = [s for s in cleaned["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            cleaned = _clean_property(non_null[0])
            cleaned["nullable"] = True
        else:
            cleaned["anyOf"] = [_clean_property(s) for s in cleaned["anyOf"]]

    if "items" in cleaned and isinstance(cleaned["items"], dict):
        cleaned["items"] = _clean_property(cleaned["items"])

    if "properties" in cleaned:
        cleaned["properties"] = {
            k: _clean_property(v) for k, v in cleaned["properties"].items()
        }

    return cleaned


def _dataclass_to_schema(cls: type) -> dict[str, Any]:
    """Convert a dataclass to JSON Schema."""
    hints = get_type_hints(cls)
    fields = dataclasses.fields(cls)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in fields:
        properties[field.name] = _type_to_schema(hints.get(field.name, str))
        if (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        ):
            required.append(field.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
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
            return {**inner, "nullable": True}

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
