"""Build Ollama ``format`` payloads from Python types."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, get_type_hints

from pydantic import BaseModel

from nucleusiq_ollama.structured_output.cleaner import clean_schema_for_ollama

logger = logging.getLogger(__name__)


def build_ollama_format(
    schema: type[BaseModel] | type | dict[str, Any],
) -> dict[str, Any] | str | None:
    """Convert *schema* to an Ollama ``chat(.., format=...)`` value.

    Returns a JSON Schema dict, or ``\"json\"`` for unconstrained JSON object mode,
    or ``None`` if unsupported.
    """
    if schema == "json":
        return "json"

    if isinstance(schema, dict):
        if schema.get("type") == "json_schema" and "json_schema" in schema:
            inner = schema["json_schema"]
            sch = inner.get("schema") if isinstance(inner, dict) else None
            if isinstance(sch, dict):
                return clean_schema_for_ollama(sch)
        # OutputSchema.for_provider("ollama" | "groq" | …) uses the generic branch:
        # { "type": "json", "schema": <json schema> } or { "type": "json" } alone.
        if schema.get("type") == "json":
            inner = schema.get("schema")
            if inner is None:
                return "json"
            if isinstance(inner, dict):
                return clean_schema_for_ollama(inner)
        return clean_schema_for_ollama(schema)

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        clean = clean_schema_for_ollama(json_schema)
        return clean

    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        json_schema = _dataclass_to_schema(schema)
        return clean_schema_for_ollama(json_schema)

    if hasattr(schema, "__annotations__"):
        json_schema = _annotations_to_schema(schema)
        return clean_schema_for_ollama(json_schema)

    logger.warning("Unknown schema type: %s, skipping format", type(schema))
    return None


def _dataclass_to_schema(cls: type) -> dict[str, Any]:
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
    hints = get_type_hints(cls)
    properties = {k: _type_to_schema(v) for k, v in hints.items()}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _type_to_schema(tp: Any) -> dict[str, Any]:
    origin = getattr(tp, "__origin__", None)
    if origin is list:
        (item_tp,) = getattr(tp, "__args__", (str,))
        return {"type": "array", "items": _type_to_schema(item_tp)}
    mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    return mapping.get(tp, {"type": "string"})
