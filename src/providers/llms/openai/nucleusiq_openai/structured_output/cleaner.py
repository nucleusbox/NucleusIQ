"""Clean and prepare JSON schemas for OpenAI's strict structured output mode.

OpenAI requires:
- ``additionalProperties: false`` on all object schemas
- All ``$ref`` pointers inlined (no ``$defs`` block)
- No Pydantic metadata (``title``, ``$schema``, ``description``)
- All properties listed in ``required``
"""

from __future__ import annotations

import copy
from typing import Any


def clean_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean a Pydantic-generated JSON schema for OpenAI structured outputs.

    Args:
        schema: Raw JSON schema (e.g. from ``Model.model_json_schema()``).

    Returns:
        Cleaned schema suitable for OpenAI's ``response_format``.
    """
    schema = copy.deepcopy(schema)

    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    defs = schema.pop("$defs", {})
    if defs:
        schema = _inline_refs(schema, defs)

    schema.pop("title", None)
    schema.pop("$schema", None)
    schema.pop("description", None)

    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
        for key, prop in schema["properties"].items():
            if isinstance(prop, dict):
                schema["properties"][key] = _clean_property(prop, defs)

    return schema


def _clean_property(prop: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Clean a property schema recursively."""
    prop = copy.deepcopy(prop)
    for key in (
        "title",
        "default",
        "description",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "ge",
        "le",
    ):
        prop.pop(key, None)

    if "anyOf" in prop:
        prop["anyOf"] = [_clean_property(opt, defs) for opt in prop["anyOf"]]

    if prop.get("type") == "object" and "properties" in prop:
        prop["additionalProperties"] = False
        prop["required"] = list(prop["properties"].keys())
        for key, nested in prop["properties"].items():
            if isinstance(nested, dict):
                prop["properties"][key] = _clean_property(nested, defs)

    if prop.get("type") == "array" and "items" in prop:
        prop["items"] = _clean_property(prop["items"], defs)

    return prop


def _inline_refs(obj: Any, defs: dict[str, Any]) -> Any:
    """Recursively inline ``$ref`` references."""
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_path = obj["$ref"]
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                if def_name in defs:
                    return _inline_refs(
                        clean_schema_for_openai(defs[def_name]),
                        defs,
                    )
        return {k: _inline_refs(v, defs) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_inline_refs(item, defs) for item in obj]
    return obj
