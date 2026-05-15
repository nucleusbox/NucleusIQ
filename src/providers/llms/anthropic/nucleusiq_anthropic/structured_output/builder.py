"""Build Claude ``output_config`` slices for structured outputs (JSON Schema).

Maps Pydantic models, dataclasses, and TypedDict to the Messages API shape::

    output_config = {
        "format": {"type": "json_schema", "schema": {<cleaned JSON Schema>}},
    }

See `Claude structured outputs <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>`_.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any, get_type_hints

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def build_anthropic_output_config(
    schema: type[BaseModel] | type | dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a schema or wire fragment to ``output_config`` for ``messages.create``.

    * **Pydantic / dataclass / TypedDict** → wrapped ``json_schema`` format.
    * **dict** — heuristics:

      - If keys include ``format`` or ``effort``, treated as a full/partial
        ``output_config`` (returned as a shallow copy).
      - If dict is exactly Claude's inner shape ``{\"type\": \"json_schema\", \"schema\": …}``
        returned as ``{\"format\": dict}``.
      - Otherwise assumed to be the **root JSON Schema** object → wrapped under
        ``format.type == \"json_schema\"``.
    """

    if isinstance(schema, dict):
        return _output_config_from_dict(schema)

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        json_schema = schema.model_json_schema()
        cleaned = _clean_schema(json_schema)
        return {"format": {"type": "json_schema", "schema": cleaned}}

    if dataclasses.is_dataclass(schema) and isinstance(schema, type):
        rooted = _dataclass_to_schema(schema)
        return {"format": {"type": "json_schema", "schema": rooted}}

    if hasattr(schema, "__annotations__"):
        rooted = _annotations_to_schema(schema)
        return {"format": {"type": "json_schema", "schema": rooted}}

    logger.warning(
        "Unknown schema type for Anthropic structured output: %s — skipping",
        type(schema),
    )
    return None


def _output_config_from_dict(d: dict[str, Any]) -> dict[str, Any] | None:
    if "format" in d or "effort" in d:
        return dict(d)

    if d.get("type") == "json_schema" and "schema" in d:
        sch = d["schema"]
        if isinstance(sch, dict):
            sch = copy.deepcopy(sch)
        return {"format": {"type": "json_schema", "schema": sch}}

    return {"format": {"type": "json_schema", "schema": copy.deepcopy(d)}}


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs`` / ``definitions`` and strip unsupported noise."""

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
