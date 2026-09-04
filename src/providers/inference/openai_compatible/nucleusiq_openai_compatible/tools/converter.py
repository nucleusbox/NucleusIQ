"""Tool-spec conversion to the OpenAI function-tool shape.

Every tool on this provider is a **local function tool**: there is no
server-side tool execution on a self-hosted inference server, so
``NATIVE_TOOL_TYPES`` is empty and every ``ToolCallRecord`` is
``executed_by="local"``.

Some servers validate tool schemas strictly, so JSON Schema metadata keys
that carry no validation semantics are stripped — the same pragmatism as
``nucleusiq_mcp.schema_adapter``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["STRIPPED_SCHEMA_KEYS", "convert_tool_spec", "convert_tool_specs"]

STRIPPED_SCHEMA_KEYS: frozenset[str] = frozenset(
    {"$schema", "$id", "title", "definitions", "$defs", "examples", "default"}
)
"""Schema keys that confuse strict validators without affecting semantics."""


def _clean_schema(schema: Any) -> Any:
    """Recursively drop non-semantic metadata keys from a JSON schema."""
    if isinstance(schema, dict):
        return {
            k: _clean_schema(v)
            for k, v in schema.items()
            if k not in STRIPPED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


def convert_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert one generic tool spec into an OpenAI function tool.

    Accepts either an already-wrapped ``{"type": "function", "function":
    {...}}`` spec or a bare ``{"name", "description", "parameters"}`` spec,
    since both shapes appear across NucleusIQ tool implementations.

    Args:
        spec: The tool specification.

    Returns:
        A spec in ``{"type": "function", "function": {...}}`` form.

    Raises:
        ValueError: The spec carries no usable function name.
    """
    if spec.get("type") == "function" and isinstance(spec.get("function"), dict):
        fn = dict(spec["function"])
    else:
        fn = {
            k: v
            for k, v in spec.items()
            if k in ("name", "description", "parameters", "strict")
        }

    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Tool spec is missing a function name: {spec!r}. Every tool must "
            "expose a non-empty 'name'."
        )

    function: dict[str, Any] = {"name": name.strip()}
    description = fn.get("description")
    if isinstance(description, str) and description.strip():
        function["description"] = description.strip()

    parameters = fn.get("parameters")
    if isinstance(parameters, dict) and parameters:
        function["parameters"] = _clean_schema(parameters)
    else:
        # Servers that validate strictly reject a function with no schema.
        function["parameters"] = {"type": "object", "properties": {}}

    if fn.get("strict") is True:
        function["strict"] = True

    return {"type": "function", "function": function}


def convert_tool_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a list of tool specs, preserving order."""
    return [convert_tool_spec(s) for s in specs]
