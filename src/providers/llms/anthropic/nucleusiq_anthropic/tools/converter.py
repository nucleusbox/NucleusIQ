"""Convert framework tool specs → Anthropic Messages API ``tools`` definitions."""

from __future__ import annotations

from typing import Any, cast

# Phase A: empty — server-side tools (web_search, …) land in Phase C.
NATIVE_TOOL_TYPES: frozenset[str] = frozenset()


def to_anthropic_tool_definition(spec: dict[str, Any]) -> dict[str, Any]:
    """Map a generic ``BaseTool.get_spec()`` dict to Anthropic tool shape.

    * **Native-style specs** containing ``input_schema`` and ``name``
      — returned with only Anthropic-supported keys preserved.
    * **OpenAI-style envelopes** ``type: function`` + nested ``function``
      — unwrapped into ``name`` / ``description`` / ``input_schema``.
    """

    tool_type = spec.get("type")
    # Pass through server/native tool payloads (Phase C+ — registry only today).
    if tool_type is not None and tool_type != "function":
        return dict(spec)

    if "input_schema" in spec:
        block: dict[str, Any] = {
            "name": spec["name"],
            "input_schema": spec["input_schema"],
        }
        if spec.get("description"):
            block["description"] = spec["description"]
        return block

    fn = spec.get("function") if isinstance(spec.get("function"), dict) else None
    name = (fn.get("name") if fn else None) or spec.get("name", "")
    description = (fn.get("description") if fn else None) or spec.get(
        "description",
        "",
    )

    raw_params = (fn.get("parameters") if fn else None) or spec.get("parameters")

    parameters: dict[str, Any]
    if isinstance(raw_params, dict):
        parameters = cast(dict[str, Any], dict(raw_params))
        if raw_params.get("type") != "object":
            parameters = {
                "type": "object",
                "properties": {"value": dict(raw_params)},
            }
        if "additionalProperties" not in parameters:
            parameters = {**parameters, "additionalProperties": False}
    else:
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    return {
        "name": name,
        "description": description or "",
        "input_schema": parameters,
    }


def spec_looks_native(spec: dict[str, Any]) -> bool:
    """Whether *spec* is a server-side Claude tool marker (non-function)."""
    t = spec.get("type")
    if t is None or t == "function":
        return False
    if t == "anthropic_builtin":
        return spec.get("name") in NATIVE_TOOL_TYPES
    return t in NATIVE_TOOL_TYPES
