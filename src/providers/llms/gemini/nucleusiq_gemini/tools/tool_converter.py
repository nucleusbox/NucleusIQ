"""Convert BaseTool specs to Gemini function declaration format.

**Single Responsibility**: Only handles tool spec conversion — no SDK
calls, no response parsing, no state management.

Gemini uses ``function_declarations`` inside a ``tools`` wrapper::

    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...],
                },
            }
        ]
    }
"""

from __future__ import annotations

from typing import Any


def convert_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a generic BaseTool spec to Gemini function declaration format.

    If the spec already has a ``"type"`` key indicating a native Gemini tool
    (google_search, code_execution, etc.), it is returned as-is.

    Otherwise the spec is transformed into Gemini's
    ``function_declarations`` format.

    Args:
        spec: A tool spec dict from ``BaseTool.get_spec()``.

    Returns:
        Gemini-compatible function declaration dict.
    """
    if "type" in spec:
        return spec

    parameters = _clean_parameters(spec.get("parameters", {}))

    return {
        "name": spec["name"],
        "description": spec.get("description", ""),
        "parameters": parameters,
    }


def build_tools_payload(
    declarations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Wrap function declarations into Gemini's tools payload format.

    Args:
        declarations: List of converted function declarations.

    Returns:
        List of tool dicts ready for the Gemini API ``tools`` parameter.
    """
    native_tools = []
    function_declarations = []

    for decl in declarations:
        if "type" in decl and decl["type"] != "function":
            native_tools.append(decl)
        else:
            function_declarations.append(decl)

    result: list[dict[str, Any]] = []
    if function_declarations:
        result.append({"function_declarations": function_declarations})
    result.extend(native_tools)
    return result


def _clean_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Clean a JSON Schema parameters dict for Gemini compatibility.

    Gemini's function calling uses OpenAPI-compatible JSON Schema but
    doesn't support some JSON Schema keywords like ``additionalProperties``
    and ``$defs``.
    """
    cleaned = dict(params)
    cleaned.pop("additionalProperties", None)
    cleaned.pop("$defs", None)

    if "properties" in cleaned:
        cleaned["properties"] = {
            k: _clean_property(v) for k, v in cleaned["properties"].items()
        }

    return cleaned


def _clean_property(prop: dict[str, Any]) -> dict[str, Any]:
    """Recursively clean a single property schema."""
    cleaned = dict(prop)
    cleaned.pop("additionalProperties", None)
    cleaned.pop("$defs", None)
    cleaned.pop("title", None)
    cleaned.pop("default", None)

    if "anyOf" in cleaned:
        non_null = [s for s in cleaned["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            base = _clean_property(non_null[0])
            cleaned = {**base, "nullable": True}
        else:
            cleaned["anyOf"] = [_clean_property(s) for s in cleaned["anyOf"]]

    if "items" in cleaned and isinstance(cleaned["items"], dict):
        cleaned["items"] = _clean_property(cleaned["items"])

    if "properties" in cleaned and isinstance(cleaned["properties"], dict):
        cleaned["properties"] = {
            k: _clean_property(v) for k, v in cleaned["properties"].items()
        }

    return cleaned
