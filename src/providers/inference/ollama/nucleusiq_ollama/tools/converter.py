"""Convert framework tool specs to Ollama function tool format."""

from __future__ import annotations

from typing import Any


def to_ollama_function_tool(spec: dict[str, Any]) -> dict[str, Any]:
    """Wrap a generic BaseTool ``get_spec()`` dict as an OpenAI-style function tool."""
    if spec.get("type") == "function" and "function" in spec:
        return spec
    name = spec.get("name", "")
    description = spec.get("description", "")
    parameters = spec.get("parameters") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
