"""Classify and split mixed tool lists for the proxy pattern.

When an agent has both native Gemini tools (``google_search``,
``code_execution``, etc.) **and** custom ``@tool`` functions, the
``generateContent`` API rejects the request with a 400 error.

This module detects the conflict and provides proxy function-declaration
specs so native tools can be presented to the LLM as regular callable
functions.  The actual native tool execution is handled by
``_GeminiNativeTool._proxy_execute()``.

The proxy pattern is transparent to the core framework — it requires
zero changes to ``Agent``, ``Executor``, ``BaseExecutionMode``, or any
execution mode.  It works because:

1. ``BaseGemini.convert_tool_specs()`` detects mixed tools and calls
   ``_enable_proxy_mode()`` on each native tool.
2. The native tool's ``is_native`` flag is flipped to ``False`` so the
   core ``Executor`` will call ``execute()`` instead of rejecting it.
3. ``execute()`` makes a separate ``generate_content`` sub-call with
   the *real* native tool to get grounded results.

This workaround is superseded by the Gemini Interactions API (v0.8.0)
which natively supports mixing native and custom tools.
"""

from __future__ import annotations

from typing import Any

PROXY_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "google_search": {
        "description": (
            "Search the web for current, real-time information. "
            "Use when you need up-to-date facts, news, prices, or data "
            "that may not be in your training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web",
                },
            },
            "required": ["query"],
        },
    },
    "code_execution": {
        "description": (
            "Execute Python code in a secure sandbox. "
            "Use for calculations, data processing, or analysis that "
            "requires running actual code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": ("Description of what code to write and execute"),
                },
            },
            "required": ["task"],
        },
    },
    "url_context": {
        "description": (
            "Fetch and understand web page content from a URL. "
            "Use when you need to read or analyse a specific web page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and analyse",
                },
            },
            "required": ["url"],
        },
    },
    "google_maps": {
        "description": (
            "Look up location information using Google Maps. "
            "Use for places, directions, distances, or geographic data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The location or directions query",
                },
            },
            "required": ["query"],
        },
    },
}


def _is_gemini_native(tool: Any) -> bool:
    """Check if *tool* is a Gemini native tool (even if currently proxied).

    Uses ``tool_type`` attribute which is immutable and unique to
    ``_GeminiNativeTool``, unlike ``is_native`` which is mutated by
    proxy mode.
    """
    return hasattr(tool, "tool_type") and hasattr(tool, "_enable_proxy_mode")


def has_mixed_tools(tools: list[Any]) -> bool:
    """Return ``True`` if *tools* contain both native and custom tools.

    Early-exits as soon as both flags are set.
    """
    has_native = False
    has_custom = False
    for tool in tools:
        if _is_gemini_native(tool):
            has_native = True
        else:
            has_custom = True
        if has_native and has_custom:
            return True
    return False


def classify_tools(
    tools: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Split *tools* into ``(native, custom)`` lists."""
    native: list[Any] = []
    custom: list[Any] = []
    for tool in tools:
        if _is_gemini_native(tool):
            native.append(tool)
        else:
            custom.append(tool)
    return native, custom


def build_proxy_spec(tool_type: str, tool_name: str) -> dict[str, Any]:
    """Build a function-declaration spec that proxies a native tool.

    Returns a dict in the same shape as ``BaseTool.get_spec()`` so
    it flows through ``convert_tool_spec()`` like any custom tool.
    """
    meta = PROXY_DESCRIPTIONS.get(tool_type)
    if meta is not None:
        return {
            "name": tool_name,
            "description": meta["description"],
            "parameters": meta["parameters"],
        }
    return {
        "name": tool_name,
        "description": f"Gemini native tool: {tool_type}",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Input for {tool_type}",
                },
            },
            "required": ["query"],
        },
    }
