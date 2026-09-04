"""Tool conversion for OpenAI-compatible servers.

``NATIVE_TOOL_TYPES`` is empty: a self-hosted inference server executes no
tools server-side, so every tool is a local function tool.
"""

from .converter import convert_tool_spec, convert_tool_specs

NATIVE_TOOL_TYPES: frozenset = frozenset()

__all__ = ["NATIVE_TOOL_TYPES", "convert_tool_spec", "convert_tool_specs"]
