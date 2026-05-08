"""Groq tool support — local function calling uses OpenAI-style tool specs.

Groq *hosted* built-in tools (server-side agentic loop) are not wired yet;
see ``docs/design/GROQ_PROVIDER.md`` Phase B.
"""

from __future__ import annotations

from nucleusiq_groq.tools.converter import to_openai_function_tool

NATIVE_TOOL_TYPES: frozenset[str] = frozenset()

__all__ = [
    "NATIVE_TOOL_TYPES",
    "to_openai_function_tool",
]
