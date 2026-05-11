"""Ollama tool wiring — local function calling."""

from __future__ import annotations

from nucleusiq_ollama.tools.converter import to_ollama_function_tool

NATIVE_TOOL_TYPES: frozenset[str] = frozenset()

__all__ = ["NATIVE_TOOL_TYPES", "to_ollama_function_tool"]
