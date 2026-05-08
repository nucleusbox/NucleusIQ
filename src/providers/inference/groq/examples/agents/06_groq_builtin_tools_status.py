"""Groq *built-in* / hosted tools — support status (no API call).

**Phase A (current):** ``nucleusiq-groq`` only declares **local** function tools
(``NATIVE_TOOL_TYPES`` is empty). Chat Completions + your ``@tool`` functions are
what the real agents in ``01``–``05`` exercise.

**Phase B (future):** Groq built-ins (compound models, hosted web search, MCP,
etc.) typically use the **Responses** API or special request shapes — see
``docs/design/GROQ_PROVIDER.md`` §6–8.

This script is intentionally **not** a mock: it documents behavior so you do not
expect OpenAI-style ``OpenAITool`` / web_search routing on Groq yet.

Run::

    uv run python examples/agents/06_groq_builtin_tools_status.py
"""

from __future__ import annotations


def main() -> None:
    print("nucleusiq-groq Phase A: local function calling only.")
    print("See examples/agents/01-05 for real Agent + Groq scenarios.")
    print("Built-in / hosted Groq tools: planned Phase B (design doc).")


if __name__ == "__main__":
    main()
