"""Groq *built-in* / hosted tools — support status (no API call).

**Phase A (current):** ``nucleusiq-groq`` wires **local** function tools only
(``NATIVE_TOOL_TYPES`` is empty). Chat Completions + your ``@tool`` functions are
what the real agents in ``01``–``05`` exercise.

**Hosted tools** (web search, code execution, … on Groq’s side) are documented
by Groq and mirrored in ``nucleusiq_groq.tools`` as identifier frozensets — not
passed through ``BaseGroq`` yet. See ``docs/design/GROQ_PROVIDER.md`` §8.1.

**Phase B:** pass-through (e.g. ``compound_custom``) or Responses API — see
design doc.

Run::

    uv run python examples/agents/06_groq_builtin_tools_status.py
"""

from __future__ import annotations

from nucleusiq_groq.tools import (
    GROQ_COMPOUND_HOSTED_TOOL_IDS,
    GROQ_GPT_OSS_HOSTED_TOOL_IDS,
    NATIVE_TOOL_TYPES,
)


def main() -> None:
    print("nucleusiq-groq Phase A (beta): local function calling is wired.")
    print("  NATIVE_TOOL_TYPES (agent-routed hosted types):", sorted(NATIVE_TOOL_TYPES))
    print()
    print(
        "Groq documented hosted tool IDs (reference only; not auto-sent by BaseGroq):"
    )
    print("  groq/compound*  ->", sorted(GROQ_COMPOUND_HOSTED_TOOL_IDS))
    print("  openai/gpt-oss-* ->", sorted(GROQ_GPT_OSS_HOSTED_TOOL_IDS))
    print()
    print(
        "Examples: examples/agents/01-05 (agents). Remote MCP / compound_custom: Phase B."
    )


if __name__ == "__main__":
    main()
