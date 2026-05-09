"""Groq tool support — local function calling uses OpenAI-style tool specs.

Hosted (server-side) tool identifiers are **documented** below for validation
and UX (aligned with Groq’s `built-in-tools` page). They are **not** registered
in :data:`NATIVE_TOOL_TYPES` until Phase B wires pass-through / Responses.

See ``docs/design/GROQ_PROVIDER.md`` §8.1.
"""

from __future__ import annotations

from nucleusiq_groq.tools.converter import to_openai_function_tool

# Compound (`groq/compound`, `groq/compound-mini`): `compound_custom.tools.enabled_tools`.
GROQ_COMPOUND_HOSTED_TOOL_IDS: frozenset[str] = frozenset(
    {
        "web_search",
        "code_interpreter",
        "visit_website",
        "browser_automation",
        "wolfram_alpha",
    }
)

# GPT-OSS on Groq (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`): `tools=[{type: ...}]`.
GROQ_GPT_OSS_HOSTED_TOOL_IDS: frozenset[str] = frozenset(
    {
        "browser_search",
        "code_interpreter",
    }
)

# Types NucleusIQ treats as *provider-native* for routing (empty in Phase A).
NATIVE_TOOL_TYPES: frozenset[str] = frozenset()

__all__ = [
    "GROQ_COMPOUND_HOSTED_TOOL_IDS",
    "GROQ_GPT_OSS_HOSTED_TOOL_IDS",
    "NATIVE_TOOL_TYPES",
    "to_openai_function_tool",
]
