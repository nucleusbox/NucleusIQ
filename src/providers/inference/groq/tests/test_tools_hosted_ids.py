"""Documented Groq hosted tool identifiers (Phase A: not wired, constants only)."""

from __future__ import annotations

from nucleusiq_groq.tools import (
    GROQ_COMPOUND_HOSTED_TOOL_IDS,
    GROQ_GPT_OSS_HOSTED_TOOL_IDS,
    NATIVE_TOOL_TYPES,
)


def test_compound_hosted_ids_match_groq_docs() -> None:
    assert (
        frozenset(
            {
                "web_search",
                "code_interpreter",
                "visit_website",
                "browser_automation",
                "wolfram_alpha",
            }
        )
        == GROQ_COMPOUND_HOSTED_TOOL_IDS
    )


def test_gpt_oss_hosted_ids_match_groq_docs() -> None:
    assert (
        frozenset({"browser_search", "code_interpreter"})
        == GROQ_GPT_OSS_HOSTED_TOOL_IDS
    )
    assert "web_search" not in GROQ_GPT_OSS_HOSTED_TOOL_IDS


def test_native_tool_types_empty_until_phase_b() -> None:
    assert frozenset() == NATIVE_TOOL_TYPES
