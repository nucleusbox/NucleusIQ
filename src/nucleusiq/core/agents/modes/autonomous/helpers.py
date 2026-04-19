"""
Pure utility helpers for ``AutonomousMode``.

Single-responsibility: none of these functions have external side
effects or hold state.  Anything here should be trivially unit-testable
with no mocking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import (
    CriticLimits,
    CritiqueResult,
    REASONING_LIMITS,
    STANDARD_LIMITS,
)
from nucleusiq.agents.components.validation import ValidationResult

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent


# ------------------------------------------------------------------ #
# Critic limits selection                                              #
# ------------------------------------------------------------------ #


def select_critic_limits(agent: "Agent") -> CriticLimits:
    """Pick REASONING or STANDARD limits based on the agent's LLM."""
    is_reasoning = getattr(
        getattr(agent, "llm", None), "is_reasoning_model", False
    )
    return REASONING_LIMITS if is_reasoning else STANDARD_LIMITS


# ------------------------------------------------------------------ #
# Result inspection                                                    #
# ------------------------------------------------------------------ #


def is_error_result(result: Any) -> bool:
    """Return True if ``result`` is a framework error-string sentinel."""
    return isinstance(result, str) and result.strip().startswith("Error:")


# ------------------------------------------------------------------ #
# Tool-result summarisation                                            #
# ------------------------------------------------------------------ #


def summarize_tool_results(
    messages: list[ChatMessage],
    *,
    max_chars: int = 4_000,
    per_item_chars: int = 500,
) -> str | None:
    """Build a bounded summary of ``role='tool'`` messages for the Refiner.

    Each tool result is head-truncated to ``per_item_chars`` and entries
    accumulate up to ``max_chars`` total.  Returns ``None`` if no tool
    messages are present.
    """
    parts: list[str] = []
    total = 0
    for msg in messages:
        if getattr(msg, "role", None) != "tool":
            continue
        content = str(getattr(msg, "content", "") or "")
        if not content:
            continue
        excerpt = content[:per_item_chars]
        tool_name = getattr(msg, "name", None) or "tool"
        entry = f"[{tool_name}] {excerpt}"
        if total + len(entry) + 1 > max_chars:
            break
        parts.append(entry)
        total += len(entry) + 1
    return "\n".join(parts) if parts else None


# ------------------------------------------------------------------ #
# Retry / revision prompt builders                                     #
# ------------------------------------------------------------------ #


def build_validation_retry(vr: ValidationResult) -> str:
    """Build the retry message injected after a Layer 1/2 validation fail."""
    parts = [f"Your previous answer had an issue: {vr.reason}"]
    if vr.details:
        parts.append(f"Details: {'; '.join(vr.details)}")
    parts.append("Please fix the issue and provide a corrected answer.")
    return "\n".join(parts)


def build_fallback_revision_message(critique: CritiqueResult) -> str:
    """Legacy-style revision message, used only when ``Refiner.revise`` fails.

    Keeps the framework resilient: if the Refiner raises, ``AutonomousMode``
    falls back to injecting this correction directive into the primary
    agent's conversation and lets it re-run (the pre-F1 behaviour).
    """
    issues = (
        "\n".join(f"- {i}" for i in critique.issues)
        if critique.issues
        else "- error detected in your answer"
    )
    suggestions = (
        "\n".join(f"- {s}" for s in critique.suggestions)
        if critique.suggestions
        else ""
    )
    msg = (
        f"VERIFICATION RESULT: {critique.verdict.value} "
        f"(score: {critique.score:.2f})\n\n"
    )
    if critique.feedback:
        msg += f"Assessment: {critique.feedback}\n\n"
    msg += f"Issues identified:\n{issues}\n"
    if suggestions:
        msg += f"\nSuggested fixes:\n{suggestions}\n"
    msg += (
        "\nREVISION INSTRUCTIONS:\n"
        "- Fix ONLY the specific error(s) listed above.\n"
        "- Do NOT redo work that was already correct.\n"
        "- Your previous answer is in the conversation above — "
        "revise it to address the issues.\n"
        "- Provide your corrected COMPLETE final answer."
    )
    return msg


__all__ = [
    "build_fallback_revision_message",
    "build_validation_retry",
    "is_error_result",
    "select_critic_limits",
    "summarize_tool_results",
]
