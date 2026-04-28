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
    REASONING_LIMITS,
    STANDARD_LIMITS,
    CriticLimits,
    CritiqueResult,
)
from nucleusiq.agents.components.validation import ValidationResult

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.agents.context.store import ContentStore


# ------------------------------------------------------------------ #
# Critic limits selection                                              #
# ------------------------------------------------------------------ #


def select_critic_limits(agent: Agent) -> CriticLimits:
    """Pick REASONING or STANDARD limits based on the agent's LLM."""
    is_reasoning = getattr(getattr(agent, "llm", None), "is_reasoning_model", False)
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
    per_tool_char_cap: int | None = None,
    total_char_cap: int | None = None,
    content_store: ContentStore | None = None,
) -> str | None:
    """Return rehydrated tool-result evidence for the Refiner.

    v0.7.8 — the pre-existing head-truncation (``per_item_chars=500``,
    total ``max_chars=4_000``) has been removed.  It was the primary
    cause of the post-F1-F9 Refiner regression: the Refiner asked for
    evidence, the helper rehydrated the real tool content, then
    silently sliced the first 500 chars of each result (invariably
    boilerplate for structured documents) and discarded the rest.

    New behaviour:

    * When ``content_store`` is provided, each ``role='tool'`` message
      is rehydrated through ``extract_raw_trace`` using
      ``per_tool_char_cap`` as the per-result budget.
    * When ``per_tool_char_cap`` is ``None``, the helper applies no
      per-tool truncation at all (callers that can see both the LLM's
      context window and the number of tool results should always
      pass a cap; absence means "give me everything").
    * ``total_char_cap`` is an optional aggregate safety ceiling —
      when provided, concatenation stops once the running byte count
      reaches it.  Intended to guard against a 400K-char PDF batch
      overflowing a small-context model even when each per-tool slice
      is within budget.
    * Returns ``None`` if no ``role='tool'`` messages are present.

    Args:
        messages: Conversation messages.
        per_tool_char_cap: Adaptive per-tool-result char cap (see
            ``compute_per_tool_cap``).  ``None`` means no per-tool
            truncation.
        total_char_cap: Aggregate char ceiling across all tool
            results.  ``None`` means no aggregate ceiling.
        content_store: ``ContentStore`` used to rehydrate masked
            observations.  When ``None`` the messages are formatted
            as-is.
    """
    if content_store is not None:
        from nucleusiq.agents.context.store import extract_raw_trace

        # When no per-tool cap is supplied, pass the extract_raw_trace
        # ceiling so a single pathological result doesn't balloon the
        # Refiner prompt — 50K chars matches the v0.7.8
        # ``tool_result_per_call_max_chars`` default.
        effective_cap = per_tool_char_cap if per_tool_char_cap is not None else 50_000
        messages = extract_raw_trace(
            messages,
            content_store,
            max_chars_per_result=effective_cap,
        )
    parts: list[str] = []
    total = 0
    for msg in messages:
        if getattr(msg, "role", None) != "tool":
            continue
        content = str(getattr(msg, "content", "") or "")
        if not content:
            continue
        tool_name = getattr(msg, "name", None) or "tool"
        entry = f"[{tool_name}] {content}"
        if total_char_cap is not None and total + len(entry) + 1 > total_char_cap:
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
