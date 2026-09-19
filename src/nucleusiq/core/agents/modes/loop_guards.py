"""Termination guards shared by the sync and streaming tool loops.

Invariant I-1 (docs/design/AUTONOMOUS_HARNESS_HARDENING.md): every loop
is bounded by a counter or a clock, no tool category is exempt from
*all* bounds, and a model that repeats itself is stopped early.

Two guards live here so ``StandardMode._tool_call_loop`` and
``BaseExecutionMode._streaming_tool_call_loop`` behave identically:

* :class:`ContextToolBudget` — context-management tools (recall,
  workspace, evidence, corpus) deliberately do not consume the user's
  ``max_tool_calls``.  They get their own cap instead.
* :class:`ProgressTracker` — detects a model that is stuck: the same
  tool calls producing the same results for several consecutive rounds,
  or rounds whose only outputs are dedup banners / recall errors.
  Polling tools whose results change never trip it.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage

NO_PROGRESS_ROUNDS = 3
"""Consecutive identical (or fully stalled) tool rounds before we stop."""

NO_PROGRESS_NUDGE = (
    "[loop guard] Your last rounds repeated the same tool calls and received "
    "the same results, or returned only duplicate-call banners / recall "
    "errors. Repeating them again will not produce new information and the "
    "run will be stopped. Do NOT call those tools again. Use the evidence "
    "already in this conversation (recall a ref only if one is visible) and "
    "produce your final answer now, noting any gaps explicitly."
)

_DEDUP_BANNER_PREFIX = "[duplicate idempotent call"
_RECALL_ERROR_PREFIX = "[recall_error"


def is_stalled_tool_content(content: Any) -> bool:
    """True when a tool message carries no new information for the model."""
    if not isinstance(content, str):
        return False
    head = content.lstrip()[:40]
    return head.startswith(_DEDUP_BANNER_PREFIX) or head.startswith(
        _RECALL_ERROR_PREFIX
    )


def _short_hash(text: str) -> str:
    return hashlib.sha1(
        text.encode("utf-8", "replace"), usedforsecurity=False
    ).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Wall clock                                                                  #
# --------------------------------------------------------------------------- #


def remaining_seconds(agent: Any) -> float | None:
    """Seconds left before ``AgentConfig.max_execution_time`` expires.

    ``None`` when the agent has no deadline (unlimited, or a test double
    without ``_run_deadline``).  Uses ``time.monotonic()`` so clock
    adjustments cannot extend or shorten a run.
    """
    deadline = getattr(agent, "_run_deadline", None)
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
        return None
    return float(deadline) - time.monotonic()


def deadline_exceeded(agent: Any) -> bool:
    """True once the run's wall-clock budget is spent."""
    left = remaining_seconds(agent)
    return left is not None and left <= 0.0


def explicit_timeout(config: Any, field: str) -> float | None:
    """Return ``config.<field>`` only when the user set it explicitly.

    ``llm_call_timeout`` / ``step_timeout`` have had defaults for years
    without being enforced; applying them retroactively would break
    reasoning models and long tools.  Enforcement is opt-in.
    """
    explicit = getattr(config, "model_fields_set", None)
    if not isinstance(explicit, (set, frozenset)) or field not in explicit:
        return None
    value = getattr(config, field, None)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return float(value)


@dataclass(frozen=True)
class ProgressVerdict:
    identical_streak: int
    stalled_streak: int
    stalled_round: bool

    @property
    def no_progress(self) -> bool:
        return (
            self.identical_streak >= NO_PROGRESS_ROUNDS
            or self.stalled_streak >= NO_PROGRESS_ROUNDS
        )

    @property
    def should_nudge(self) -> bool:
        """One round before stopping: tell the model to answer with what it has."""
        threshold = NO_PROGRESS_ROUNDS - 1
        return not self.no_progress and (
            self.identical_streak == threshold or self.stalled_streak == threshold
        )

    @property
    def reason(self) -> str:
        if self.identical_streak >= NO_PROGRESS_ROUNDS:
            return (
                f"the same tool calls returned the same results for "
                f"{self.identical_streak} consecutive rounds"
            )
        return (
            f"{self.stalled_streak} consecutive rounds produced only duplicate-call "
            "banners or recall errors"
        )


class ProgressTracker:
    """Track per-round tool signatures to detect a stuck model."""

    def __init__(self) -> None:
        self._last_signature: str | None = None
        self._identical_streak = 0
        self._stalled_streak = 0
        self.nudged = False

    def nudge_message(self) -> Any:
        """Return the one-time nudge ``ChatMessage`` or ``None`` if already sent."""
        if self.nudged:
            return None
        self.nudged = True
        from nucleusiq.agents.chat_models import ChatMessage

        return ChatMessage(role="user", content=NO_PROGRESS_NUDGE)

    def observe(self, round_messages: list[ChatMessage]) -> ProgressVerdict:
        """Feed the messages appended during one tool round.

        ``round_messages`` is the slice of the transcript added by this
        round: one assistant message with ``tool_calls`` followed by its
        ``role="tool"`` results (before any masking).
        """
        parts: list[str] = []
        tool_contents: list[Any] = []
        for m in round_messages:
            role = getattr(m, "role", None)
            if role == "assistant":
                for tc in getattr(m, "tool_calls", None) or []:
                    parts.append(
                        f"call:{getattr(tc, 'name', '')}:"
                        f"{_short_hash(str(getattr(tc, 'arguments', '') or ''))}"
                    )
            elif role == "tool":
                content = getattr(m, "content", None)
                tool_contents.append(content)
                parts.append(
                    f"result:{getattr(m, 'name', '')}:{_short_hash(str(content or ''))}"
                )

        if not parts:
            return ProgressVerdict(self._identical_streak, self._stalled_streak, False)

        signature = _short_hash("|".join(sorted(parts)))
        if signature == self._last_signature:
            self._identical_streak += 1
        else:
            self._identical_streak = 1
        self._last_signature = signature

        stalled_round = bool(tool_contents) and all(
            is_stalled_tool_content(c) for c in tool_contents
        )
        self._stalled_streak = self._stalled_streak + 1 if stalled_round else 0

        return ProgressVerdict(
            self._identical_streak, self._stalled_streak, stalled_round
        )


class ContextToolBudget:
    """Counter for framework context-management tool calls."""

    def __init__(self, cap: int) -> None:
        self.cap = max(0, int(cap))
        self.used = 0

    def consume(self, n: int = 1) -> None:
        self.used += int(n)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.cap


def split_tool_call_counts(names: list[str | None]) -> tuple[int, int]:
    """Return ``(business_calls, context_management_calls)`` for a round."""
    from nucleusiq.agents.context.workspace_tools import is_context_management_tool_name

    ctx = sum(1 for n in names if is_context_management_tool_name(n))
    return len(names) - ctx, ctx
