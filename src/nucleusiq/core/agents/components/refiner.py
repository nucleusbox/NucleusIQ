"""
Refiner Component — the revision role in NucleusIQ's Autonomous mode.

The Refiner is the third role in the Aletheia-style loop that
``AutonomousMode`` runs.  Mapping the paper's terminology onto actual
NucleusIQ classes:

    Primary agent pass (``StandardMode._tool_call_loop``)
        -> ``Critic``  (the Verifier role — produces a ``CritiqueResult``)
        -> ``Refiner`` (this module — produces a revised candidate)
        -> ``Critic`` re-runs on the revised candidate

Unlike earlier revisions of this component — which only built a prompt
string that was then appended to the primary agent's conversation — the
``Refiner`` now drives its own LLM call.  It reuses
``StandardMode._tool_call_loop`` as its execution engine so the
framework's plugin pipeline, usage tracker and (optional) tool access
all apply, but every call is tagged with ``CallPurpose.REFINER`` so
telemetry keeps the primary / ``Critic`` / ``Refiner`` traffic
separated.

Design principles
-----------------
1. **Own LLM call, own purpose.** The ``Refiner`` is a real role, not a
   prompt-builder helper.  Its calls are tagged ``CallPurpose.REFINER``
   in the usage tracker.
2. **Bounded input scope.** The ``Refiner`` sees the task objective,
   the previous candidate, the ``Critic``'s critique, and (optionally)
   a bounded tool-result summary.  It does **not** see the primary
   agent's full conversation — that would re-introduce the same bias
   the ``Critic`` just flagged.
3. **Tool-enabled, but re-synthesis-biased.** The ``Refiner`` *may*
   call tools if the critique specifically requires missing data, but
   the system prompt tells it to prefer re-synthesising from the
   summary.
4. **Structured output.** Returns a ``RevisionCandidate`` carrying the
   revised content and metrics (tool calls, char delta, addressed
   issues) consumed by ``AutonomousMode`` telemetry and by the upcoming
   ``ComputeBudget``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import CritiqueResult

# ------------------------------------------------------------------ #
# Output model                                                        #
# ------------------------------------------------------------------ #


class RevisionCandidate(BaseModel):
    """The result of a single ``Refiner.revise`` pass.

    ``content`` is the new candidate answer; the remaining fields are
    bookkeeping consumed by ``AutonomousMode`` telemetry (``RevisionRecord``)
    and by the upcoming ``ComputeBudget`` scheduler.
    """

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="The revised candidate answer.")
    addressed_issues: tuple[str, ...] = Field(
        default=(),
        description=(
            "Issues the Refiner believes it addressed. Today we copy the "
            "Critic's issue list verbatim; a future revision may parse "
            "this from a structured Refiner response."
        ),
    )
    tool_calls_made: int = Field(
        default=0,
        description="Number of tool calls the Refiner made during this pass.",
    )
    char_delta: int = Field(
        default=0,
        description=(
            "len(new candidate) - len(previous candidate). Useful as a "
            "cheap proxy for how much the Refiner actually changed."
        ),
    )
    duration_ms: float = Field(
        default=0.0,
        description="Wall-clock time spent inside Refiner.revise, in milliseconds.",
    )


_REFINER_SYSTEM_PROMPT = (
    "You are the Refiner component in NucleusIQ's Autonomous mode. "
    "The primary agent pass produced a candidate answer and the Critic "
    "reviewed it and flagged specific issues. Your job is to produce a "
    "corrected final answer that addresses the Critic's feedback "
    "without redoing work that was already correct. Prefer "
    "re-synthesising from the evidence provided over calling new "
    "tools; only call a tool if the Critic explicitly identified "
    "MISSING data that cannot be inferred from the summary. Return a "
    "single, self-contained final answer — no preamble, no apology, "
    "no meta-commentary."
)


_MAX_CANDIDATE_CHARS = 8_000
_MAX_TOOL_SUMMARY_CHARS = 4_000


class Refiner:
    """The revision role in NucleusIQ's Autonomous mode loop.

    Runs after the ``Critic`` rejects a candidate: takes the task,
    the prior candidate and the ``CritiqueResult``, and produces a
    new ``RevisionCandidate`` via its own LLM pass (tagged
    ``CallPurpose.REFINER``).

    Usage (the only supported entry point)::

        refiner = Refiner()
        revision = await refiner.revise(
            agent=agent,
            task_objective=task.objective,
            candidate=previous_result,
            critique=critique,
            tool_result_summary=tool_summary,  # optional
        )
        new_candidate = revision.content
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def revise(
        self,
        agent: Agent,
        task_objective: str,
        candidate: Any,
        critique: CritiqueResult,
        tool_result_summary: str | None = None,
    ) -> RevisionCandidate:
        """Produce a revised candidate using the ``Critic``'s feedback.

        Runs an independent tool-enabled LLM pass via
        ``StandardMode._tool_call_loop`` with ``CallPurpose.REFINER``.
        Does **not** mutate the primary agent's conversation — the
        caller is responsible for deciding how the returned candidate
        flows into the next ``Critic`` pass.

        Raises any exception from the underlying LLM / tool loop; the
        caller (``AutonomousMode``) is responsible for graceful
        degradation.

        Args:
            agent: The active Agent (supplies ``llm``, ``tools``,
                plugin pipeline, usage tracker).
            task_objective: The original user task text.
            candidate: The previous candidate answer the ``Critic``
                rejected.
            critique: The ``Critic``'s structured feedback.
            tool_result_summary: Optional bounded summary (<= 4 KB) of
                tool outputs produced earlier in this task.  Letting
                the ``Refiner`` see the evidence that grounded the
                original answer makes re-synthesis much more reliable
                than blind revision.

        Returns:
            A frozen ``RevisionCandidate``.
        """
        if agent.llm is None:
            raise RuntimeError(
                "Refiner.revise requires an LLM; call only when agent.llm is set."
            )

        # Delayed imports to avoid circular dependency at module load.
        from nucleusiq.agents.modes.standard_mode import StandardMode
        from nucleusiq.agents.task import Task
        from nucleusiq.agents.usage.usage_tracker import CallPurpose

        std_mode = StandardMode()
        std_mode._ensure_executor(agent)
        tool_specs = std_mode._get_tool_specs(agent)

        prompt = self._build_revision_prompt(
            task_objective=task_objective,
            candidate=candidate,
            critique=critique,
            tool_result_summary=tool_result_summary,
        )

        revision_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=_REFINER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]

        revise_task = Task(
            id="refiner_pass",
            objective=task_objective,
        )

        tool_calls_before = self._count_tool_calls(agent)
        previous_len = len(str(candidate or ""))

        self._logger.info(
            "Refiner: revising candidate (verdict=%s, score=%.2f, issues=%d)",
            critique.verdict.value,
            critique.score,
            len(critique.issues),
        )

        start = time.perf_counter()
        try:
            revised = await std_mode._tool_call_loop(
                agent,
                revise_task,
                revision_messages,
                tool_specs,
                purpose_override=CallPurpose.REFINER,
            )
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0

        content = str(revised) if revised is not None else ""
        tool_calls_made = max(0, self._count_tool_calls(agent) - tool_calls_before)

        return RevisionCandidate(
            content=content,
            addressed_issues=tuple(critique.issues),
            tool_calls_made=tool_calls_made,
            char_delta=len(content) - previous_len,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------ #
    # Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_revision_prompt(
        task_objective: str,
        candidate: Any,
        critique: CritiqueResult,
        tool_result_summary: str | None,
    ) -> str:
        issues_block = (
            "\n".join(f"- {i}" for i in critique.issues)
            if critique.issues
            else "- (Critic did not list specific issues — see feedback above)"
        )
        suggestions_block = (
            "\n".join(f"- {s}" for s in critique.suggestions)
            if critique.suggestions
            else "(no specific suggestions)"
        )

        tool_block = ""
        if tool_result_summary:
            truncated = tool_result_summary[:_MAX_TOOL_SUMMARY_CHARS]
            tool_block = (
                "\n## Tool Results From Previous Attempt (bounded summary)\n"
                f"{truncated}\n"
            )

        candidate_str = str(candidate or "")[:_MAX_CANDIDATE_CHARS]

        return (
            f"## Task\n{task_objective}\n\n"
            f"## Previous Candidate Answer\n{candidate_str}\n\n"
            f"## Critic Verdict\n"
            f"- verdict: {critique.verdict.value}\n"
            f"- score: {critique.score:.2f}\n"
            f"- overall: {critique.feedback or '(none)'}\n\n"
            f"## Issues To Fix\n{issues_block}\n\n"
            f"## Suggested Improvements\n{suggestions_block}\n"
            f"{tool_block}\n"
            "## Revision Instructions\n"
            "1. Fix ONLY the specific issues listed above — keep what was correct.\n"
            "2. Re-synthesise the answer using the tool results already gathered.\n"
            "3. Do NOT call tools unless the Critic specifically identified "
            "MISSING data that cannot be inferred from the summary above.\n"
            "4. Return a complete, self-contained final answer that addresses "
            "the original task.\n"
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_tool_calls(agent: Agent) -> int:
        """Best-effort count of tool calls observed so far, from the tracer.

        Used to compute the per-revision ``tool_calls_made`` delta.
        Falls back to 0 if no tracer is attached (e.g. in tests).
        """
        tracer = getattr(agent, "_tracer", None)
        if tracer is None:
            return 0
        try:
            return len(tracer.tool_calls)
        except Exception:
            return 0


__all__ = ["Refiner", "RevisionCandidate"]
