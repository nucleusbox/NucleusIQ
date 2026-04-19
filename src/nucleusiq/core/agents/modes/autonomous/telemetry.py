"""
Telemetry helpers for ``AutonomousMode``.

Single-responsibility module: every function here writes a structured
record into the agent's ``_tracer`` (if one is attached) and swallows
errors so that telemetry failures never break execution.

Split out of ``autonomous_mode.py`` as part of the Gear-3 modularisation
(SRP): the mode orchestrator should not own tracer-wire-format concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nucleusiq.agents.agent_result import (
    CritiqueSnapshot,
    EscalationRecord,
    LLMCallRecord,
    RevisionRecord,
    ValidationRecord,
)
from nucleusiq.agents.components.compute_budget import (
    ComputeBudget,
    EscalationReason,
)
from nucleusiq.agents.components.critic import CritiqueResult
from nucleusiq.agents.components.refiner import RevisionCandidate

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent


def _tracer(agent: "Agent") -> Any:
    """Return the agent's tracer or ``None``."""
    return getattr(agent, "_tracer", None)


def record_validation(
    agent: "Agent",
    attempt: int,
    valid: bool,
    layer: str,
    reason: str,
) -> None:
    """Append a ``ValidationRecord`` to the tracer."""
    tracer = _tracer(agent)
    if tracer is None:
        return
    try:
        tracer.record_validation(
            ValidationRecord(
                attempt=attempt,
                valid=valid,
                layer=layer,
                reason=reason,
            )
        )
    except Exception:
        pass


def record_revision(
    agent: "Agent",
    attempt: int,
    critique: CritiqueResult,
    revision: RevisionCandidate,
) -> None:
    """Append a ``RevisionRecord`` to ``AutonomousDetail.revisions``."""
    tracer = _tracer(agent)
    if tracer is None:
        return
    try:
        existing = (tracer.autonomous_detail or {}).get("revisions", ())
        record = RevisionRecord(
            attempt=attempt,
            triggered_by_verdict=critique.verdict.value,
            triggered_by_score=critique.score,
            char_delta=revision.char_delta,
            tool_calls_made=revision.tool_calls_made,
            addressed_issues=revision.addressed_issues,
            duration_ms=revision.duration_ms,
        )
        tracer.set_autonomous_detail(revisions=tuple(existing) + (record,))
    except Exception:
        pass


def record_critic_verdict(
    agent: "Agent",
    attempt: int,
    critique: CritiqueResult,
) -> None:
    """Append a ``CritiqueSnapshot`` to ``AutonomousDetail.critic_verdicts``."""
    tracer = _tracer(agent)
    if tracer is None:
        return
    try:
        existing = (tracer.autonomous_detail or {}).get("critic_verdicts", ())
        snap = CritiqueSnapshot(
            attempt=attempt,
            verdict=critique.verdict.value,
            score=critique.score,
            feedback=critique.feedback or "",
            issues=tuple(critique.issues),
            suggestions=tuple(critique.suggestions),
        )
        tracer.set_autonomous_detail(critic_verdicts=tuple(existing) + (snap,))
    except Exception:
        pass


def record_escalation(
    agent: "Agent",
    attempt: int,
    reason: EscalationReason,
    before: ComputeBudget,
    after: ComputeBudget,
) -> None:
    """Append an ``EscalationRecord`` to ``AutonomousDetail.escalations``.

    F3 — compute-budget escalation telemetry.  Written whenever
    ``ComputeBudget.escalate`` is called so the research harness can
    correlate cumulative compute with problem difficulty.
    """
    tracer = _tracer(agent)
    if tracer is None:
        return
    try:
        existing = (tracer.autonomous_detail or {}).get("escalations", ())
        record = EscalationRecord(
            attempt=attempt,
            reason=reason,
            retries_before=before.max_retries,
            retries_after=after.max_retries,
            max_output_tokens_before=before.max_output_tokens,
            max_output_tokens_after=after.max_output_tokens,
            max_tool_calls_before=before.max_tool_calls,
            max_tool_calls_after=after.max_tool_calls,
            cumulative_tokens_at_escalation=after.cumulative_tokens_spent,
        )
        tracer.set_autonomous_detail(escalations=tuple(existing) + (record,))
    except Exception:
        pass


def set_autonomous_detail(
    agent: "Agent",
    attempts: int,
    max_attempts: int,
    sub_tasks: tuple[str, ...] = (),
    complexity: str = "simple",
    refined: bool = False,
    cumulative_tokens: int | None = None,
) -> None:
    """Write the final ``AutonomousDetail`` summary into the tracer."""
    tracer = _tracer(agent)
    if tracer is None:
        return
    try:
        kwargs: dict[str, Any] = dict(
            attempts=attempts,
            max_attempts=max_attempts,
            sub_tasks=sub_tasks,
            complexity=complexity,
            refined=refined,
        )
        if cumulative_tokens is not None:
            kwargs["cumulative_tokens"] = cumulative_tokens
        tracer.set_autonomous_detail(**kwargs)
    except Exception:
        pass


def rollup_sub_agent_metrics(agent: "Agent", sub_results: list) -> None:
    """Roll up sub-agent LLM / tool / context telemetry into the parent.

    Called after ``Decomposer.run_sub_tasks`` in the complex path.  The
    parent ``AgentResult`` needs to reflect *total* work done, not just
    the synthesis pass.
    """
    tracer = _tracer(agent)
    if tracer is None:
        return

    sub_context_tels: list = []

    for sub in sub_results:
        for lc in getattr(sub, "llm_calls", ()):
            tracer.record_llm_call(
                LLMCallRecord(
                    round=len(tracer._llm_calls) + 1,
                    purpose=(
                        f"sub-agent:{lc.purpose}" if lc.purpose else "sub-agent"
                    ),
                    model=lc.model,
                    prompt_tokens=lc.prompt_tokens,
                    completion_tokens=lc.completion_tokens,
                    total_tokens=lc.total_tokens,
                    reasoning_tokens=lc.reasoning_tokens,
                    has_tool_calls=lc.has_tool_calls,
                    tool_call_count=lc.tool_call_count,
                    duration_ms=lc.duration_ms,
                    prompt_technique=lc.prompt_technique,
                )
            )
        for tc in getattr(sub, "tool_calls", ()):
            tracer.record_tool_call(tc)
        ct = getattr(sub, "context_telemetry", None)
        if ct is not None:
            sub_context_tels.append(ct)

    if sub_context_tels:
        agent._sub_agent_context_tels = sub_context_tels


__all__ = [
    "record_validation",
    "record_revision",
    "record_critic_verdict",
    "record_escalation",
    "set_autonomous_detail",
    "rollup_sub_agent_metrics",
]
