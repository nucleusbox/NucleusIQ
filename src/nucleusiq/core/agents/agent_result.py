"""AgentResult — immutable execution result from Agent.execute().

This module defines the public response contract between the agent
framework and its users.  Every field is populated by the
``ExecutionTracer`` during execution and frozen at teardown.

Design patterns applied:
    - **Immutable Value Object**: ``frozen=True`` prevents post-creation mutation.
    - **Composite**: ``AgentResult`` composes independent sub-models.
    - **Open/Closed**: New fields with defaults grow the model without breakage.

Backward compatible: ``str(result)`` returns the output text, so code
that previously treated the return value as a string still works.

Usage::

    result = await agent.execute(task)

    # Backward-compatible string access
    print(result)  # prints output text

    # Typed access
    if result:  # True when status == SUCCESS
        print(result.output)
    else:
        print(f"Failed: {result.error} ({result.error_type})")

    # Observability (tool_calls, llm_calls, warnings populated since 0.7.4)
    for tc in result.tool_calls:
        print(f"  {tc.tool_name}: {tc.duration_ms}ms")
    for lc in result.llm_calls:
        print(f"  LLM round {lc.round}: {lc.total_tokens} tokens, {lc.duration_ms}ms")

    # Serialization
    result.model_dump_json()  # JSON string
    result.summary()  # dict (exclude_none)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from nucleusiq.agents.components.compute_budget import AbstainReason
    from nucleusiq.agents.components.critic import CritiqueResult

# ------------------------------------------------------------------ #
# Enums                                                                #
# ------------------------------------------------------------------ #


class ResultStatus(str, Enum):
    """Final outcome of an ``Agent.execute()`` call.

    * ``SUCCESS``   — the agent produced an answer that passed every
      internal gate (validation + Critic).
    * ``ERROR``     — execution raised an unhandled exception.
    * ``HALTED``    — a plugin raised ``PluginHalt`` to end execution
      early.
    * ``ABSTAINED`` — (Autonomous mode, F2) the agent exhausted its
      retries and the Critic's final verdict was FAIL or
      UNCERTAIN-below-threshold; the ``AgentResult`` carries the best
      candidate plus a machine-readable ``abstention_reason`` and the
      final ``CritiqueResult`` via ``autonomous.critic_verdicts[-1]``.
      This is a first-class outcome distinct from ``SUCCESS`` and
      ``ERROR`` — callers decide whether to retry with a better model,
      escalate to a human, hand off to another agent, or treat the
      task as failed.  No human-in-the-loop is part of the contract.
    """

    SUCCESS = "success"
    ERROR = "error"
    HALTED = "halted"
    ABSTAINED = "abstained"


# ------------------------------------------------------------------ #
# Sub-models (leaf-level, no cross-imports)                            #
# ------------------------------------------------------------------ #


class ToolCallRecord(BaseModel):
    """One tool invocation during execution."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    tool_call_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    success: bool = True
    error: str | None = None
    error_type: str | None = None
    duration_ms: float = 0.0
    round: int = 1


class LLMCallRecord(BaseModel):
    """One LLM API call during execution."""

    model_config = ConfigDict(frozen=True)

    round: int
    purpose: str = ""
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    has_tool_calls: bool = False
    tool_call_count: int = 0
    duration_ms: float = 0.0
    prompt_technique: str | None = None


class PluginEvent(BaseModel):
    """One plugin hook execution."""

    model_config = ConfigDict(frozen=True)

    plugin_name: str
    hook: str
    action: str = "executed"
    detail: str | None = None
    duration_ms: float = 0.0


class MemorySnapshot(BaseModel):
    """Post-execution memory state."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    message_count: int = 0
    token_count: int | None = None
    messages: tuple[dict[str, str], ...] = ()


class ValidationRecord(BaseModel):
    """One validation attempt in autonomous mode."""

    model_config = ConfigDict(frozen=True)

    attempt: int
    valid: bool
    layer: str = ""
    reason: str = ""


class CritiqueSnapshot(BaseModel):
    """Frozen snapshot of a Critic verdict for telemetry.

    Produced by ``AutonomousMode`` after every ``_run_critic`` call and
    stored on ``AutonomousDetail.critic_verdicts``. Intentionally a thin
    copy of ``CritiqueResult`` so the runtime type can keep evolving
    without breaking the persisted record shape.
    """

    model_config = ConfigDict(frozen=True)

    attempt: int
    verdict: str
    score: float = 0.0
    feedback: str = ""
    issues: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()


class RevisionRecord(BaseModel):
    """One Refiner pass — produced a revised candidate from a critique.

    Written by ``AutonomousMode`` whenever ``Refiner.revise`` runs and
    yields a ``RevisionCandidate``. Surfaces the cost and effect of each
    revision pass so the research harness (and later ``ComputeBudget``)
    can measure whether revision is moving scores in the right direction.
    """

    model_config = ConfigDict(frozen=True)

    attempt: int
    triggered_by_verdict: str = ""
    triggered_by_score: float = 0.0
    char_delta: int = 0
    tool_calls_made: int = 0
    addressed_issues: tuple[str, ...] = ()
    duration_ms: float = 0.0


class EscalationRecord(BaseModel):
    """F3 — one budget escalation event.

    Emitted by ``AutonomousMode`` whenever ``ComputeBudget.escalate`` is
    called.  Surfaces *why* we spent more compute so the research harness
    can correlate cost with problem difficulty.
    """

    model_config = ConfigDict(frozen=True)

    attempt: int
    #: Escalation reason from ``compute_budget.EscalationReason``
    #: (``"uncertain_close"`` or ``"stuck"``).  Stored as ``str`` so the
    #: set can grow without a schema migration.
    reason: str
    retries_before: int
    retries_after: int
    max_output_tokens_before: int
    max_output_tokens_after: int
    max_tool_calls_before: int
    max_tool_calls_after: int
    cumulative_tokens_at_escalation: int = 0


class AutonomousDetail(BaseModel):
    """Autonomous-mode execution details."""

    model_config = ConfigDict(frozen=True)

    attempts: int = 1
    max_attempts: int = 1
    sub_tasks: tuple[str, ...] = ()
    complexity: str | None = None
    validations: tuple[ValidationRecord, ...] = ()
    refined: bool = False
    # F1/F5 — Reviser-as-role telemetry
    revisions: tuple[RevisionRecord, ...] = ()
    critic_verdicts: tuple[CritiqueSnapshot, ...] = ()
    # F3/F5 — compute-budget telemetry
    escalations: tuple[EscalationRecord, ...] = ()
    cumulative_tokens: int = 0
    # F4/F5 — Best-of-N parallel attempts. Empty when
    # ``n_parallel_attempts == 1`` (default).  Otherwise contains one
    # ``AutonomousDetail`` per attempt (the detail on the enclosing
    # model describes the selected attempt / abstention outcome).
    parallel_attempts: tuple[AutonomousDetail, ...] = ()
    # F4 — which attempt produced the returned candidate (0-based index
    # into ``parallel_attempts``).  ``None`` when ``parallel_attempts``
    # is empty (single-attempt runs).
    selected_attempt: int | None = None


# ------------------------------------------------------------------ #
# AgentResult — the root response model                                #
# ------------------------------------------------------------------ #


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentResult(BaseModel):
    """Immutable execution result from ``Agent.execute()``.

    This is the public contract between the agent framework and its
    users.  Every field is populated by the ``ExecutionTracer`` during
    execution and frozen at teardown.
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    agent_id: str
    agent_name: str
    task_id: str
    mode: str
    model: str | None = None
    created_at: str = Field(default_factory=_utc_now_iso)

    # --- Outcome ---
    output: Any = None
    status: ResultStatus = ResultStatus.SUCCESS
    error: str | None = None
    error_type: str | None = None
    duration_ms: float = 0.0
    # Populated when ``status == ResultStatus.ABSTAINED`` (F2): a plain-
    # text explanation (typically the final Critic's ``feedback``).  The
    # full structured critique is available via
    # ``autonomous.critic_verdicts[-1]``.
    abstention_reason: str | None = None
    # F5 — machine-readable abstention reason.  Today one of
    # ``"budget_exhausted"`` or ``"stuck_after_escalation"`` (see
    # ``compute_budget.AbstainReason``).  ``None`` when the run did not
    # abstain, or when abstention was triggered by a path that does
    # not flow through the ComputeBudget controller (e.g. Best-of-N
    # synthesis abstention).  Callers should treat an unknown value as
    # equivalent to ``None`` so the literal can grow without breaking
    # clients.
    abstention_code: str | None = None

    # --- Tool observability (populated since 0.7.4) ---
    tool_calls: tuple[ToolCallRecord, ...] = ()

    # --- LLM observability (populated since 0.7.4) ---
    llm_calls: tuple[LLMCallRecord, ...] = ()

    # --- Conversation history ---
    messages: tuple[dict[str, Any], ...] = ()

    # --- Usage (reuses existing UsageSummary via dict for now) ---
    usage: dict[str, Any] | None = None

    # --- Memory state (wired in future 0.7.x) ---
    memory_snapshot: MemorySnapshot | None = None

    # --- Plugin audit trail (wired in future 0.7.x) ---
    plugin_events: tuple[PluginEvent, ...] = ()

    # --- Autonomous-mode detail (wired in future 0.7.x) ---
    autonomous: AutonomousDetail | None = None

    # --- Context window telemetry (populated since 0.7.6) ---
    context_telemetry: Any = None

    # --- Non-fatal issues (populated since 0.7.4) ---
    warnings: tuple[str, ...] = ()

    # --- Extension point (Open/Closed) ---
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Backward compatibility                                               #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        """``str(result)`` returns output text — backward compat."""
        return str(self.output) if self.output is not None else ""

    def __bool__(self) -> bool:
        """``if result:`` is True when status is SUCCESS."""
        return self.status == ResultStatus.SUCCESS

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def is_error(self) -> bool:
        """True when status is ERROR."""
        return self.status == ResultStatus.ERROR

    @property
    def is_halted(self) -> bool:
        """True when status is HALTED (plugin early-exit)."""
        return self.status == ResultStatus.HALTED

    @property
    def is_abstained(self) -> bool:
        """True when status is ABSTAINED (F2: Autonomous mode self-abstention)."""
        return self.status == ResultStatus.ABSTAINED

    @property
    def tool_call_count(self) -> int:
        """Total number of tool calls in this execution."""
        return len(self.tool_calls)

    @property
    def failed_tool_calls(self) -> tuple[ToolCallRecord, ...]:
        """Tool calls that ended in failure."""
        return tuple(tc for tc in self.tool_calls if not tc.success)

    # ------------------------------------------------------------------ #
    # Serialization helpers                                                #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        """Plain dict for JSON serialization / logging / dashboards."""
        return self.model_dump(exclude_none=True)

    def display(self) -> str:
        """Human-readable execution summary with full observability."""
        lines: list[str] = []
        lines.append(f"AgentResult(status={self.status.value})")
        lines.append(f"  Agent  : {self.agent_name} ({self.agent_id})")
        lines.append(f"  Task   : {self.task_id}")
        lines.append(f"  Mode   : {self.mode}")
        if self.model:
            lines.append(f"  Model  : {self.model}")
        lines.append(f"  Time   : {self.duration_ms:.1f}ms")

        if self.is_error:
            lines.append(f"  Error  : [{self.error_type}] {self.error}")
        else:
            output_preview = str(self.output)[:200] if self.output else "(none)"
            lines.append(f"  Output : {output_preview}")

        if self.tool_calls:
            lines.append(f"  Tools  : {len(self.tool_calls)} calls")
            for tc in self.tool_calls:
                status = "OK" if tc.success else "FAIL"
                lines.append(f"    [{status}] {tc.tool_name}({tc.duration_ms:.0f}ms)")

        if self.llm_calls:
            total_tokens = sum(lc.total_tokens for lc in self.llm_calls)
            lines.append(
                f"  LLM    : {len(self.llm_calls)} calls, {total_tokens} tokens"
            )
            for lc in self.llm_calls:
                purpose = f" [{lc.purpose}]" if lc.purpose else ""
                lines.append(
                    f"    Round {lc.round}{purpose}: "
                    f"{lc.total_tokens} tokens, {lc.duration_ms:.0f}ms"
                )

        if self.plugin_events:
            lines.append(f"  Plugins: {len(self.plugin_events)} events")
            for pe in self.plugin_events:
                lines.append(
                    f"    {pe.plugin_name}.{pe.hook} "
                    f"[{pe.action}] {pe.duration_ms:.1f}ms"
                )

        if self.memory_snapshot:
            ms = self.memory_snapshot
            token_info = f", ~{ms.token_count} tokens" if ms.token_count else ""
            lines.append(
                f"  Memory : {ms.strategy} ({ms.message_count} messages{token_info})"
            )

        if self.autonomous:
            ad = self.autonomous
            lines.append(
                f"  Auto   : {ad.complexity or 'unknown'} "
                f"({ad.attempts}/{ad.max_attempts} attempts)"
            )
            if ad.sub_tasks:
                lines.append(f"    Sub-tasks: {', '.join(ad.sub_tasks[:5])}")
            if ad.validations:
                for v in ad.validations:
                    verdict = "PASS" if v.valid else "FAIL"
                    lines.append(
                        f"    [{verdict}] attempt {v.attempt} ({v.layer}): {v.reason}"
                    )
            if ad.refined:
                lines.append("    Refined: yes")

        if self.context_telemetry is not None:
            ct = self.context_telemetry
            lines.append(
                f"  Context: {ct.context_limit} tokens "
                f"(peak {ct.peak_utilization:.0%}, final {ct.final_utilization:.0%})"
            )
            if ct.compaction_count > 0:
                lines.append(
                    f"    Compactions: {ct.compaction_count} "
                    f"(freed {ct.tokens_freed_total} tokens)"
                )
                for ce in ct.compaction_events:
                    lines.append(
                        f"    [{ce.strategy}] {ce.tokens_before}→{ce.tokens_after} "
                        f"({ce.tokens_freed} freed, {ce.duration_ms:.1f}ms)"
                    )
            if ct.artifacts_offloaded > 0:
                lines.append(f"    Offloaded: {ct.artifacts_offloaded} artifacts")
            if ct.region_breakdown:
                regions = ", ".join(
                    f"{k}={v}" for k, v in ct.region_breakdown.items() if v > 0
                )
                lines.append(f"    Regions: {regions}")

        if self.warnings:
            lines.append(f"  Warns  : {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    - {w}")

        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Control-flow signals                                                 #
# ------------------------------------------------------------------ #


class AbstentionSignal(Exception):
    """Raised by ``AutonomousMode`` when the agent self-abstains (F2).

    This is a deliberate control-flow signal, not a failure, mirroring
    the ``PluginHalt`` pattern.  It is raised from inside the execution
    mode and caught by ``Agent.execute()`` which converts it into an
    ``AgentResult`` with ``status=ResultStatus.ABSTAINED``.

    Abstention means the agent tried, exhausted its retry budget, and
    the ``Critic`` still refused to pass the candidate.  Rather than
    silently shipping a bad answer (the pre-F2 behaviour) the framework
    surfaces a structured outcome so callers can:

      * retry with a stronger model,
      * escalate to a human,
      * hand off to another agent,
      * or treat the task as failed.

    Args:
        best_candidate: The best candidate produced across all attempts.
            Still included so callers can inspect partial work.
        critique: The ``CritiqueResult`` from the final Critic pass
            (FAIL or UNCERTAIN-below-threshold).  Provides structured
            ``issues`` and ``suggestions``.
        reason: Human-readable explanation — defaults to
            ``critique.feedback``.
        abstain_reason: F5 — machine-readable abstention code (one of
            ``compute_budget.AbstainReason``).  ``None`` when raised
            from a path outside the budget controller (e.g. Best-of-N
            synthesis failure).  Surfaced to
            ``AgentResult.abstention_code``.
    """

    def __init__(
        self,
        best_candidate: Any,
        critique: CritiqueResult,
        reason: str | None = None,
        *,
        abstain_reason: AbstainReason | None = None,
    ) -> None:
        self.best_candidate = best_candidate
        self.critique = critique
        self.abstain_reason = abstain_reason  # F5
        self.reason = reason or (
            getattr(critique, "feedback", None)
            or f"Critic rejected candidate ({getattr(critique, 'verdict', 'unknown')})"
        )
        super().__init__(self.reason)
