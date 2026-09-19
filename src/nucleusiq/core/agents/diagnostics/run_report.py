"""Run diagnostics — the always-on, shareable record of one ``Agent.execute()``.

Design (docs/design/AUTONOMOUS_HARNESS_HARDENING.md, WS-7):

* :class:`TerminationReason` is set at **every** exit path of every
  loop so a failed run never has to be reverse-engineered from an
  error string.
* :class:`RunRecorder` is a mutable collector that lives on the agent
  for the duration of one run.  Modes call its ``record_*`` methods;
  they are cheap counters and never raise.
* :class:`RunReport` is the frozen artefact attached to
  ``AgentResult.diagnostics``.  It is populated at **summary** level by
  default: counters, resolved configuration, decisions and the
  termination record — never task text, tool arguments or payloads —
  so it can be pasted into an issue as-is.

The analyzer that turns a report into findings + recommendations
lives in :mod:`nucleusiq.agents.diagnostics.analyzer`.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REPORT_SCHEMA_VERSION = 1


class TerminationReason(str, Enum):
    """Why a run stopped.  Exactly one is recorded per run."""

    COMPLETED = "completed"
    STRUCTURED_OUTPUT = "structured_output"
    SCHEMA_INVALID = "schema_invalid"
    TOOL_BUDGET = "tool_budget"
    CONTEXT_TOOL_BUDGET = "context_tool_budget"
    NO_PROGRESS = "no_progress"
    EMERGENCY_COMPACTION = "emergency_compaction"
    DEADLINE = "deadline"
    LLM_TIMEOUT = "llm_timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    CRITIC_ABSTAIN = "critic_abstain"
    TOKEN_CEILING = "token_ceiling"
    PLUGIN_HALT = "plugin_halt"
    EMPTY_RESPONSE = "empty_response"
    REFUSAL = "refusal"
    PREFLIGHT_DOWNGRADED = "preflight_downgraded"
    ERROR = "error"


class TerminationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: TerminationReason
    message: str = ""
    at_round: int | None = None
    at_tool_call: int | None = None
    elapsed_ms: float = 0.0


class RunCounters(BaseModel):
    """Counters that describe the shape of a run without any payload."""

    llm_calls: int = 0
    rounds: int = 0
    tool_calls_business: int = 0
    tool_calls_context: int = 0
    dedup_banners: int = 0
    recall_errors: int = 0
    tool_errors: int = 0
    empty_responses: int = 0
    no_progress_rounds: int = 0
    stalled_rounds: int = 0
    compactions_by_strategy: dict[str, int] = Field(default_factory=dict)
    emergency_count: int = 0
    unseen_evidence_evicted: int = 0
    writebacks: int = 0
    synthesis_runs: int = 0
    finalizer_runs: int = 0
    schema_validation_failures: int = 0
    critic_verdicts: dict[str, int] = Field(default_factory=dict)
    #: Critic passes in which the verifier saw less than the generator
    #: (package dropped/cut items AND the raw trace was capped) — I-10.
    critic_partial_views: int = 0
    #: FAIL verdicts reached on a partial view and downgraded to UNCERTAIN.
    critic_fail_downgraded: int = 0
    escalations: int = 0
    children_spawned: int = 0
    children_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_prompt_tokens_seen: int = 0


class TimelineEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    t_ms: float
    round: int | None = None
    kind: str
    detail: str = ""


class Finding(BaseModel):
    """One analyzer conclusion.  ``evidence`` names the report fields used."""

    model_config = ConfigDict(frozen=True)

    code: str
    severity: str  # "info" | "warning" | "critical"
    title: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    cause: str = ""
    fix_now: str = ""
    fix_release: str = ""


class RunReport(BaseModel):
    """Frozen diagnostics artefact for one run."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = REPORT_SCHEMA_VERSION
    framework_version: str = ""
    provider: str | None = None
    model: str | None = None
    mode: str = ""
    agent_name: str = ""
    task_id: str = ""
    status: str = ""

    config_resolved: dict[str, Any] = Field(default_factory=dict)
    preflight: dict[str, Any] | None = None
    decisions: dict[str, Any] = Field(default_factory=dict)
    counters: RunCounters = Field(default_factory=RunCounters)
    children: tuple[dict[str, Any], ...] = ()
    coverage: dict[str, Any] | None = None
    termination: TerminationRecord
    timeline: tuple[TimelineEvent, ...] = ()
    findings: tuple[Finding, ...] = ()
    recommendations: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    # Views                                                                #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def redacted(self) -> RunReport:
        """Return a copy safe to share outside the team.

        Summary level already excludes payloads.  Redaction additionally
        hashes resource identifiers in ``coverage`` / ``children`` and
        drops any free-text ``decisions`` previews.
        """
        import hashlib

        def _h(value: Any) -> str:
            return hashlib.sha1(
                str(value).encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:10]

        def _hash_lists(value: Any) -> Any:
            if isinstance(value, list):
                return [_h(v) for v in value]
            if isinstance(value, dict):
                return {k: _hash_lists(v) for k, v in value.items()}
            return value

        coverage = None
        if self.coverage:
            coverage = {k: _hash_lists(val) for k, val in self.coverage.items()}
        children = tuple(
            {
                **{k: v for k, v in child.items() if k != "objective_preview"},
                "resources": _hash_lists(child.get("resources", [])),
                "touched_resources": _hash_lists(child.get("touched_resources", [])),
            }
            for child in self.children
        )
        decisions = {
            k: v
            for k, v in self.decisions.items()
            if not k.endswith("_preview") and k != "reasoning"
        }
        if "classification" in decisions and isinstance(
            decisions["classification"], dict
        ):
            decisions["classification"] = {
                k: v
                for k, v in decisions["classification"].items()
                if k not in ("reasoning", "sub_task_objectives")
            }
        red = self.model_copy(
            update={
                "coverage": coverage,
                "children": children,
                "decisions": decisions,
                "task_id": _h(self.task_id),
                "findings": (),
                "recommendations": (),
            }
        )
        # Findings cite report fields verbatim, so recompute them from the
        # redacted data instead of carrying the originals over.
        if self.findings:
            red = red.analyze()
        return red

    def explain(self, *, redact: bool = False) -> str:
        """Markdown block for an issue report or a log line.

        ``redact=True`` renders :meth:`redacted` instead, so the block can
        be pasted outside the team without leaking resource names.
        """
        if redact:
            return self.redacted().explain()
        c = self.counters
        t = self.termination
        lines = [
            f"### NucleusIQ run report (schema v{self.schema_version})",
            "",
            f"- framework `{self.framework_version}` · mode `{self.mode}` · "
            f"provider `{self.provider or '?'}` · model `{self.model or '?'}`",
            f"- status **{self.status}** · termination **{t.reason.value}**"
            + (f" — {t.message}" if t.message else ""),
            f"- rounds {c.rounds} · llm calls {c.llm_calls} · tool calls "
            f"{c.tool_calls_business} (+{c.tool_calls_context} context-mgmt) · "
            f"dedup banners {c.dedup_banners} · recall errors {c.recall_errors}",
            f"- compactions {sum(c.compactions_by_strategy.values())} "
            f"(emergency {c.emergency_count}) · stalled rounds {c.stalled_rounds}"
            + (
                f" · unseen tool results evicted {c.unseen_evidence_evicted}"
                if c.unseen_evidence_evicted
                else ""
            ),
        ]
        if c.critic_verdicts or c.critic_partial_views:
            verdicts = ", ".join(f"{k} {v}" for k, v in c.critic_verdicts.items())
            lines.append(
                f"- critic verdicts {verdicts or 'none'}"
                + (
                    f" · partial-evidence views {c.critic_partial_views}"
                    f" (FAIL→UNCERTAIN downgrades {c.critic_fail_downgraded})"
                    if c.critic_partial_views
                    else ""
                )
            )
        cfg = self.config_resolved
        if cfg:
            lines.append(
                f"- window {cfg.get('context_window')} · reserve "
                f"{cfg.get('response_reserve')} · working budget "
                f"{cfg.get('optimal_budget')} · max_tool_calls "
                f"{cfg.get('max_tool_calls')} · tools {cfg.get('tool_count')} "
                f"({cfg.get('idempotent_tool_count')} idempotent)"
            )
        if self.preflight:
            lines.append(f"- preflight: {self.preflight}")
        cls = self.decisions.get("classification")
        if cls:
            lines.append(f"- classification: {cls}")
        if self.coverage:
            lines.append(f"- coverage: {self.coverage}")
        if self.children:
            lines.append(f"- children: {len(self.children)}")
            for child in self.children:
                lines.append(
                    f"  - `{child.get('id')}` status {child.get('status', '?')}"
                    f" · ended {child.get('termination_reason', '?')}"
                    f" · tools {child.get('tool_calls_business', '?')}"
                    f" (+{child.get('tool_calls_context', '?')} ctx)"
                    f" · cap {child.get('max_tool_calls', '?')}"
                    f" · window {child.get('window', '?')}"
                    f" · touched {len(child.get('touched_resources') or [])}"
                    f"/{len(child.get('resources') or [])}"
                )
        if self.findings:
            lines.append("")
            lines.append("**Findings**")
            for f in self.findings:
                lines.append(f"- [{f.severity}] `{f.code}` — {f.title}")
                if f.cause:
                    lines.append(f"  - cause: {f.cause}")
                if f.fix_now:
                    lines.append(f"  - now: {f.fix_now}")
                if f.fix_release:
                    lines.append(f"  - release: {f.fix_release}")
                if f.evidence:
                    lines.append(f"  - evidence: {f.evidence}")
        else:
            lines.append("")
            lines.append("**Findings**: none")
        return "\n".join(lines)

    def analyze(self) -> RunReport:
        """Return a copy with ``findings`` / ``recommendations`` (re)computed."""
        from nucleusiq.agents.diagnostics import analyzer as _an

        # Inline ``attach_findings`` so the copy stays this class identity.
        # Pyrefly sees this module as ``agents.diagnostics`` (search-path)
        # and the imported helper as ``nucleusiq.agents.diagnostics``.
        found = _an.analyze(self)  # pyrefly: ignore[bad-argument-type]
        return self.model_copy(
            update={
                "findings": tuple(found),
                "recommendations": tuple(_an.recommendations(found)),
            }
        )


class RunRecorder:
    """Mutable per-run collector.  Every method is safe to call blind."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.counters = RunCounters()
        self.decisions: dict[str, Any] = {}
        self.timeline: list[TimelineEvent] = []
        self.children: list[dict[str, Any]] = []
        self.coverage: dict[str, Any] | None = None
        self.preflight: dict[str, Any] | None = None
        self._termination: TerminationRecord | None = None
        self._current_round: int | None = None
        self._max_timeline = 500

    # -- time --------------------------------------------------------------

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    # -- counters ----------------------------------------------------------

    def record_round(self, round_no: int) -> None:
        self._current_round = round_no
        self.counters.rounds = max(self.counters.rounds, round_no)

    def record_llm_call(
        self, *, prompt_tokens: int | None = None, completion_tokens: int | None = None
    ) -> None:
        c = self.counters
        c.llm_calls += 1
        if prompt_tokens:
            c.prompt_tokens += int(prompt_tokens)
            c.max_prompt_tokens_seen = max(c.max_prompt_tokens_seen, int(prompt_tokens))
        if completion_tokens:
            c.completion_tokens += int(completion_tokens)

    def record_tool_call(self, *, context_management: bool) -> None:
        if context_management:
            self.counters.tool_calls_context += 1
        else:
            self.counters.tool_calls_business += 1

    def record_dedup_banner(self) -> None:
        self.counters.dedup_banners += 1

    def record_recall_error(self) -> None:
        self.counters.recall_errors += 1

    def record_tool_error(self) -> None:
        self.counters.tool_errors += 1

    def record_empty_response(self) -> None:
        self.counters.empty_responses += 1

    def record_stalled_round(self) -> None:
        self.counters.stalled_rounds += 1

    def record_no_progress_round(self) -> None:
        self.counters.no_progress_rounds += 1

    def record_compaction(self, strategy: str, *, emergency: bool = False) -> None:
        by = self.counters.compactions_by_strategy
        by[strategy] = by.get(strategy, 0) + 1
        if emergency:
            self.counters.emergency_count += 1

    def record_writeback(self) -> None:
        self.counters.writebacks += 1

    def record_synthesis_run(self) -> None:
        self.counters.synthesis_runs += 1

    def record_finalizer_run(self) -> None:
        self.counters.finalizer_runs += 1

    def record_schema_validation_failure(self) -> None:
        self.counters.schema_validation_failures += 1

    def record_critic_verdict(self, verdict: str) -> None:
        cv = self.counters.critic_verdicts
        cv[verdict] = cv.get(verdict, 0) + 1

    def record_escalation(self) -> None:
        self.counters.escalations += 1

    def record_child(self, info: dict[str, Any], *, failed: bool = False) -> None:
        self.counters.children_spawned += 1
        if failed:
            self.counters.children_failed += 1
        self.children.append(dict(info))

    # -- decisions / events -------------------------------------------------

    def record_decision(self, key: str, value: Any) -> None:
        self.decisions[key] = value

    def record_event(self, kind: str, detail: str = "") -> None:
        if len(self.timeline) >= self._max_timeline:
            return
        self.timeline.append(
            TimelineEvent(
                t_ms=self.elapsed_ms(),
                round=self._current_round,
                kind=kind,
                detail=detail[:200],
            )
        )

    # -- termination ---------------------------------------------------------

    def set_termination(
        self,
        reason: TerminationReason | str,
        message: str = "",
        *,
        at_tool_call: int | None = None,
    ) -> None:
        """Record why the run stopped.  Last writer wins.

        Loops call this at every exit; a retry that later completes
        overwrites the earlier abnormal reason, so the final record
        always describes the *last* exit.
        """
        if isinstance(reason, str):
            try:
                reason = TerminationReason(reason)
            except ValueError:
                reason = TerminationReason.ERROR
        self._termination = TerminationRecord(
            reason=reason,
            message=message[:500],
            at_round=self._current_round,
            at_tool_call=at_tool_call,
            elapsed_ms=self.elapsed_ms(),
        )
        self.record_event("termination", f"{reason.value}: {message}"[:200])

    @property
    def termination(self) -> TerminationRecord | None:
        return self._termination

    @property
    def termination_reason(self) -> TerminationReason | None:
        return self._termination.reason if self._termination else None

    # -- build ---------------------------------------------------------------

    def build(
        self,
        *,
        framework_version: str,
        provider: str | None,
        model: str | None,
        mode: str,
        agent_name: str,
        task_id: str,
        status: str,
        config_resolved: dict[str, Any],
        default_reason: TerminationReason,
        default_message: str = "",
    ) -> RunReport:
        termination = self._termination or TerminationRecord(
            reason=default_reason,
            message=default_message[:500],
            at_round=self._current_round,
            elapsed_ms=self.elapsed_ms(),
        )
        return RunReport(
            framework_version=framework_version,
            provider=provider,
            model=model,
            mode=mode,
            agent_name=agent_name,
            task_id=task_id,
            status=status,
            config_resolved=dict(config_resolved),
            preflight=self.preflight,
            decisions=dict(self.decisions),
            counters=self.counters.model_copy(deep=True),
            children=tuple(self.children),
            coverage=self.coverage,
            termination=termination,
            timeline=tuple(self.timeline),
        )


def recorder_for(agent: Any) -> RunRecorder | None:
    """Return the agent's live :class:`RunRecorder`, or ``None``.

    Tolerates mocks and legacy agents: only a real recorder is returned.
    """
    rec = getattr(agent, "_run_recorder", None)
    return rec if isinstance(rec, RunRecorder) else None
