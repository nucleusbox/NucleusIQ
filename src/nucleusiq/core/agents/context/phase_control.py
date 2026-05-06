"""Autonomous phase telemetry and evidence gate helpers.

L6 starts as framework-visible state, not a Task E controller. The controller
records phases and the gate evaluates generic evidence tags against the dossier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier

AgentPhase = Literal[
    "PLAN",
    "RESEARCH",
    "ORGANIZE_EVIDENCE",
    "SYNTHESIZE",
    "VALIDATE",
    "REFINE",
    "FINAL",
]


@dataclass(frozen=True)
class EvidenceGateDecision:
    """Result of checking evidence coverage."""

    passed: bool
    blocked: bool
    required_tags: tuple[str, ...] = ()
    present_tags: tuple[str, ...] = ()
    missing_tags: tuple[str, ...] = ()
    gap_tags: tuple[str, ...] = ()
    enforce: bool = False

    def to_dict(self) -> dict[str, bool | list[str]]:
        return {
            "passed": self.passed,
            "blocked": self.blocked,
            "required_tags": list(self.required_tags),
            "present_tags": list(self.present_tags),
            "missing_tags": list(self.missing_tags),
            "gap_tags": list(self.gap_tags),
            "enforce": self.enforce,
        }


@dataclass(frozen=True)
class PhaseStats:
    """Telemetry snapshot for phase control."""

    phase_current: str | None = None
    phase_transitions: tuple[str, ...] = ()
    phase_durations_ms: dict[str, float] = field(default_factory=dict)
    evidence_required_tags: tuple[str, ...] = ()
    evidence_missing_tags: tuple[str, ...] = ()
    evidence_gate_passed: bool | None = None
    evidence_gate_blocked: bool = False
    synthesis_used_package: bool = False
    critic_used_package: bool = False
    refiner_used_gaps: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_current": self.phase_current,
            "phase_transitions": list(self.phase_transitions),
            "phase_durations_ms": dict(self.phase_durations_ms),
            "evidence_required_tags": list(self.evidence_required_tags),
            "evidence_missing_tags": list(self.evidence_missing_tags),
            "evidence_gate_passed": self.evidence_gate_passed,
            "evidence_gate_blocked": self.evidence_gate_blocked,
            "synthesis_used_package": self.synthesis_used_package,
            "critic_used_package": self.critic_used_package,
            "refiner_used_gaps": self.refiner_used_gaps,
        }


class PhaseController:
    """Record ordered phase transitions and durations."""

    def __init__(self) -> None:
        self._current: str | None = None
        self._current_started_at: float | None = None
        self._transitions: list[str] = []
        self._durations_ms: dict[str, float] = {}
        self._last_gate: EvidenceGateDecision | None = None
        self.synthesis_used_package = False
        self.critic_used_package = False
        self.refiner_used_gaps = False

    def enter(self, phase: AgentPhase | str) -> None:
        now = time.perf_counter()
        self._close_current(now)
        phase_text = phase
        self._current = phase_text
        self._current_started_at = now
        self._transitions.append(phase_text)

    def finish(self) -> None:
        self._close_current(time.perf_counter())
        self._current_started_at = None

    def record_evidence_gate(self, decision: EvidenceGateDecision) -> None:
        self._last_gate = decision

    def stats(self) -> PhaseStats:
        durations = dict(self._durations_ms)
        if self._current is not None and self._current_started_at is not None:
            durations[self._current] = durations.get(self._current, 0.0) + (
                (time.perf_counter() - self._current_started_at) * 1000
            )
        gate = self._last_gate
        return PhaseStats(
            phase_current=self._current,
            phase_transitions=tuple(self._transitions),
            phase_durations_ms=durations,
            evidence_required_tags=gate.required_tags if gate else (),
            evidence_missing_tags=gate.missing_tags if gate else (),
            evidence_gate_passed=gate.passed if gate else None,
            evidence_gate_blocked=gate.blocked if gate else False,
            synthesis_used_package=self.synthesis_used_package,
            critic_used_package=self.critic_used_package,
            refiner_used_gaps=self.refiner_used_gaps,
        )

    def _close_current(self, now: float) -> None:
        if self._current is None or self._current_started_at is None:
            return
        elapsed = (now - self._current_started_at) * 1000
        self._durations_ms[self._current] = (
            self._durations_ms.get(self._current, 0.0) + elapsed
        )


class EvidenceGate:
    """Evaluate whether required evidence tags are covered."""

    def __init__(
        self,
        *,
        required_tags: tuple[str, ...] = (),
        enforce: bool = True,
    ) -> None:
        self.required_tags = required_tags
        self.enforce = enforce

    def evaluate(
        self,
        evidence: InMemoryEvidenceDossier,
        *,
        record_gaps: bool = False,
    ) -> EvidenceGateDecision:
        coverage = evidence.coverage(self.required_tags)
        missing = coverage.missing_tags
        if record_gaps:
            existing_gap_tags = {
                tag for item in evidence.list(status="gap") for tag in item.tags
            }
            for tag in missing:
                if tag in existing_gap_tags:
                    continue
                evidence.add_gap(
                    question=f"Need evidence for {tag}.",
                    reason="Required evidence tag is missing before synthesis.",
                    tags=(tag,),
                    metadata={"gate": "evidence_completeness"},
                )
        blocked = bool(missing and self.enforce)
        return EvidenceGateDecision(
            passed=not blocked,
            blocked=blocked,
            required_tags=coverage.required_tags,
            present_tags=coverage.present_tags,
            missing_tags=missing,
            gap_tags=coverage.gap_tags,
            enforce=self.enforce,
        )


__all__ = [
    "AgentPhase",
    "EvidenceGate",
    "EvidenceGateDecision",
    "PhaseController",
    "PhaseStats",
]
