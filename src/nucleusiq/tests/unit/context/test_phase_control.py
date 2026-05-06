"""L6 phase telemetry and evidence gate tests."""

from __future__ import annotations

from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.phase_control import EvidenceGate, PhaseController


def test_phase_controller_records_ordered_transitions() -> None:
    controller = PhaseController()

    controller.enter("PLAN")
    controller.enter("RESEARCH")
    controller.enter("ORGANIZE_EVIDENCE")
    controller.enter("SYNTHESIZE")
    controller.enter("VALIDATE")
    controller.enter("FINAL")
    controller.finish()

    stats = controller.stats()
    assert stats.phase_current == "FINAL"
    assert stats.phase_transitions == (
        "PLAN",
        "RESEARCH",
        "ORGANIZE_EVIDENCE",
        "SYNTHESIZE",
        "VALIDATE",
        "FINAL",
    )


def test_phase_controller_records_duration_per_phase() -> None:
    controller = PhaseController()

    controller.enter("PLAN")
    controller.enter("RESEARCH")
    controller.finish()

    stats = controller.stats()
    assert set(stats.phase_durations_ms) == {"PLAN", "RESEARCH"}
    assert all(value >= 0 for value in stats.phase_durations_ms.values())


def test_evidence_gate_blocks_synthesis_when_required_tags_missing() -> None:
    dossier = InMemoryEvidenceDossier()
    dossier.add_evidence(
        claim="TCS FY25 revenue found.",
        source_ref="obs:tcs",
        tags=("company:tcs", "metric:revenue"),
    )
    gate = EvidenceGate(required_tags=("company:tcs", "company:wipro"))

    decision = gate.evaluate(dossier)

    assert decision.passed is False
    assert decision.blocked is True
    assert decision.missing_tags == ("company:wipro",)


def test_evidence_gate_allows_synthesis_when_coverage_complete() -> None:
    dossier = InMemoryEvidenceDossier()
    dossier.add_evidence(
        claim="TCS FY25 revenue found.",
        source_ref="obs:tcs",
        tags=("company:tcs", "metric:revenue"),
    )
    gate = EvidenceGate(required_tags=("company:tcs",))

    decision = gate.evaluate(dossier)

    assert decision.passed is True
    assert decision.blocked is False
    assert decision.missing_tags == ()


def test_evidence_gate_records_gaps_in_dossier() -> None:
    dossier = InMemoryEvidenceDossier()
    gate = EvidenceGate(required_tags=("metric:margin",))

    decision = gate.evaluate(dossier, record_gaps=True)

    assert decision.blocked is True
    assert dossier.stats().gap_count == 1
    assert dossier.list(status="gap")[0].tags == ("metric:margin",)


def test_phase_controller_fail_open_when_disabled() -> None:
    dossier = InMemoryEvidenceDossier()
    gate = EvidenceGate(required_tags=("metric:revenue",), enforce=False)

    decision = gate.evaluate(dossier)

    assert decision.passed is True
    assert decision.blocked is False
    assert decision.missing_tags == ("metric:revenue",)
