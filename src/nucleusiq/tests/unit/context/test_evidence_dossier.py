"""L3 evidence dossier state tests."""

from __future__ import annotations

from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier


def test_evidence_dossier_starts_empty() -> None:
    dossier = InMemoryEvidenceDossier()

    assert dossier.list() == []
    assert dossier.stats().item_count == 0
    assert dossier.stats().gap_count == 0


def test_add_supported_evidence_round_trip() -> None:
    dossier = InMemoryEvidenceDossier()

    item = dossier.add_evidence(
        claim="TCS FY25 revenue was 255,324 Cr.",
        source_ref="obs:annual_report:tcs",
        title="TCS FY25 Annual Report",
        locator="p.84",
        confidence=0.95,
        tags=("company:tcs", "metric:revenue"),
        quote="Revenue from operations 255,324",
    )

    loaded = dossier.list()[0]
    assert loaded == item
    assert loaded.status == "supported"
    assert loaded.source.ref == "obs:annual_report:tcs"
    assert loaded.source.locator == "p.84"
    assert loaded.confidence == 0.95
    assert loaded.tags == ("company:tcs", "metric:revenue")


def test_add_gap_creates_first_class_gap() -> None:
    dossier = InMemoryEvidenceDossier()

    gap = dossier.add_gap(
        question="Need Wipro FY25 operating margin.",
        reason="Annual report excerpt not read yet.",
        tags=("company:wipro", "metric:margin"),
    )

    assert gap.status == "gap"
    assert gap.claim == "Need Wipro FY25 operating margin."
    assert gap.metadata["reason"] == "Annual report excerpt not read yet."
    assert dossier.stats().gap_count == 1


def test_list_evidence_filters_by_status_and_tags() -> None:
    dossier = InMemoryEvidenceDossier()
    supported = dossier.add_evidence(
        claim="Infosys revenue found.",
        source_ref="obs:infosys",
        tags=("company:infosys", "metric:revenue"),
    )
    dossier.add_gap(
        question="Need Infosys risk details.",
        reason="Risk section missing.",
        tags=("company:infosys", "topic:risk"),
    )

    assert dossier.list(status="supported", tags=("company:infosys",)) == [supported]
    assert dossier.list(status="gap", tags=("topic:risk",))[0].status == "gap"


def test_query_evidence_matches_claims_and_tags() -> None:
    dossier = InMemoryEvidenceDossier()
    revenue = dossier.add_evidence(
        claim="HCLTech revenue increased year over year.",
        source_ref="obs:hcltech",
        tags=("company:hcltech", "metric:revenue"),
    )
    dossier.add_evidence(
        claim="TCS margin was stable.",
        source_ref="obs:tcs",
        tags=("company:tcs", "metric:margin"),
    )

    assert dossier.query("hcltech revenue") == [revenue]


def test_evidence_summary_is_bounded_and_deterministic() -> None:
    dossier = InMemoryEvidenceDossier()
    dossier.add_evidence(
        claim="TCS has large revenue from operations.",
        source_ref="obs:tcs",
        tags=("company:tcs",),
    )
    dossier.add_gap(
        question="Need peer risk comparison.",
        reason="Missing risk notes.",
        tags=("topic:risk",),
    )

    first = dossier.summarize(max_chars=120)
    second = dossier.summarize(max_chars=120)

    assert first == second
    assert len(first) <= 120
    assert "evidence dossier" in first


def test_coverage_reports_missing_required_tags() -> None:
    dossier = InMemoryEvidenceDossier()
    dossier.add_evidence(
        claim="TCS revenue found.",
        source_ref="obs:tcs",
        tags=("company:tcs", "metric:revenue"),
    )

    coverage = dossier.coverage(("company:tcs", "company:wipro"))

    assert coverage.is_complete is False
    assert coverage.present_tags == ("company:tcs",)
    assert coverage.missing_tags == ("company:wipro",)


def test_conflicting_evidence_is_preserved() -> None:
    dossier = InMemoryEvidenceDossier()
    supported = dossier.add_evidence(
        claim="TCS margin improved.",
        source_ref="obs:tcs:one",
        tags=("company:tcs", "metric:margin"),
    )
    conflict = dossier.add_evidence(
        claim="TCS margin declined.",
        source_ref="obs:tcs:two",
        tags=("company:tcs", "metric:margin"),
        status="conflict",
    )

    assert dossier.list(status="supported") == [supported]
    assert dossier.list(status="conflict") == [conflict]
    assert dossier.stats().conflict_count == 1
