"""L4 synthesis package builder tests."""

from __future__ import annotations

import asyncio

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.synthesis_package import build_synthesis_package
from nucleusiq.agents.context.workspace import InMemoryWorkspace


def _state() -> tuple[InMemoryWorkspace, InMemoryEvidenceDossier]:
    workspace = InMemoryWorkspace()
    workspace.write_note(
        title="Research progress",
        content="Checked TCS and Infosys annual report excerpts.",
    )
    dossier = InMemoryEvidenceDossier()
    dossier.add_evidence(
        claim="TCS FY25 revenue was 255,324 Cr.",
        source_ref="obs:tcs",
        title="TCS FY25 Annual Report",
        locator="p.84",
        tags=("company:tcs", "metric:revenue"),
    )
    dossier.add_gap(
        question="Need Wipro FY25 revenue.",
        reason="Wipro annual report excerpt not collected yet.",
        tags=("company:wipro", "metric:revenue"),
    )
    return workspace, dossier


def test_synthesis_package_includes_task_and_constraints() -> None:
    workspace, dossier = _state()

    package = build_synthesis_package(
        task="Compare TCS, Infosys, Wipro, and HCLTech.",
        output_shape="Use a table and recommendations.",
        workspace=workspace,
        evidence=dossier,
        max_chars=4000,
    )

    assert "Compare TCS" in package.text
    assert "Use a table" in package.text
    assert package.metadata["char_count"] <= 4000


def test_synthesis_package_includes_supported_evidence_and_gaps() -> None:
    workspace, dossier = _state()

    package = build_synthesis_package(
        task="Task",
        output_shape="Output",
        workspace=workspace,
        evidence=dossier,
        max_chars=4000,
    )

    assert "TCS FY25 revenue was 255,324 Cr." in package.text
    assert "Need Wipro FY25 revenue." in package.text
    assert "obs:tcs" in package.text


def test_synthesis_package_includes_workspace_summary() -> None:
    workspace, dossier = _state()

    package = build_synthesis_package(
        task="Task",
        output_shape="Output",
        workspace=workspace,
        evidence=dossier,
        max_chars=4000,
    )

    assert "Research progress" in package.text


def test_synthesis_package_omits_low_priority_state_when_tight() -> None:
    workspace, dossier = _state()
    workspace.write_artifact(title="Large notes", content="x " * 2000)

    package = build_synthesis_package(
        task="Task",
        output_shape="Output",
        workspace=workspace,
        evidence=dossier,
        recalled_snippets=("low priority snippet " * 200,),
        max_chars=450,
    )

    assert len(package.text) <= 450
    assert "TCS FY25 revenue" in package.text
    assert package.metadata["omitted_sections"]


def test_synthesis_package_is_deterministic() -> None:
    workspace, dossier = _state()

    first = build_synthesis_package(
        task="Task",
        output_shape="Output",
        workspace=workspace,
        evidence=dossier,
        max_chars=4000,
    )
    second = build_synthesis_package(
        task="Task",
        output_shape="Output",
        workspace=workspace,
        evidence=dossier,
        max_chars=4000,
    )

    assert first.text == second.text
    assert first.metadata == second.metadata


def test_synthesis_package_respects_max_chars() -> None:
    workspace, dossier = _state()

    package = build_synthesis_package(
        task="Task " * 100,
        output_shape="Output " * 100,
        workspace=workspace,
        evidence=dossier,
        max_chars=300,
    )

    assert len(package.text) <= 300
    assert package.metadata["char_count"] <= 300


def _many_notes(n: int, *, chars: int) -> InMemoryWorkspace:
    workspace = InMemoryWorkspace()
    for i in range(1, n + 1):
        workspace.write_note(
            title=f"Observed read_document {i:02d}",
            content=f"[office/invoice_{i:02d}.pdf] " + ("fact " * (chars // 5)),
        )
    return workspace


class TestHonestOmission:
    """I-10: a cut list must never read as a shorter complete list."""

    def test_full_fit_is_marked_complete(self) -> None:
        workspace, dossier = _state()
        package = build_synthesis_package(
            task="Task",
            output_shape="Out",
            workspace=workspace,
            evidence=dossier,
            max_chars=8_000,
        )
        assert package.complete is True
        assert package.metadata["complete"] is True
        assert package.metadata["omitted_items"] == {
            "Supported Evidence": 0,
            "Known Gaps And Conflicts": 0,
            "Selected Workspace Notes": 0,
            "Source Index": 0,
        }
        assert package.visibility_note() == ""
        assert "omitted for space" not in package.text

    def test_notes_shrink_before_any_note_is_dropped(self) -> None:
        # 9 notes of ~900 chars = 8.1K; give the package 5K.  Every note must
        # still be present (shortened), so the reader sees all nine sources.
        package = build_synthesis_package(
            task="Task", workspace=_many_notes(9, chars=900), max_chars=5_000
        )
        assert len(package.text) <= 5_000
        for i in range(1, 10):
            assert f"[office/invoice_{i:02d}.pdf]" in package.text
        assert package.metadata["included_items"]["Selected Workspace Notes"] == 9
        assert package.metadata["omitted_items"]["Selected Workspace Notes"] == 0
        assert package.metadata["cut_items"]["Selected Workspace Notes"] == 9
        assert package.complete is False
        assert "9 shortened" in package.visibility_note()

    def test_dropped_evidence_items_are_announced(self) -> None:
        dossier = InMemoryEvidenceDossier()
        for i in range(1, 21):
            dossier.add_evidence(
                claim=f"Invoice {i:02d} total is {i * 1000} USD. " + "x" * 120,
                source_ref=f"obs:read:{i}",
                tags=("metric:total",),
            )
        package = build_synthesis_package(
            task="Task", evidence=dossier, max_chars=1_800
        )
        assert len(package.text) <= 1_800
        omitted = package.metadata["omitted_items"]["Supported Evidence"]
        included = package.metadata["included_items"]["Supported Evidence"]
        assert omitted > 0 and included > 0 and included + omitted == 20
        assert (
            f"[{omitted} more evidence items omitted for space — this list is "
            "INCOMPLETE; absence here is not absence of evidence]"
        ) in package.text
        # Items are whole: the last included claim is not cut mid-sentence.
        body = package.text.split("## Supported Evidence\n", 1)[1]
        kept = [ln for ln in body.splitlines() if ln.startswith("- supported")]
        assert len(kept) == included
        assert all(ln.endswith("[metric:total]") for ln in kept)
        assert package.complete is False
        note = package.visibility_note()
        assert note.startswith("EVIDENCE VISIBILITY: this package is PARTIAL")
        assert f"Supported Evidence: {included} of 20 shown" in note

    def test_coverage_section_states_harness_facts(self) -> None:
        package = build_synthesis_package(
            task="Task",
            coverage={
                "resources": ["a.pdf", "b.pdf", "c.pdf"],
                "touched": ["a.pdf", "b.pdf"],
                "unprocessed": ["c.pdf"],
            },
            max_chars=4_000,
        )
        assert "## Resources Processed" in package.text
        assert "2 of 3 declared resources were read by tools" in package.text
        assert "- read: a.pdf, b.pdf" in package.text
        assert "- NOT read: c.pdf" in package.text

    def test_coverage_is_omitted_when_no_resources(self) -> None:
        package = build_synthesis_package(
            task="Task", coverage={"resources": [], "touched": []}, max_chars=4_000
        )
        assert "Resources Processed" not in package.text

    def test_section_with_no_room_is_listed_as_omitted(self) -> None:
        package = build_synthesis_package(
            task="Task " * 40, workspace=_many_notes(3, chars=800), max_chars=260
        )
        assert len(package.text) <= 260
        assert "Selected Workspace Notes" in package.metadata["omitted_sections"]
        assert package.complete is False


def test_synthesis_package_passes_through_context_engine() -> None:
    workspace, dossier = _state()
    package = build_synthesis_package(
        task="Task",
        output_shape="Output",
        workspace=workspace,
        evidence=dossier,
        max_chars=2000,
    )
    engine = ContextEngine(ContextConfig(max_context_tokens=4000))
    messages = [ChatMessage(role="user", content=package.text)]

    prepared = asyncio.run(engine.prepare(messages))

    assert prepared
    assert prepared[0].content
