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
