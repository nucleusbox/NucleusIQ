"""L4.5 context state activation tests."""

from __future__ import annotations

from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus
from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.state_activator import ContextStateActivator
from nucleusiq.agents.context.workspace import InMemoryWorkspace


def test_activator_ignores_framework_context_tools() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    activator = ContextStateActivator(workspace=workspace, evidence=dossier)

    metrics = activator.activate_tool_result(
        tool_name="add_evidence",
        tool_call_id="call_1",
        tool_result={"claim": "TCS revenue was found."},
    )

    assert metrics.tool_results_seen == 1
    assert metrics.activation_skipped_framework_tools == 1
    assert workspace.stats().entry_count == 0
    assert dossier.stats().item_count == 0


def test_activator_records_workspace_progress_for_business_tool_result() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    activator = ContextStateActivator(workspace=workspace, evidence=dossier)

    activator.activate_tool_result(
        tool_name="read_annual_report_excerpt",
        tool_call_id="call_tcs",
        tool_result="TCS FY25 annual report excerpt. Revenue from operations was Rs 255,324 crore.",
    )

    entries = workspace.list()
    assert len(entries) == 1
    assert entries[0].source_refs == ("obs:read_annual_report_excerpt:call_tcs",)
    assert "read_annual_report_excerpt" in entries[0].content


def test_activator_promotes_business_tool_result_to_evidence() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    activator = ContextStateActivator(workspace=workspace, evidence=dossier)

    metrics = activator.activate_tool_result(
        tool_name="read_annual_report_excerpt",
        tool_call_id="call_tcs",
        tool_result=(
            "TCS FY25 annual report excerpt. "
            "Revenue from operations was Rs 255,324 crore. "
            "Operating margin was 24.3%."
        ),
    )

    items = dossier.list()
    assert metrics.evidence_items_promoted >= 1
    assert any("Revenue from operations" in item.claim for item in items)
    assert all(item.source.tool_name == "read_annual_report_excerpt" for item in items)
    assert all(
        item.source.ref == "obs:read_annual_report_excerpt:call_tcs" for item in items
    )


def test_activator_records_gap_when_required_tag_missing() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    activator = ContextStateActivator(
        workspace=workspace,
        evidence=dossier,
        required_tags=("metric:revenue", "metric:margin"),
    )

    activator.activate_tool_result(
        tool_name="read_annual_report_excerpt",
        tool_call_id="call_tcs",
        tool_result="TCS FY25 annual report excerpt. Revenue from operations was Rs 255,324 crore.",
    )

    gaps = dossier.list(status="gap")
    assert len(gaps) == 1
    assert "metric:margin" in gaps[0].tags


def test_activator_respects_content_limits() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    activator = ContextStateActivator(
        workspace=workspace,
        evidence=dossier,
        max_inspect_chars=80,
    )

    activator.activate_tool_result(
        tool_name="read_annual_report_excerpt",
        tool_call_id="call_tcs",
        tool_result="x" * 5000,
    )

    assert workspace.stats().total_chars < 1000
    assert dossier.stats().item_count == 0


def test_activator_light_ingests_read_tool_without_financial_keywords() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    corpus = InMemoryDocumentCorpus(chunk_chars=400)
    activator = ContextStateActivator(
        workspace=workspace,
        evidence=dossier,
        document_corpus=corpus,
        ingest_min_chars=200,
    )
    body = (
        "This section discusses corporate governance and board composition. "
        "No financial tables appear on these pages. "
    ) * 15

    activator.activate_tool_result(
        tool_name="read_annual_report_excerpt",
        tool_call_id="call_mdna",
        tool_args={"filename": "ExampleCo_FY25.pdf"},
        tool_result=body,
    )

    assert workspace.stats().entry_count == 1
    assert corpus.stats().documents_indexed >= 1
    assert dossier.stats().supported_count == 0
    assert activator.metrics.light_ingests == 1
    assert activator.metrics.tool_results_activated == 1


def test_activator_skips_short_read_when_below_ingest_min_chars() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    corpus = InMemoryDocumentCorpus()
    activator = ContextStateActivator(
        workspace=workspace,
        evidence=dossier,
        document_corpus=corpus,
        ingest_min_chars=200,
    )
    # No metric keywords or digits — only length decides light ingest eligibility.
    activator.activate_tool_result(
        tool_name="read_annual_report_excerpt",
        tool_call_id="c1",
        tool_result="z" * 199,
    )

    assert workspace.stats().entry_count == 0
    assert corpus.stats().documents_indexed == 0
    assert activator.metrics.activation_skipped_non_evidence_tools == 1


def test_activator_indexes_into_document_corpus_when_configured() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    corpus = InMemoryDocumentCorpus(chunk_chars=200)
    activator = ContextStateActivator(
        workspace=workspace,
        evidence=dossier,
        document_corpus=corpus,
        max_corpus_index_chars=50_000,
    )

    activator.activate_tool_result(
        tool_name="read_pdf_excerpt",
        tool_call_id="call_1",
        tool_args={"filename": "ACME_FY25.pdf"},
        tool_result=(
            "ACME FY25 annual report excerpt. Revenue from operations was Rs 100 crore."
        ),
    )

    assert corpus.stats().documents_indexed >= 1
    hits = corpus.search("revenue", limit=2)
    assert hits


def test_activator_includes_company_tag_from_tool_args() -> None:
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    activator = ContextStateActivator(workspace=workspace, evidence=dossier)

    activator.activate_tool_result(
        tool_name="read_report",
        tool_call_id="call_acme",
        tool_args={"company": "ACME Corp"},
        tool_result="ACME FY25 report. Revenue from operations was Rs 10 crore.",
    )

    items = dossier.list(status="supported")
    assert items
    assert any("company:acme_corp" in item.tags for item in items)


def test_activator_does_not_treat_said_as_ai_keyword() -> None:
    """``ai`` substring inside unrelated words must not trigger AI topic."""
    workspace = InMemoryWorkspace()
    dossier = InMemoryEvidenceDossier()
    activator = ContextStateActivator(
        workspace=workspace,
        evidence=dossier,
        ingest_min_chars=200,
    )
    body = ("zzzzzzzz dolor lorem block paragraph text filler content. ") * 20

    metrics = activator.activate_tool_result(
        tool_name="read_excerpt",
        tool_call_id="c1",
        tool_result=body,
    )

    assert dossier.stats().supported_count == 0
    assert metrics.evidence_items_promoted == 0
    assert metrics.light_ingests == 1
