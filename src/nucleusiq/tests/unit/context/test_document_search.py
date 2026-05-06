"""L5 document search and chunk retrieval tests."""

from __future__ import annotations

from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus
from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier


def test_document_index_splits_text_into_stable_chunks() -> None:
    corpus = InMemoryDocumentCorpus(chunk_chars=80, chunk_overlap=10)

    first = corpus.index_document(
        "doc:tcs",
        "TCS FY25 revenue from operations was Rs 255,324 crore. " * 5,
        metadata={"company": "tcs", "year": "fy25"},
        title="TCS FY25 Annual Report",
    )
    second = corpus.index_document(
        "doc:tcs",
        "TCS FY25 revenue from operations was Rs 255,324 crore. " * 5,
        metadata={"company": "tcs", "year": "fy25"},
        title="TCS FY25 Annual Report",
    )

    assert first.id == second.id
    assert first.chunk_count == second.chunk_count
    assert [c.id for c in corpus.chunks_for_document("doc:tcs")]


def test_document_search_returns_relevant_chunks_before_full_document() -> None:
    corpus = InMemoryDocumentCorpus(chunk_chars=120)
    full_text = (
        "Risk section discusses macro uncertainty. "
        "FY25 revenue from operations was Rs 255,324 crore for TCS. "
        "People section discusses hiring and learning."
    )
    corpus.index_document("doc:tcs", full_text, metadata={"company": "tcs"})

    hits = corpus.search("FY25 revenue", limit=2)

    assert len(hits) >= 1
    assert "revenue" in hits[0].preview.lower()
    assert len(hits[0].text) < len(full_text)


def test_document_search_respects_filters() -> None:
    corpus = InMemoryDocumentCorpus(chunk_chars=120)
    corpus.index_document(
        "doc:tcs",
        "TCS FY25 revenue from operations was Rs 255,324 crore.",
        metadata={"company": "tcs", "year": "fy25"},
    )
    corpus.index_document(
        "doc:wipro",
        "Wipro FY25 revenue from operations was Rs 89,088 crore.",
        metadata={"company": "wipro", "year": "fy25"},
    )

    hits = corpus.search("revenue", filters={"company": "wipro"})

    assert hits
    assert {hit.metadata["company"] for hit in hits} == {"wipro"}


def test_get_chunk_returns_text_and_provenance() -> None:
    corpus = InMemoryDocumentCorpus(chunk_chars=120)
    corpus.index_document(
        "doc:infosys",
        "Infosys FY25 revenue from operations was Rs 162,990 crore.",
        metadata={"company": "infosys"},
        title="Infosys FY25 Annual Report",
    )
    hit = corpus.search("Infosys revenue", limit=1)[0]

    chunk = corpus.get_chunk(hit.chunk_id)

    assert chunk is not None
    assert chunk.id == hit.chunk_id
    assert chunk.document_id == "doc:infosys"
    assert chunk.title == "Infosys FY25 Annual Report"
    assert "Infosys FY25 revenue" in chunk.text


def test_document_search_result_is_bounded() -> None:
    corpus = InMemoryDocumentCorpus(chunk_chars=500, preview_chars=60)
    corpus.index_document("doc:large", "revenue " * 1000)

    hits = corpus.search("revenue", limit=3)

    assert len(hits) == 3
    assert all(len(hit.preview) <= 60 for hit in hits)


def test_selected_chunk_can_be_promoted_to_evidence() -> None:
    corpus = InMemoryDocumentCorpus(chunk_chars=120)
    dossier = InMemoryEvidenceDossier()
    corpus.index_document(
        "doc:tcs",
        "TCS FY25 revenue from operations was Rs 255,324 crore.",
        metadata={"company": "tcs", "year": "fy25"},
        title="TCS FY25 Annual Report",
    )
    hit = corpus.search("TCS revenue", limit=1)[0]

    item = corpus.promote_chunk_to_evidence(
        hit.chunk_id,
        dossier,
        claim="TCS FY25 revenue from operations was Rs 255,324 crore.",
        tags=("company:tcs", "metric:revenue", "year:fy25"),
    )

    assert item.source.ref == f"chunk:{hit.chunk_id}"
    assert item.source.locator == hit.locator
    assert dossier.stats().supported_count == 1
    assert corpus.stats().chunk_promotions_to_evidence == 1


def test_document_search_no_live_provider_required() -> None:
    corpus = InMemoryDocumentCorpus()

    corpus.index_document("doc:local", "Local text only.")

    assert corpus.stats().documents_indexed == 1
