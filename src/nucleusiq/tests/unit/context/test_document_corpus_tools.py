"""Tests for L5 document corpus agent tools."""

from __future__ import annotations

import pytest
from nucleusiq.agents.context.document_corpus_tools import (
    GET_DOCUMENT_CHUNK_TOOL_NAME,
    PROMOTE_DOCUMENT_CHUNK_TOOL_NAME,
    SEARCH_DOCUMENT_CORPUS_TOOL_NAME,
    build_document_corpus_tools,
    is_document_corpus_tool_name,
)
from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus
from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.workspace_tools import is_context_management_tool_name


def test_document_corpus_tool_names_are_context_management() -> None:
    assert is_document_corpus_tool_name(SEARCH_DOCUMENT_CORPUS_TOOL_NAME)
    assert is_context_management_tool_name(SEARCH_DOCUMENT_CORPUS_TOOL_NAME)


@pytest.mark.asyncio
async def test_search_and_get_chunk_tools() -> None:
    corpus = InMemoryDocumentCorpus(chunk_chars=120)
    dossier = InMemoryEvidenceDossier()
    corpus.index_document("doc:test", "Revenue from operations was Rs 100 crore.")

    tools = {t.name: t for t in build_document_corpus_tools(corpus, evidence=dossier)}
    assert SEARCH_DOCUMENT_CORPUS_TOOL_NAME in tools
    assert GET_DOCUMENT_CHUNK_TOOL_NAME in tools
    assert PROMOTE_DOCUMENT_CHUNK_TOOL_NAME in tools

    hits = await tools[SEARCH_DOCUMENT_CORPUS_TOOL_NAME].execute(query="revenue", limit=3)
    assert isinstance(hits, list)
    assert hits
    chunk_id = hits[0]["chunk_id"]

    chunk = await tools[GET_DOCUMENT_CHUNK_TOOL_NAME].execute(chunk_id=chunk_id)
    assert isinstance(chunk, dict)
    assert "revenue" in chunk["text"].lower()
