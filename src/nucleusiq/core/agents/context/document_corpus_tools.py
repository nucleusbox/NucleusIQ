"""Agent-facing tools for the run-local L5 document corpus."""

from __future__ import annotations

from typing import Any

from nucleusiq.agents.context.document_search import (
    ChunkHit,
    DocumentChunk,
    DocumentRef,
    InMemoryDocumentCorpus,
)
from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
from nucleusiq.agents.context.policy import ContextPolicy
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.decorators import DecoratedTool

SEARCH_DOCUMENT_CORPUS_TOOL_NAME = "search_document_corpus"
GET_DOCUMENT_CHUNK_TOOL_NAME = "get_document_chunk"
LIST_INDEXED_DOCUMENTS_TOOL_NAME = "list_indexed_documents"
PROMOTE_DOCUMENT_CHUNK_TOOL_NAME = "promote_document_chunk_to_evidence"

_DOCUMENT_CORPUS_TOOL_NAMES: frozenset[str] = frozenset(
    {
        SEARCH_DOCUMENT_CORPUS_TOOL_NAME,
        GET_DOCUMENT_CHUNK_TOOL_NAME,
        LIST_INDEXED_DOCUMENTS_TOOL_NAME,
        PROMOTE_DOCUMENT_CHUNK_TOOL_NAME,
    }
)

__all__ = [
    "SEARCH_DOCUMENT_CORPUS_TOOL_NAME",
    "GET_DOCUMENT_CHUNK_TOOL_NAME",
    "LIST_INDEXED_DOCUMENTS_TOOL_NAME",
    "PROMOTE_DOCUMENT_CHUNK_TOOL_NAME",
    "build_document_corpus_tools",
    "is_document_corpus_tool_name",
]


def is_document_corpus_tool_name(tool_name: str | None) -> bool:
    """Return True when ``tool_name`` is an auto-injected document corpus tool."""
    return tool_name in _DOCUMENT_CORPUS_TOOL_NAMES if tool_name else False


def _format_document_corpus_error(message: str) -> str:
    return f"[document_corpus_error: {message}]"


def _normalize_tags(tags: list | tuple | str | None) -> tuple[str, ...]:
    if tags is None:
        return ()
    if isinstance(tags, str):
        return (tags,) if tags.strip() else ()
    if isinstance(tags, (list, tuple)):
        return tuple(str(t) for t in tags if isinstance(t, str) and t.strip())
    return ()


def _ref_to_dict(ref: DocumentRef) -> dict[str, Any]:
    return {
        "id": ref.id,
        "title": ref.title,
        "metadata": dict(ref.metadata),
        "chunk_count": ref.chunk_count,
        "char_count": ref.char_count,
    }


def _hit_to_dict(hit: ChunkHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "score": hit.score,
        "preview": hit.preview,
        "title": hit.title,
        "locator": hit.locator,
        "metadata": dict(hit.metadata),
    }


def _chunk_to_dict(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "title": chunk.title,
        "locator": chunk.locator,
        "metadata": dict(chunk.metadata),
    }


def build_document_corpus_tools(
    corpus: InMemoryDocumentCorpus,
    *,
    evidence: InMemoryEvidenceDossier | None = None,
) -> list[BaseTool]:
    """Build document corpus tools bound to one execution's corpus (and optionally dossier)."""

    async def search_document_corpus(
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]] | str:
        """Lexical search over chunks indexed for this run (bounded ``limit``)."""
        if not isinstance(limit, int) or limit < 1:
            return _format_document_corpus_error("limit must be a positive integer")
        hits = corpus.search(query.strip(), limit=limit)
        return [_hit_to_dict(h) for h in hits]

    async def get_document_chunk(chunk_id: str) -> dict[str, Any] | str:
        """Fetch one chunk by id (from search results)."""
        chunk = corpus.get_chunk(chunk_id.strip())
        if chunk is None:
            return _format_document_corpus_error(f"chunk_id {chunk_id!r} not found")
        return _chunk_to_dict(chunk)

    async def list_indexed_documents() -> list[dict[str, Any]]:
        """List document refs currently indexed in the run-local corpus."""
        return [_ref_to_dict(ref) for ref in corpus.list_documents()]

    async def promote_document_chunk_to_evidence(
        chunk_id: str,
        claim: str,
        tags: list | None = None,
        confidence: float = 0.75,
    ) -> dict[str, Any] | str:
        """Promote one corpus chunk into the evidence dossier as a supported fact."""
        if evidence is None:
            return _format_document_corpus_error(
                "evidence dossier is not bound to document corpus tools"
            )
        try:
            item = corpus.promote_chunk_to_evidence(
                chunk_id.strip(),
                evidence,
                claim=claim.strip(),
                tags=_normalize_tags(tags),
                confidence=confidence,
                status="supported",
            )
        except KeyError:
            return _format_document_corpus_error(f"chunk_id {chunk_id!r} not found")
        except Exception as exc:
            return _format_document_corpus_error(str(exc))
        return {
            "id": item.id,
            "claim": item.claim,
            "tags": list(item.tags),
            "source_ref": item.source.ref,
        }

    tools: list[BaseTool] = [
        DecoratedTool(
            search_document_corpus,
            tool_name=SEARCH_DOCUMENT_CORPUS_TOOL_NAME,
            tool_description=(search_document_corpus.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            get_document_chunk,
            tool_name=GET_DOCUMENT_CHUNK_TOOL_NAME,
            tool_description=(get_document_chunk.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
        DecoratedTool(
            list_indexed_documents,
            tool_name=LIST_INDEXED_DOCUMENTS_TOOL_NAME,
            tool_description=(list_indexed_documents.__doc__ or "").strip(),
            context_policy=ContextPolicy.EPHEMERAL,
        ),
    ]
    if evidence is not None:
        tools.append(
            DecoratedTool(
                promote_document_chunk_to_evidence,
                tool_name=PROMOTE_DOCUMENT_CHUNK_TOOL_NAME,
                tool_description=(
                    promote_document_chunk_to_evidence.__doc__ or ""
                ).strip(),
                context_policy=ContextPolicy.EPHEMERAL,
            )
        )
    return tools
