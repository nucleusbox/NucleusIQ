"""Generic in-memory document search and chunk retrieval.

This is L5 framework infrastructure: it indexes caller-provided text into
bounded chunks with provenance, then provides deterministic lexical search.
It intentionally contains no PDF parsing, no embeddings, and no research-task
logic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from nucleusiq.agents.context.evidence import EvidenceItem, InMemoryEvidenceDossier

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_:-]*")


@dataclass(frozen=True)
class DocumentRef:
    """One indexed document."""

    id: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    char_count: int = 0


@dataclass(frozen=True)
class DocumentChunk:
    """A bounded, provenance-carrying document chunk."""

    id: str
    document_id: str
    text: str
    title: str | None = None
    locator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkHit:
    """One search result."""

    chunk_id: str
    document_id: str
    score: float
    preview: str
    text: str
    title: str | None = None
    locator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentSearchStats:
    """Telemetry snapshot for L5 document search."""

    backend: str = "memory"
    documents_indexed: int = 0
    chunks_indexed: int = 0
    document_search_count: int = 0
    document_chunks_returned: int = 0
    document_search_chars_returned: int = 0
    chunk_retrieval_count: int = 0
    chunk_promotions_to_evidence: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "backend": self.backend,
            "documents_indexed": self.documents_indexed,
            "chunks_indexed": self.chunks_indexed,
            "document_search_count": self.document_search_count,
            "document_chunks_returned": self.document_chunks_returned,
            "document_search_chars_returned": self.document_search_chars_returned,
            "chunk_retrieval_count": self.chunk_retrieval_count,
            "chunk_promotions_to_evidence": self.chunk_promotions_to_evidence,
        }


class InMemoryDocumentCorpus:
    """Small deterministic document corpus for run-local retrieval."""

    def __init__(
        self,
        *,
        chunk_chars: int = 1_500,
        chunk_overlap: int = 150,
        preview_chars: int = 240,
    ) -> None:
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_chars:
            chunk_overlap = max(0, chunk_chars // 10)
        self._chunk_chars = chunk_chars
        self._chunk_overlap = chunk_overlap
        self._preview_chars = preview_chars
        self._documents: dict[str, DocumentRef] = {}
        self._chunks: dict[str, DocumentChunk] = {}
        self._document_chunks: dict[str, list[str]] = {}
        self._search_count = 0
        self._chunks_returned = 0
        self._chars_returned = 0
        self._chunk_retrieval_count = 0
        self._chunk_promotions = 0

    def index_document(
        self,
        document_id: str,
        text: str,
        *,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRef:
        """Index text into stable chunks and return its document ref."""
        meta = dict(metadata or {})
        chunks = self._split_chunks(document_id, text, title=title, metadata=meta)
        old_ids = self._document_chunks.get(document_id, [])
        for chunk_id in old_ids:
            self._chunks.pop(chunk_id, None)

        self._document_chunks[document_id] = [chunk.id for chunk in chunks]
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

        ref = DocumentRef(
            id=document_id,
            title=title,
            metadata=meta,
            chunk_count=len(chunks),
            char_count=len(text),
        )
        self._documents[document_id] = ref
        return ref

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkHit]:
        """Return lexical chunk hits ranked by simple term overlap."""
        self._search_count += 1
        if limit <= 0:
            return []
        terms = _terms(query)
        if not terms:
            return []
        out: list[ChunkHit] = []
        for chunk in self._chunks.values():
            if not _metadata_matches(chunk.metadata, filters or {}):
                continue
            score = _score_chunk(terms, chunk)
            if score <= 0:
                continue
            out.append(
                ChunkHit(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    score=score,
                    preview=_cap(" ".join(chunk.text.split()), self._preview_chars),
                    text=chunk.text,
                    title=chunk.title,
                    locator=chunk.locator,
                    metadata=dict(chunk.metadata),
                )
            )
        out.sort(key=lambda hit: (-hit.score, hit.document_id, hit.chunk_id))
        hits = out[:limit]
        self._chunks_returned += len(hits)
        self._chars_returned += sum(len(hit.text) for hit in hits)
        return hits

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        self._chunk_retrieval_count += 1
        return self._chunks.get(chunk_id)

    def list_documents(self) -> list[DocumentRef]:
        return list(self._documents.values())

    def chunks_for_document(self, document_id: str) -> list[DocumentChunk]:
        return [
            self._chunks[chunk_id]
            for chunk_id in self._document_chunks.get(document_id, [])
            if chunk_id in self._chunks
        ]

    def promote_chunk_to_evidence(
        self,
        chunk_id: str,
        evidence: InMemoryEvidenceDossier,
        *,
        claim: str,
        tags: tuple[str, ...] = (),
        confidence: float = 0.75,
        status: str = "supported",
    ) -> EvidenceItem:
        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            raise KeyError(chunk_id)
        item = evidence.add_evidence(
            claim=claim,
            source_ref=f"chunk:{chunk.id}",
            title=chunk.title,
            locator=chunk.locator,
            tool_name="document_search",
            confidence=confidence,
            tags=tags,
            quote=chunk.text,
            status=status,  # type: ignore[arg-type]
            metadata={"document_id": chunk.document_id, **chunk.metadata},
        )
        self._chunk_promotions += 1
        return item

    def stats(self) -> DocumentSearchStats:
        return DocumentSearchStats(
            documents_indexed=len(self._documents),
            chunks_indexed=len(self._chunks),
            document_search_count=self._search_count,
            document_chunks_returned=self._chunks_returned,
            document_search_chars_returned=self._chars_returned,
            chunk_retrieval_count=self._chunk_retrieval_count,
            chunk_promotions_to_evidence=self._chunk_promotions,
        )

    def _split_chunks(
        self,
        document_id: str,
        text: str,
        *,
        title: str | None,
        metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        if not text:
            return chunks
        step = self._chunk_chars - self._chunk_overlap
        idx = 0
        start = 0
        while start < len(text):
            end = min(len(text), start + self._chunk_chars)
            chunk_text = text[start:end]
            chunk_id = _chunk_id(document_id, idx, chunk_text)
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    text=chunk_text,
                    title=title,
                    locator=f"chars:{start}-{end}",
                    metadata=dict(metadata),
                )
            )
            idx += 1
            if end >= len(text):
                break
            start += step
        return chunks


def _terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}


def _score_chunk(query_terms: set[str], chunk: DocumentChunk) -> float:
    haystack_terms = _terms(
        " ".join(
            [
                chunk.text,
                chunk.title or "",
                chunk.document_id,
                " ".join(str(v) for v in chunk.metadata.values()),
            ]
        )
    )
    overlap = query_terms & haystack_terms
    if not overlap:
        return 0.0
    return len(overlap) / max(len(query_terms), 1)


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(key) == expected for key, expected in filters.items())


def _chunk_id(document_id: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{document_id}:{index}:{text}".encode()).hexdigest()
    return f"{document_id}#chunk-{index}-{digest[:10]}"


def _cap(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


__all__ = [
    "ChunkHit",
    "DocumentChunk",
    "DocumentRef",
    "DocumentSearchStats",
    "InMemoryDocumentCorpus",
]
