"""Shared evidence between a parent agent and its COMPLEX sub-agents (WS-3).

Design (docs/design/AUTONOMOUS_HARNESS_HARDENING.md, I-5 / I-6).

Before this module every sub-agent started from an empty ``ContentStore``,
document corpus and evidence dossier.  Three children working on the same
nine documents each re-read them, each offloaded their own copy, and the
parent saw none of it after ``gather`` — the office run spent most of its
tool budget re-fetching what a sibling already had.

The model here is **read-through, write-local, merge-after**:

* :class:`ParentEvidenceView` is what a child receives: read-only handles
  on the parent's store / corpus / dossier plus the resources it owns.
* :class:`LayeredContentStore`, :class:`LayeredDocumentCorpus` and
  :class:`LayeredEvidenceDossier` subclass the in-memory stores so every
  existing tool (``recall_tool_result``, ``search_document_corpus`` …)
  works unchanged: reads fall through to the parent, writes stay local.
  Recall / search telemetry stays on the child.
* :func:`merge_child_evidence` copies a child's *local* entries into the
  parent once the child finished (single writer — the parent — so the
  gather is deterministic; idempotent by key / document id).

Everything is in-memory and synchronous; nothing here awaits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nucleusiq.agents.context.document_search import (
    ChunkHit,
    DocumentChunk,
    DocumentRef,
    InMemoryDocumentCorpus,
)
from nucleusiq.agents.context.evidence import (
    EvidenceCoverage,
    EvidenceItem,
    EvidenceStatus,
    InMemoryEvidenceDossier,
)
from nucleusiq.agents.context.store import ContentMetadata, ContentStore

# --------------------------------------------------------------------------- #
# Parent view handed to children                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParentEvidenceView:
    """Read-only handles a sub-agent may consult (never mutate)."""

    store: ContentStore | None = None
    corpus: InMemoryDocumentCorpus | None = None
    dossier: InMemoryEvidenceDossier | None = None
    #: Resources already read into the parent stores (gather-first).
    prefetched_resources: tuple[str, ...] = ()

    @property
    def has_content(self) -> bool:
        if self.store is not None and self.store.size > 0:
            return True
        if self.corpus is not None and self.corpus.list_documents():
            return True
        return bool(self.dossier is not None and self.dossier.list())

    def system_hint(self) -> str:
        """Sentence appended to the child's system prompt when there is
        something to find — tells the model not to re-read what is indexed."""
        if not self.has_content:
            return ""
        parts: list[str] = []
        if self.corpus is not None and self.corpus.list_documents():
            n = len(self.corpus.list_documents())
            parts.append(
                f"{n} document(s) are already indexed — use "
                "search_document_corpus / get_document_chunk before calling "
                "any read or fetch tool"
            )
        if self.store is not None and self.store.size > 0:
            parts.append(
                "earlier tool results are recallable via list_recalled_evidence "
                "/ recall_tool_result"
            )
        if self.prefetched_resources:
            shown = ", ".join(self.prefetched_resources[:12])
            more = (
                f" (+{len(self.prefetched_resources) - 12} more)"
                if len(self.prefetched_resources) > 12
                else ""
            )
            parts.append(f"already fetched: {shown}{more}; do not fetch them again")
        return "Shared evidence: " + "; ".join(parts) + "."

    @classmethod
    def from_agent(
        cls, agent: Any, *, prefetched: Sequence[str] = ()
    ) -> ParentEvidenceView:
        engine = getattr(agent, "_context_engine", None)
        store = getattr(engine, "store", None)
        return cls(
            store=store if isinstance(store, ContentStore) else None,
            corpus=_as(
                getattr(agent, "_document_corpus", None), InMemoryDocumentCorpus
            ),
            dossier=_as(
                getattr(agent, "_evidence_dossier", None), InMemoryEvidenceDossier
            ),
            prefetched_resources=tuple(str(r) for r in prefetched if str(r).strip()),
        )


def _as(value: Any, klass: type) -> Any:
    return value if isinstance(value, klass) else None


# --------------------------------------------------------------------------- #
# Layered stores                                                               #
# --------------------------------------------------------------------------- #


class LayeredContentStore(ContentStore):
    """``ContentStore`` whose reads fall through to a read-only parent."""

    __slots__ = ("_parent",)

    def __init__(self, parent: ContentStore | None) -> None:
        super().__init__()
        self._parent = parent

    @property
    def parent(self) -> ContentStore | None:
        return self._parent

    def local_keys(self) -> list[str]:
        """Keys written by *this* layer (what a merge copies)."""
        return list(self._store.keys())

    def retrieve(self, key: str) -> str | None:
        found = super().retrieve(key)
        if found is None and self._parent is not None:
            return self._parent.retrieve(key)
        return found

    def preview(self, key: str) -> str | None:
        found = super().preview(key)
        if found is None and self._parent is not None:
            return self._parent.preview(key)
        return found

    def metadata(self, key: str) -> ContentMetadata | None:
        found = super().metadata(key)
        if found is None and self._parent is not None:
            return self._parent.metadata(key)
        return found

    def contains(self, key: str) -> bool:
        return super().contains(key) or (
            self._parent is not None and self._parent.contains(key)
        )

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.contains(key)

    def keys(self) -> list[str]:
        local = list(self._store.keys())
        if self._parent is None:
            return local
        seen = set(local)
        parent_keys: list[str] = self._parent.keys()
        return local + [k for k in parent_keys if k not in seen]

    def __iter__(self):
        return iter(self.keys())

    @property
    def size(self) -> int:
        return len(self.keys())


class LayeredDocumentCorpus(InMemoryDocumentCorpus):
    """Corpus whose search / lookups fall through to a read-only parent.

    Search results from both layers are ranked together; a locally
    re-indexed document id shadows the parent's copy.  Search / retrieval
    counters are the child's own — the parent's telemetry is untouched.
    """

    def __init__(self, parent: InMemoryDocumentCorpus | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._parent = parent

    @property
    def parent(self) -> InMemoryDocumentCorpus | None:
        return self._parent

    def local_documents(self) -> list[DocumentRef]:
        return list(self._documents.values())

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkHit]:
        local_hits = super().search(query, limit=limit, filters=filters)
        if self._parent is None:
            return local_hits
        # Scan the parent without touching its counters.
        parent_hits = _scan_corpus(
            self._parent, query, filters=filters, exclude_docs=set(self._documents)
        )
        merged = local_hits + parent_hits
        merged.sort(key=lambda hit: (-hit.score, hit.document_id, hit.chunk_id))
        hits = merged[: max(0, limit)]
        # ``super().search`` already counted the local hits; account only
        # for the parent-sourced ones that made the cut.
        extra = [h for h in hits if h not in local_hits]
        self._chunks_returned += len(extra)
        self._chars_returned += sum(len(h.text) for h in extra)
        return hits

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        found = super().get_chunk(chunk_id)
        if found is None and self._parent is not None:
            return self._parent._chunks.get(chunk_id)
        return found

    def list_documents(self) -> list[DocumentRef]:
        if self._parent is None:
            return super().list_documents()
        merged = {ref.id: ref for ref in self._parent.list_documents()}
        merged.update(self._documents)
        return list(merged.values())

    def chunks_for_document(self, document_id: str) -> list[DocumentChunk]:
        if document_id in self._documents or self._parent is None:
            return super().chunks_for_document(document_id)
        return self._parent.chunks_for_document(document_id)


def _scan_corpus(
    corpus: InMemoryDocumentCorpus,
    query: str,
    *,
    filters: dict[str, Any] | None,
    exclude_docs: set[str],
) -> list[ChunkHit]:
    """``corpus.search`` semantics without mutating ``corpus`` telemetry."""
    from nucleusiq.agents.context import document_search as ds

    terms = ds._terms(query)
    if not terms:
        return []
    out: list[ChunkHit] = []
    for chunk in corpus._chunks.values():
        if chunk.document_id in exclude_docs:
            continue
        if not ds._metadata_matches(chunk.metadata, filters or {}):
            continue
        score = ds._score_chunk(terms, chunk)
        if score <= 0:
            continue
        out.append(
            ChunkHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                score=score,
                preview=ds._cap(" ".join(chunk.text.split()), corpus._preview_chars),
                text=chunk.text,
                title=chunk.title,
                locator=chunk.locator,
                metadata=dict(chunk.metadata),
            )
        )
    return out


class LayeredEvidenceDossier(InMemoryEvidenceDossier):
    """Dossier whose reads include a read-only parent's items."""

    __slots__ = ("_parent",)

    def __init__(
        self, parent: InMemoryEvidenceDossier | None, *, max_items: int = 500
    ) -> None:
        super().__init__(max_items=max_items)
        self._parent = parent

    @property
    def parent(self) -> InMemoryEvidenceDossier | None:
        return self._parent

    def local_items(self) -> list[EvidenceItem]:
        return list(self._items)

    def _all_items(self) -> list[EvidenceItem]:
        if self._parent is None:
            return list(self._items)
        return self._parent.list() + list(self._items)

    def list(
        self,
        *,
        tags: tuple[str, ...] = (),
        status: EvidenceStatus | None = None,
    ) -> list[EvidenceItem]:
        out = self._all_items()
        if status is not None:
            out = [item for item in out if item.status == status]
        if tags:
            required = set(tags)
            out = [item for item in out if required.issubset(set(item.tags))]
        return out

    def query(self, text: str, *, limit: int = 10) -> Sequence[EvidenceItem]:
        needle_terms = [term for term in text.lower().split() if term]
        if not needle_terms:
            return []
        matches: list[EvidenceItem] = []
        for item in self._all_items():
            haystack = " ".join(
                [
                    item.claim,
                    item.quote or "",
                    item.source.ref,
                    item.source.title or "",
                    " ".join(item.tags),
                ]
            ).lower()
            if all(term in haystack for term in needle_terms):
                matches.append(item)
            if len(matches) >= limit:
                break
        return matches

    def summarize(self, *, max_chars: int, include_gaps: bool = True) -> str:
        if max_chars <= 0:
            return ""
        lines = ["[evidence dossier]"]
        for item in self._all_items():
            if item.status == "gap" and not include_gaps:
                continue
            locator = f" {item.source.locator}" if item.source.locator else ""
            tags = f" tags={','.join(item.tags)}" if item.tags else ""
            lines.append(
                f"- {item.status}: {item.claim} "
                f"(source={item.source.ref}{locator}, confidence={item.confidence:.2f}{tags})"
            )
        summary = "\n".join(lines)
        if len(summary) <= max_chars:
            return summary
        return summary[: max(0, max_chars - 3)].rstrip() + "..."

    def coverage(self, required_tags: tuple[str, ...]) -> EvidenceCoverage:
        items = self._all_items()
        present = {
            tag
            for item in items
            if item.status in {"supported", "partial", "conflict"}
            for tag in item.tags
        }
        gap_tags = {tag for item in items if item.status == "gap" for tag in item.tags}
        required = tuple(required_tags)
        return EvidenceCoverage(
            required_tags=required,
            present_tags=tuple(tag for tag in required if tag in present),
            missing_tags=tuple(tag for tag in required if tag not in present),
            gap_tags=tuple(tag for tag in required if tag in gap_tags),
        )


# --------------------------------------------------------------------------- #
# Merge after gather                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class MergeReport:
    """What one child contributed to the parent."""

    child: str = ""
    store_entries: int = 0
    documents: int = 0
    evidence_items: int = 0
    skipped_existing: int = 0
    document_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "store_entries": self.store_entries,
            "documents": self.documents,
            "evidence_items": self.evidence_items,
            "skipped_existing": self.skipped_existing,
        }


def merge_store(parent: ContentStore, child: ContentStore, report: MergeReport) -> None:
    keys = (
        child.local_keys() if isinstance(child, LayeredContentStore) else child.keys()
    )
    for key in keys:
        if parent.contains(key):
            report.skipped_existing += 1
            continue
        content = child.retrieve(key)
        meta = child.metadata(key)
        if content is None:
            continue
        parent.store(
            key,
            content,
            meta.original_tokens if meta is not None else max(1, len(content) // 4),
            trusted=meta.trusted if meta is not None else True,
            tool_name=meta.tool_name if meta is not None else None,
        )
        report.store_entries += 1


def merge_corpus(
    parent: InMemoryDocumentCorpus,
    child: InMemoryDocumentCorpus,
    report: MergeReport,
) -> None:
    """Copy the child's *local* documents (chunks intact) into the parent."""
    docs = (
        child.local_documents()
        if isinstance(child, LayeredDocumentCorpus)
        else child.list_documents()
    )
    for ref in docs:
        if ref.id in parent._documents:
            report.skipped_existing += 1
            continue
        chunk_ids: list[str] = []
        for chunk in child.chunks_for_document(ref.id):
            parent._chunks[chunk.id] = chunk
            chunk_ids.append(chunk.id)
        parent._document_chunks[ref.id] = chunk_ids
        parent._documents[ref.id] = ref
        report.documents += 1
        report.document_ids.append(ref.id)


def merge_dossier(
    parent: InMemoryEvidenceDossier,
    child: InMemoryEvidenceDossier,
    report: MergeReport,
) -> None:
    items = (
        child.local_items()
        if isinstance(child, LayeredEvidenceDossier)
        else child.list()
    )
    existing = {item.id for item in parent.list()}
    for item in items:
        if item.id in existing:
            report.skipped_existing += 1
            continue
        try:
            parent._append(item)
        except ValueError:
            # Parent dossier full — keep what fit; the report says how many.
            break
        report.evidence_items += 1


def merge_child_evidence(parent: Any, child: Any, *, label: str = "") -> MergeReport:
    """Merge a finished child's local stores into ``parent`` (idempotent)."""
    report = MergeReport(child=label or str(getattr(child, "name", "")))
    p_engine = getattr(parent, "_context_engine", None)
    c_engine = getattr(child, "_context_engine", None)
    p_store = getattr(p_engine, "store", None)
    c_store = getattr(c_engine, "store", None)
    if isinstance(p_store, ContentStore) and isinstance(c_store, ContentStore):
        merge_store(p_store, c_store, report)

    p_corpus = getattr(parent, "_document_corpus", None)
    c_corpus = getattr(child, "_document_corpus", None)
    if isinstance(p_corpus, InMemoryDocumentCorpus) and isinstance(
        c_corpus, InMemoryDocumentCorpus
    ):
        merge_corpus(p_corpus, c_corpus, report)

    p_dossier = getattr(parent, "_evidence_dossier", None)
    c_dossier = getattr(child, "_evidence_dossier", None)
    if isinstance(p_dossier, InMemoryEvidenceDossier) and isinstance(
        c_dossier, InMemoryEvidenceDossier
    ):
        merge_dossier(p_dossier, c_dossier, report)
    return report


__all__ = [
    "LayeredContentStore",
    "LayeredDocumentCorpus",
    "LayeredEvidenceDossier",
    "MergeReport",
    "ParentEvidenceView",
    "merge_child_evidence",
    "merge_corpus",
    "merge_dossier",
    "merge_store",
]
