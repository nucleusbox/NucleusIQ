"""Run-local evidence dossier state for long research tasks."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

EvidenceStatus = Literal["supported", "partial", "conflict", "gap"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class EvidenceSource:
    """Provenance for one evidence item."""

    ref: str
    title: str | None = None
    locator: str | None = None
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    """One structured fact, conflict, partial finding, or gap."""

    id: str
    claim: str
    source: EvidenceSource
    confidence: float
    tags: tuple[str, ...] = ()
    quote: str | None = None
    status: EvidenceStatus = "supported"
    created_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceCoverage:
    """Coverage result for required evidence tags."""

    required_tags: tuple[str, ...]
    present_tags: tuple[str, ...]
    missing_tags: tuple[str, ...]
    gap_tags: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_tags

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_tags": list(self.required_tags),
            "present_tags": list(self.present_tags),
            "missing_tags": list(self.missing_tags),
            "gap_tags": list(self.gap_tags),
            "is_complete": self.is_complete,
        }


@dataclass(frozen=True)
class EvidenceStats:
    """Small telemetry snapshot for evidence dossier usage."""

    backend: str = "memory"
    item_count: int = 0
    supported_count: int = 0
    partial_count: int = 0
    conflict_count: int = 0
    gap_count: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "backend": self.backend,
            "item_count": self.item_count,
            "supported_count": self.supported_count,
            "partial_count": self.partial_count,
            "conflict_count": self.conflict_count,
            "gap_count": self.gap_count,
        }


class InMemoryEvidenceDossier:
    """Bounded run-local evidence store.

    The first implementation is intentionally deterministic: no LLM
    summarization, no embeddings, no persistence.
    """

    __slots__ = ("_items", "_max_items")

    def __init__(self, *, max_items: int = 500) -> None:
        self._items: list[EvidenceItem] = []
        self._max_items = max_items

    def add_evidence(
        self,
        *,
        claim: str,
        source_ref: str,
        title: str | None = None,
        locator: str | None = None,
        tool_name: str | None = None,
        confidence: float = 1.0,
        tags: tuple[str, ...] = (),
        quote: str | None = None,
        status: EvidenceStatus = "supported",
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        if status == "gap":
            raise ValueError("Use add_gap() for gap evidence")
        item = EvidenceItem(
            id=f"ev:{uuid.uuid4().hex[:12]}",
            claim=claim,
            source=EvidenceSource(
                ref=source_ref, title=title, locator=locator, tool_name=tool_name
            ),
            confidence=max(0.0, min(1.0, confidence)),
            tags=tuple(tags),
            quote=quote,
            status=status,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        self._append(item)
        return item

    def add_gap(
        self,
        *,
        question: str,
        reason: str,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        gap_metadata = dict(metadata or {})
        gap_metadata["reason"] = reason
        item = EvidenceItem(
            id=f"ev:{uuid.uuid4().hex[:12]}",
            claim=question,
            source=EvidenceSource(ref="gap", title="Unresolved evidence gap"),
            confidence=0.0,
            tags=tuple(tags),
            quote=reason,
            status="gap",
            created_at=_utc_now(),
            metadata=gap_metadata,
        )
        self._append(item)
        return item

    def list(
        self,
        *,
        tags: tuple[str, ...] = (),
        status: EvidenceStatus | None = None,
    ) -> list[EvidenceItem]:
        out = self._items
        if status is not None:
            out = [item for item in out if item.status == status]
        if tags:
            required = set(tags)
            out = [item for item in out if required.issubset(set(item.tags))]
        return out.copy()

    def query(self, text: str, *, limit: int = 10) -> Sequence[EvidenceItem]:
        needle_terms = [term for term in text.lower().split() if term]
        if not needle_terms:
            return []
        matches: list[EvidenceItem] = []
        for item in self._items:
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
        for item in self._items:
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
        present = {
            tag
            for item in self._items
            if item.status in {"supported", "partial", "conflict"}
            for tag in item.tags
        }
        gap_tags = {
            tag for item in self._items if item.status == "gap" for tag in item.tags
        }
        required = tuple(required_tags)
        present_required = tuple(tag for tag in required if tag in present)
        missing = tuple(tag for tag in required if tag not in present)
        return EvidenceCoverage(
            required_tags=required,
            present_tags=present_required,
            missing_tags=missing,
            gap_tags=tuple(tag for tag in required if tag in gap_tags),
        )

    def clear(self) -> None:
        self._items.clear()

    def stats(self) -> EvidenceStats:
        return EvidenceStats(
            item_count=len(self._items),
            supported_count=sum(
                1 for item in self._items if item.status == "supported"
            ),
            partial_count=sum(1 for item in self._items if item.status == "partial"),
            conflict_count=sum(1 for item in self._items if item.status == "conflict"),
            gap_count=sum(1 for item in self._items if item.status == "gap"),
        )

    def _append(self, item: EvidenceItem) -> None:
        if len(self._items) >= self._max_items:
            raise ValueError("Evidence dossier item limit exceeded")
        self._items.append(item)
