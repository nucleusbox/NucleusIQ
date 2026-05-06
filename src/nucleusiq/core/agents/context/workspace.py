"""Run-local workspace state for agents.

The workspace is a bounded in-memory notebook for one agent execution. It is
not long-term memory, not a skill system, and not persisted to disk.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

WorkspaceEntryKind = Literal["note", "artifact", "summary"]


class WorkspaceLimitError(ValueError):
    """Raised when a workspace write would exceed configured limits."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class WorkspaceEntry:
    """One run-local workspace item."""

    id: str
    kind: WorkspaceEntryKind
    title: str
    content: str
    source_refs: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceStats:
    """Small telemetry snapshot for workspace usage."""

    backend: str = "memory"
    entry_count: int = 0
    total_chars: int = 0
    max_entries: int = 0
    max_total_chars: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "backend": self.backend,
            "entry_count": self.entry_count,
            "total_chars": self.total_chars,
            "max_entries": self.max_entries,
            "max_total_chars": self.max_total_chars,
        }


class InMemoryWorkspace:
    """Bounded run-local workspace.

    This is intentionally simple and deterministic. It stores entries in memory
    for the current agent run and exposes a bounded summary for later synthesis
    package work.
    """

    __slots__ = (
        "_entries",
        "_max_entries",
        "_max_entry_chars",
        "_max_total_chars",
    )

    def __init__(
        self,
        *,
        max_entries: int = 100,
        max_entry_chars: int = 20_000,
        max_total_chars: int = 200_000,
    ) -> None:
        self._entries: list[WorkspaceEntry] = []
        self._max_entries = max_entries
        self._max_entry_chars = max_entry_chars
        self._max_total_chars = max_total_chars

    def write_note(
        self,
        *,
        title: str,
        content: str,
        source_refs: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceEntry:
        return self._write(
            kind="note",
            title=title,
            content=content,
            source_refs=source_refs,
            metadata=metadata,
        )

    def write_artifact(
        self,
        *,
        title: str,
        content: str,
        source_refs: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceEntry:
        return self._write(
            kind="artifact",
            title=title,
            content=content,
            source_refs=source_refs,
            metadata=metadata,
        )

    def write_summary(
        self,
        *,
        title: str,
        content: str,
        source_refs: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceEntry:
        return self._write(
            kind="summary",
            title=title,
            content=content,
            source_refs=source_refs,
            metadata=metadata,
        )

    def append_note(self, entry_id: str, content: str) -> WorkspaceEntry:
        for idx, entry in enumerate(self._entries):
            if entry.id != entry_id:
                continue
            if entry.kind != "note":
                raise WorkspaceLimitError("Only note entries can be appended")
            combined = f"{entry.content}\n{content}" if entry.content else content
            combined = self._cap_entry(combined)
            updated = WorkspaceEntry(
                id=entry.id,
                kind=entry.kind,
                title=entry.title,
                content=combined,
                source_refs=entry.source_refs,
                created_at=entry.created_at,
                updated_at=_utc_now(),
                metadata=dict(entry.metadata),
            )
            old_total = self._total_chars() - len(entry.content)
            self._ensure_total_limit(old_total + len(updated.content))
            self._entries[idx] = updated
            return updated
        raise KeyError(entry_id)

    def read(self, entry_id: str) -> WorkspaceEntry | None:
        return next((entry for entry in self._entries if entry.id == entry_id), None)

    def list(self, kind: WorkspaceEntryKind | None = None) -> list[WorkspaceEntry]:
        if kind is None:
            return list(self._entries)
        return [entry for entry in self._entries if entry.kind == kind]

    def summarize(self, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        lines = ["[workspace summary]"]
        for entry in self._entries:
            content = " ".join(entry.content.split())
            lines.append(f"- {entry.kind}: {entry.title} — {content}")
        summary = "\n".join(lines)
        if len(summary) <= max_chars:
            return summary
        return summary[: max(0, max_chars - 3)].rstrip() + "..."

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> WorkspaceStats:
        return WorkspaceStats(
            entry_count=len(self._entries),
            total_chars=self._total_chars(),
            max_entries=self._max_entries,
            max_total_chars=self._max_total_chars,
        )

    def _write(
        self,
        *,
        kind: WorkspaceEntryKind,
        title: str,
        content: str,
        source_refs: tuple[str, ...],
        metadata: dict[str, Any] | None,
    ) -> WorkspaceEntry:
        if len(self._entries) >= self._max_entries:
            raise WorkspaceLimitError("Workspace entry limit exceeded")
        capped = self._cap_entry(content)
        self._ensure_total_limit(self._total_chars() + len(capped))
        now = _utc_now()
        entry = WorkspaceEntry(
            id=f"ws:{uuid.uuid4().hex[:12]}",
            kind=kind,
            title=title,
            content=capped,
            source_refs=source_refs,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        self._entries.append(entry)
        return entry

    def _cap_entry(self, content: str) -> str:
        if len(content) <= self._max_entry_chars:
            return content
        return content[: self._max_entry_chars]

    def _total_chars(self) -> int:
        return sum(len(entry.content) for entry in self._entries)

    def _ensure_total_limit(self, total_chars: int) -> None:
        if total_chars > self._max_total_chars:
            raise WorkspaceLimitError("Workspace total character limit exceeded")
