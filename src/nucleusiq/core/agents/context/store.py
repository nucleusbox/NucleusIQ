"""ContentStore — offloaded artifact storage (swap space analogy).

Large tool results are "paged out" here and replaced with a
reference + preview in the conversation.  Like OS swap, it trades
context space for retrieval latency.

The interface is abstract enough for future durable backends
(Redis, file, S3) without breaking existing consumers.

Rehydration markers (Gap 4):
    Offloaded content includes ``[context_ref: {key}]`` markers so
    future versions can implement rehydration policies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContentRef:
    """Reference to an offloaded artifact."""

    key: str
    original_tokens: int
    preview: str
    trusted: bool = True

    def to_marker(self) -> str:
        """Generate the in-context replacement text.

        Includes a machine-readable reference for future rehydration
        and a human/model-readable preview of the content.
        """
        lines = [
            f"[context_ref: {self.key}]",
            f"[original size: ~{self.original_tokens} tokens, offloaded]",
            "--- preview ---",
            self.preview,
            "--- end preview ---",
        ]
        return "\n".join(lines)


class ContentStore:
    """In-memory store for offloaded tool results.

    Interface designed for future durable backends (Redis, file, S3)
    without breaking existing consumers.
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, str]] = {}  # key → (full_content, preview)

    def store(
        self,
        key: str,
        content: str,
        original_tokens: int,
        *,
        preview_lines: int = 10,
        preview_max_chars: int | None = None,
        trusted: bool = True,
    ) -> ContentRef:
        """Offload content and return a reference.

        Args:
            key: Unique identifier (typically ``tool_name:call_id``).
            content: Full content to store.
            original_tokens: Token count of the original content.
            preview_lines: Number of lines to include in the preview.
            preview_max_chars: Hard cap on preview character length.
                When the line-based preview exceeds this budget, the
                preview is rebuilt using character truncation instead.
                Prevents dense content (few newlines) from producing
                previews nearly as large as the original.
            trusted: Whether this content comes from a trusted source.

        Returns:
            A ``ContentRef`` with preview and rehydration marker.
        """
        lines = content.split("\n")
        preview = "\n".join(lines[:preview_lines])
        if len(lines) > preview_lines:
            preview += f"\n... ({len(lines) - preview_lines} more lines)"

        if preview_max_chars is not None and len(preview) > preview_max_chars:
            preview = content[:preview_max_chars]
            if len(content) > preview_max_chars:
                remaining = len(content) - preview_max_chars
                preview += f"\n... ({remaining:,} chars remaining)"

        self._store[key] = (content, preview)
        return ContentRef(
            key=key,
            original_tokens=original_tokens,
            preview=preview,
            trusted=trusted,
        )

    def retrieve(self, key: str) -> str | None:
        """Retrieve full content by key (for future rehydration)."""
        entry = self._store.get(key)
        return entry[0] if entry else None

    def preview(self, key: str) -> str | None:
        """Retrieve preview only."""
        entry = self._store.get(key)
        return entry[1] if entry else None

    def contains(self, key: str) -> bool:
        """Check if a key exists in the store."""
        return key in self._store

    def remove(self, key: str) -> bool:
        """Remove an artifact. Returns True if it existed."""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all stored artifacts."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Number of stored artifacts."""
        return len(self._store)

    def keys(self) -> list[str]:
        """List all stored artifact keys."""
        return list(self._store.keys())
