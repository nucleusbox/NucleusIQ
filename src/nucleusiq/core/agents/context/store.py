"""ContentStore — offloaded artifact storage (swap space analogy).

Large tool results are "paged out" here and replaced with a
reference + preview in the conversation.  Like OS swap, it trades
context space for retrieval latency.

The interface is abstract enough for future durable backends
(Redis, file, S3) without breaking existing consumers.

Rehydration markers (Gap 4):
    Offloaded content includes ``[context_ref: {key}]`` markers so
    future versions can implement rehydration policies.

Rehydration (F2):
    ``extract_raw_trace`` re-hydrates masked tool messages from the
    store so inspectors (Critic, Refiner) see the raw tool output
    instead of the opaque marker.  Without this, ``ObservationMasker``
    would blind every downstream consumer of the shared ``messages``
    list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage


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


# ------------------------------------------------------------------ #
# F2 — Rehydration helper                                             #
# ------------------------------------------------------------------ #

#: Regex used to pull the ``ref: {key}`` line out of a masked marker.
#: Kept deliberately loose (``[^\s\]]+``) so it accepts any key shape
#: the masker may produce now or in the future.
_REF_LINE_RE = re.compile(r"^ref:\s*(\S+)\s*$", re.MULTILINE)

#: Any message content that starts with this prefix is a masked marker.
#: Mirrors ``observation_masker.MASK_PREFIX`` — duplicated here only to
#: avoid a circular import (``store`` is imported by the masker, not
#: the other way around).
_MASK_PREFIX = "[observation consumed"


def extract_raw_trace(
    messages: "list[ChatMessage]",
    store: "ContentStore | None",
    *,
    max_chars_per_result: int = 8_000,
) -> "list[ChatMessage]":
    """Return a copy of ``messages`` with masked tool content rehydrated.

    For every ``role='tool'`` message whose content is a masked
    marker, parse the ``ref: {key}`` line and look the key up in the
    store.  When the key resolves to real content, replace the marker
    with that content (optionally truncated to
    ``max_chars_per_result``).  When the key is missing or the store
    is ``None``, keep the marker as-is — inspectors still have the
    tool/args/summary slots from F1 to reason over.

    Design properties:

    * **Pure / non-mutating.** Returns a new list; input messages are
      untouched (important — the production ``messages`` list is
      shared across the autonomous loop).
    * **Idempotent.** Non-masked messages pass through unchanged, and
      rehydrated messages will not be touched on a subsequent call
      (they no longer start with the mask prefix).
    * **Fail-open.** Never raises on malformed markers — an invalid
      marker is kept as-is so the inspector still sees *something*.

    Args:
        messages: Conversation messages as ``ChatMessage`` (or an
            empty list).
        store: The active ``ContentStore``.  ``None`` is accepted for
            callers that may run without an engine (e.g. unit tests of
            the components in isolation); in that case the function
            returns ``messages`` unchanged.
        max_chars_per_result: Hard cap on rehydrated payload length
            per message.  Prevents a single huge tool result from
            blowing up a Critic prompt that only needs to *see* the
            evidence, not re-ingest it fully.

    Returns:
        A new list where applicable tool markers are rehydrated.
    """
    if store is None or not messages:
        return list(messages)

    from nucleusiq.agents.chat_models import ChatMessage as CM

    rehydrated: list[ChatMessage] = []
    for msg in messages:
        content = msg.content
        if (
            msg.role != "tool"
            or not isinstance(content, str)
            or not content.startswith(_MASK_PREFIX)
        ):
            rehydrated.append(msg)
            continue

        match = _REF_LINE_RE.search(content)
        if not match:
            rehydrated.append(msg)
            continue

        key = match.group(1)
        raw = store.retrieve(key)
        if raw is None:
            rehydrated.append(msg)
            continue

        if len(raw) > max_chars_per_result:
            raw = raw[:max_chars_per_result] + "\n... (truncated)"

        rehydrated.append(
            CM(
                role=msg.role,
                content=raw,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            )
        )

    return rehydrated
