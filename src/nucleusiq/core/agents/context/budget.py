"""Region, ContextBudget, and ContextLedger — token accounting system.

This module implements the "memory allocator" side of context management.
The ledger only *tracks* allocations; it never *decides* what to evict
(that's the compaction strategies' job — separation of concerns).

Design patterns:
    - **Value Object**: ``ContextBudget`` is frozen — safe to log, cache,
      pass to strategies without mutation risk.
    - **SRP**: ``ContextLedger`` only tracks numbers. Strategies decide eviction.

Region-aware budgeting is a key differentiator: no other framework tracks
tokens by region, making it impossible to answer "what's eating my context?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Region(str, Enum):
    """Categorizes context content for region-aware budgeting.

    Enables targeted compaction: "87% of context is TOOL_RESULT — enable
    offloading" is a diagnosis that only region tracking can provide.
    """

    SYSTEM = "system"
    MEMORY = "memory"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    RESERVED = "reserved"


@dataclass(frozen=True)
class ContextBudget:
    """Point-in-time snapshot of context token budget.

    Immutable value object — safe to pass around, log, and store
    in telemetry without mutation concerns.
    """

    max_tokens: int
    response_reserve: int
    allocated: int
    by_region: dict[str, int]

    @property
    def effective_limit(self) -> int:
        """Tokens available for context (max minus response reserve)."""
        return self.max_tokens - self.response_reserve

    @property
    def available(self) -> int:
        """Tokens still available for new content."""
        return max(0, self.effective_limit - self.allocated)

    @property
    def utilization(self) -> float:
        """Current utilization as 0.0-1.0 ratio."""
        if self.effective_limit <= 0:
            return 1.0
        return min(1.0, self.allocated / self.effective_limit)

    def can_fit(self, additional_tokens: int) -> bool:
        """Check if additional content would fit within budget."""
        return additional_tokens <= self.available


@dataclass
class _LedgerEntry:
    """Internal: one tracked message allocation."""

    msg_id: str
    tokens: int
    region: Region
    importance: float = 0.5
    source_type: str = "unknown"
    restorable: bool = False
    trusted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextLedger:
    """Tracks per-message token allocations grouped by region.

    Analogous to a memory allocator: every message is an allocation,
    every removal is a deallocation.  Provides O(1) budget snapshots.

    SRP: only tracks numbers. Does not decide what to evict.
    """

    __slots__ = ("_max_tokens", "_response_reserve", "_entries", "_region_totals")

    def __init__(self, max_tokens: int, response_reserve: int) -> None:
        self._max_tokens = max_tokens
        self._response_reserve = response_reserve
        self._entries: dict[str, _LedgerEntry] = {}
        self._region_totals: dict[Region, int] = {r: 0 for r in Region}
        self._region_totals[Region.RESERVED] = response_reserve

    def allocate(
        self,
        msg_id: str,
        tokens: int,
        region: Region,
        *,
        importance: float = 0.5,
        source_type: str = "unknown",
        restorable: bool = False,
        trusted: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a token allocation for a message."""
        if msg_id in self._entries:
            self.deallocate(msg_id)
        entry = _LedgerEntry(
            msg_id=msg_id,
            tokens=tokens,
            region=region,
            importance=importance,
            source_type=source_type,
            restorable=restorable,
            trusted=trusted,
            metadata=metadata or {},
        )
        self._entries[msg_id] = entry
        self._region_totals[region] = self._region_totals.get(region, 0) + tokens

    def deallocate(self, msg_id: str) -> int:
        """Remove a message allocation. Returns freed tokens (0 if not found)."""
        entry = self._entries.pop(msg_id, None)
        if entry is None:
            return 0
        self._region_totals[entry.region] = max(
            0, self._region_totals.get(entry.region, 0) - entry.tokens
        )
        return entry.tokens

    def get_entry(self, msg_id: str) -> _LedgerEntry | None:
        """Retrieve a ledger entry by message ID."""
        return self._entries.get(msg_id)

    def entries_by_region(self, region: Region) -> list[_LedgerEntry]:
        """Get all entries for a specific region, ordered by insertion."""
        return [e for e in self._entries.values() if e.region == region]

    @property
    def total_allocated(self) -> int:
        """Total tokens allocated (excluding response reserve)."""
        return sum(t for r, t in self._region_totals.items() if r != Region.RESERVED)

    def by_region(self) -> dict[str, int]:
        """Per-region token breakdown."""
        return {r.value: t for r, t in self._region_totals.items()}

    def snapshot(self) -> ContextBudget:
        """Create an immutable budget snapshot."""
        return ContextBudget(
            max_tokens=self._max_tokens,
            response_reserve=self._response_reserve,
            allocated=self.total_allocated,
            by_region=self.by_region(),
        )

    def reset(self) -> None:
        """Clear all allocations."""
        self._entries.clear()
        self._region_totals = {r: 0 for r in Region}
        self._region_totals[Region.RESERVED] = self._response_reserve

    @property
    def entry_count(self) -> int:
        """Number of tracked messages."""
        return len(self._entries)
