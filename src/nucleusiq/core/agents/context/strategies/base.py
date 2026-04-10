"""CompactionStrategy ABC and CompactionResult — the strategy contract.

New compaction strategies can be added without modifying existing code
(Open/Closed Principle).  Each strategy handles one type of compaction
at a specific urgency level.

Design patterns:
    - **Strategy** (GoF): interchangeable compaction algorithms.
    - **Value Object**: ``CompactionResult`` is frozen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context.budget import ContextBudget
    from nucleusiq.agents.context.config import ContextConfig
    from nucleusiq.agents.context.counter import TokenCounter
    from nucleusiq.agents.context.store import ContentStore


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of a compaction operation."""

    messages: list[ChatMessage]
    tokens_freed: int
    tokens_remaining: int
    strategy_used: str
    artifacts_offloaded: int = 0
    summaries_inserted: int = 0
    entries_removed: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)


class CompactionStrategy(ABC):
    """Abstract base for compaction strategies.

    New strategies can be added without modifying existing code
    (Open/Closed Principle).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name for telemetry."""
        ...

    @abstractmethod
    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
    ) -> CompactionResult:
        """Apply compaction to messages.

        Must return a new (possibly shorter) message list and report
        how many tokens were freed.  Must NOT mutate the input list.
        """
        ...
