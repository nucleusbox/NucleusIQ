"""CompactionPipeline — progressive compaction orchestrator.

Applies compaction strategies in priority order (cheapest first) until
utilization drops below the target threshold.  Modeled after JVM GC
generations: Minor → Major → Full.

Design patterns:
    - **Composite** (GoF): Pipeline composes multiple strategies.
    - **Chain of Responsibility** (GoF): Each strategy gets a chance
      to reduce utilization; stops early when budget is satisfied.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from nucleusiq.agents.context.strategies.base import (
    CompactionResult,
    CompactionStrategy,
)
from nucleusiq.agents.context.telemetry import CompactionEvent

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context.budget import ContextBudget
    from nucleusiq.agents.context.config import ContextConfig
    from nucleusiq.agents.context.counter import TokenCounter
    from nucleusiq.agents.context.store import ContentStore


class CompactionPipeline:
    """Progressive compaction: applies strategies in priority order.

    Each tier is a ``(trigger_threshold, strategy)`` pair, sorted by
    threshold ascending.  The pipeline runs each strategy whose
    trigger has been reached, stopping when utilization is acceptable.
    """

    __slots__ = ("_tiers",)

    def __init__(self, tiers: list[tuple[float, CompactionStrategy]]) -> None:
        self._tiers = sorted(tiers, key=lambda t: t[0])

    async def run(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
    ) -> tuple[list[ChatMessage], list[CompactionEvent]]:
        """Apply strategies progressively until budget is within limits.

        Returns:
            Tuple of (compacted messages, list of CompactionEvents for telemetry).
        """
        current_messages = list(messages)
        events: list[CompactionEvent] = []
        current_util = budget.utilization

        for trigger_threshold, strategy in self._tiers:
            if current_util < trigger_threshold:
                continue

            t0 = time.perf_counter()
            tokens_before = token_counter.count_messages(current_messages)

            result: CompactionResult = await strategy.compact(
                current_messages, budget, config, token_counter, store
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            tokens_after = token_counter.count_messages(result.messages)

            events.append(
                CompactionEvent(
                    strategy=strategy.name,
                    trigger_utilization=current_util,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    tokens_freed=result.tokens_freed,
                    artifacts_offloaded=result.artifacts_offloaded,
                    duration_ms=elapsed_ms,
                )
            )

            current_messages = result.messages

            if budget.effective_limit > 0:
                current_util = tokens_after / budget.effective_limit
            else:
                current_util = 1.0

            if current_util < config.tool_compaction_trigger:
                break

        return current_messages, events

    @property
    def tier_count(self) -> int:
        """Number of registered compaction tiers."""
        return len(self._tiers)
