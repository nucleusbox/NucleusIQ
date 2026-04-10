"""EmergencyCompactor — last-resort compaction (Full GC / OOM Killer).

Keeps only the system prompt + the last atomic message group
(assistant + tool results).  Everything else is dropped with a
prominent warning marker.

Groups assistant(tool_calls) messages with their subsequent tool
result messages into atomic units.  All major LLM providers
require tool results to follow their originating tool call —
splitting them causes API validation errors regardless of provider.

**Always succeeds** in freeing space — this is the guarantee.

Cost: Zero LLM calls. Instant.
Trigger: ~95% utilization (configurable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nucleusiq.agents.context.strategies.base import (
    CompactionResult,
    CompactionStrategy,
)

if TYPE_CHECKING:
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context.budget import ContextBudget
    from nucleusiq.agents.context.config import ContextConfig
    from nucleusiq.agents.context.counter import TokenCounter
    from nucleusiq.agents.context.store import ContentStore

_EMERGENCY_MARKER = (
    "[CONTEXT COMPACTED: emergency reduction triggered at {util:.0%} utilization. "
    "{dropped} messages removed (~{tokens} tokens). "
    "Only system prompt and last {kept} messages preserved.]"
)


class EmergencyCompactor(CompactionStrategy):
    """Last-resort compaction — harsh but guarantees the system can continue."""

    @property
    def name(self) -> str:
        return "emergency_compactor"

    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
    ) -> CompactionResult:
        from nucleusiq.agents.chat_models import ChatMessage as CM

        system_msgs: list[ChatMessage] = []
        rest: list[ChatMessage] = []
        for msg in messages:
            if msg.role == "system" and not rest:
                system_msgs.append(msg)
            else:
                rest.append(msg)

        groups: list[list[ChatMessage]] = []
        i = 0
        while i < len(rest):
            msg = rest[i]
            group = [msg]
            if msg.role == "assistant" and getattr(msg, "tool_calls", None):
                j = i + 1
                while j < len(rest) and rest[j].role == "tool":
                    group.append(rest[j])
                    j += 1
                i = j
            else:
                i += 1
            groups.append(group)

        tail_group_count = 1
        if len(groups) <= tail_group_count:
            return CompactionResult(
                messages=list(messages),
                tokens_freed=0,
                tokens_remaining=budget.allocated,
                strategy_used=self.name,
                warnings=("emergency_compactor: nothing to evict",),
            )

        kept_groups = groups[-tail_group_count:]
        dropped_groups = groups[:-tail_group_count]
        kept_tail = [m for g in kept_groups for m in g]
        dropped = [m for g in dropped_groups for m in g]

        dropped_tokens = sum(
            token_counter.count(m.content) if isinstance(m.content, str) else 0
            for m in dropped
        )

        marker_text = _EMERGENCY_MARKER.format(
            util=budget.utilization,
            dropped=len(dropped),
            tokens=dropped_tokens,
            kept=len(kept_tail),
        )
        marker = CM(role="system", content=marker_text)
        marker_cost = token_counter.count(marker_text)

        compacted = system_msgs + [marker] + kept_tail
        freed = max(0, dropped_tokens - marker_cost)

        return CompactionResult(
            messages=compacted,
            tokens_freed=freed,
            tokens_remaining=max(0, budget.allocated - freed),
            strategy_used=self.name,
            entries_removed=len(dropped),
            warnings=(
                f"Emergency compaction: dropped {len(dropped)} messages "
                f"(~{dropped_tokens} tokens) at {budget.utilization:.0%} utilization",
            ),
        )
