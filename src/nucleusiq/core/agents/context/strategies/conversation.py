"""ConversationCompactor — moderate compaction tier (Major GC).

Targets old conversation turns beyond ``preserve_recent_turns``.
Two modes:
    - **Truncation (default)**: Remove oldest turns, insert marker.
    - **Structured summary (opt-in)**: Insert a ``SummarySchema``-based
      working state summary extracted from the evicted turns.

Cost: 0 LLM calls. Instant.
Trigger: ~75% utilization of optimal_budget (configurable).

Phase 2 enhancement:
    When ``enable_summarization`` is True, the marker includes a
    structured working state extracted heuristically from evicted
    messages — no extra LLM call needed.
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


class ConversationCompactor(CompactionStrategy):
    """Compacts old conversation turns while preserving recent context.

    Phase 2: when ``enable_summarization=True``, inserts a structured
    working state marker (goals, decisions, open items) extracted
    heuristically from the evicted messages.
    """

    @property
    def name(self) -> str:
        return "conversation_compactor"

    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
    ) -> CompactionResult:
        from nucleusiq.agents.chat_models import ChatMessage as CM

        if len(messages) <= 2:
            return CompactionResult(
                messages=list(messages),
                tokens_freed=0,
                tokens_remaining=budget.allocated,
                strategy_used=self.name,
            )

        pinned_head, evictable, pinned_tail = self._partition(
            messages, config.preserve_recent_turns
        )

        if not evictable:
            return CompactionResult(
                messages=list(messages),
                tokens_freed=0,
                tokens_remaining=budget.allocated,
                strategy_used=self.name,
            )

        evicted_tokens = sum(
            token_counter.count(m.content) if isinstance(m.content, str) else 0
            for m in evictable
        )

        if config.enable_summarization:
            marker_content = self._build_structured_summary(evictable, evicted_tokens)
        else:
            marker_content = (
                f"[{len(evictable)} earlier messages compacted — "
                f"~{evicted_tokens} tokens freed. "
                f"Recent {len(pinned_tail)} messages preserved.]"
            )

        marker = CM(role="system", content=marker_content)

        compacted = pinned_head + [marker] + pinned_tail
        tokens_remaining = budget.allocated - evicted_tokens
        marker_cost = (
            token_counter.count(marker.content)
            if isinstance(marker.content, str)
            else 0
        )
        freed = max(0, evicted_tokens - marker_cost)
        summaries_inserted = 1 if config.enable_summarization else 0

        return CompactionResult(
            messages=compacted,
            tokens_freed=freed,
            tokens_remaining=max(0, tokens_remaining + marker_cost),
            strategy_used=self.name,
            entries_removed=len(evictable),
            summaries_inserted=summaries_inserted,
        )

    @staticmethod
    def _build_structured_summary(
        evicted: list[ChatMessage], evicted_tokens: int
    ) -> str:
        """Build a SummarySchema-style structured working state from evicted messages.

        Extracts key information heuristically without an LLM call:
        - Assistant conclusions → decisions
        - User requests → goals
        - Tool results → findings (abbreviated)
        """
        goals: list[str] = []
        decisions: list[str] = []
        tool_findings: list[str] = []

        for msg in evicted:
            text = msg.content if isinstance(msg.content, str) else ""
            if not text:
                continue

            first_line = text.split("\n")[0][:200]

            if msg.role == "user":
                goals.append(first_line)
            elif msg.role == "assistant":
                decisions.append(first_line)
            elif msg.role == "tool":
                name = msg.name or "tool"
                tool_findings.append(f"{name}: {first_line[:100]}")

        parts = [
            f"[WORKING STATE SUMMARY — {len(evicted)} messages, ~{evicted_tokens} tokens compacted]",
        ]
        if goals:
            parts.append(f"Goals: {'; '.join(goals[:5])}")
        if decisions:
            parts.append(f"Decisions: {'; '.join(decisions[:5])}")
        if tool_findings:
            parts.append(f"Tool findings: {'; '.join(tool_findings[:5])}")

        return "\n".join(parts)

    @staticmethod
    def _partition(
        messages: list[ChatMessage],
        preserve_recent: int,
    ) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
        """Split messages into pinned-head, evictable-middle, pinned-tail.

        Respects tool-call boundaries: an assistant message with
        ``tool_calls`` and all following ``tool`` result messages form
        an atomic group that is never split across partitions.  All
        major LLM providers (OpenAI, Gemini, Anthropic, etc.) require
        tool results to be paired with their originating tool call —
        splitting them causes validation errors at the API level.

        - System messages at the start are always pinned.
        - The last ``preserve_recent`` atomic groups are pinned.
        - Everything in between is evictable.
        """
        pinned_head: list[ChatMessage] = []
        idx = 0
        for msg in messages:
            if msg.role == "system":
                pinned_head.append(msg)
                idx += 1
            else:
                break

        remaining = messages[idx:]
        if not remaining:
            return pinned_head, [], []

        groups: list[list[ChatMessage]] = []
        i = 0
        while i < len(remaining):
            msg = remaining[i]
            group = [msg]
            if msg.role == "assistant" and getattr(msg, "tool_calls", None):
                j = i + 1
                while j < len(remaining) and remaining[j].role == "tool":
                    group.append(remaining[j])
                    j += 1
                i = j
            else:
                i += 1
            groups.append(group)

        tail_count = max(preserve_recent, 1)
        if len(groups) <= tail_count:
            return pinned_head, [], remaining

        evict_groups = groups[: len(groups) - tail_count]
        tail_groups = groups[len(groups) - tail_count :]

        evictable = [m for g in evict_groups for m in g]
        pinned_tail = [m for g in tail_groups for m in g]
        return pinned_head, evictable, pinned_tail
