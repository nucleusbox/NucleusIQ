"""ToolResultCompactor — cheapest compaction tier (Minor GC).

Targets tool-result messages that exceed ``tool_result_threshold``.
Two modes:
    - **Truncation**: Keep first N + last M lines, insert marker.
    - **Offloading**: Store full result in ``ContentStore``, replace
      with preview + rehydration reference.

Cost: Zero LLM calls. Near-instant execution.
Trigger: ~70% utilization (configurable).
"""

from __future__ import annotations

import uuid
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

_PREVIEW_HEAD_LINES = 8
_PREVIEW_TAIL_LINES = 4
_TRUNCATION_MARKER = (
    "\n[...truncated {freed} tokens — tool result exceeded threshold...]\n"
)


class ToolResultCompactor(CompactionStrategy):
    """Compacts oversized tool-result messages."""

    @property
    def name(self) -> str:
        return "tool_result_compactor"

    async def compact(
        self,
        messages: list[ChatMessage],
        budget: ContextBudget,
        config: ContextConfig,
        token_counter: TokenCounter,
        store: ContentStore | None = None,
    ) -> CompactionResult:
        from nucleusiq.agents.chat_models import ChatMessage as CM

        compacted: list[ChatMessage] = []
        total_freed = 0
        artifacts_offloaded = 0
        warnings: list[str] = []

        for msg in messages:
            if msg.role != "tool" or not isinstance(msg.content, str):
                compacted.append(msg)
                continue

            content_tokens = token_counter.count(msg.content)
            if content_tokens <= config.tool_result_threshold:
                compacted.append(msg)
                continue

            if config.enable_offloading and store is not None:
                new_content, freed = self._offload(
                    msg.content,
                    content_tokens,
                    msg.name or "tool",
                    token_counter,
                    store,
                )
                artifacts_offloaded += 1
            else:
                new_content, freed = self._truncate(
                    msg.content, content_tokens, token_counter
                )

            total_freed += freed
            compacted.append(
                CM(
                    role=msg.role,
                    content=new_content,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                )
            )

        tokens_remaining = budget.allocated - total_freed
        return CompactionResult(
            messages=compacted,
            tokens_freed=total_freed,
            tokens_remaining=max(0, tokens_remaining),
            strategy_used=self.name,
            artifacts_offloaded=artifacts_offloaded,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _truncate(
        content: str,
        original_tokens: int,
        token_counter: TokenCounter,
    ) -> tuple[str, int]:
        """Keep head + tail lines, insert truncation marker.

        Falls back to character-based truncation for dense content
        (fewer lines than the head+tail window) to guarantee savings.
        """
        lines = content.split("\n")
        if len(lines) > _PREVIEW_HEAD_LINES + _PREVIEW_TAIL_LINES:
            head = "\n".join(lines[:_PREVIEW_HEAD_LINES])
            tail = "\n".join(lines[-_PREVIEW_TAIL_LINES:])
            dropped = len(lines) - _PREVIEW_HEAD_LINES - _PREVIEW_TAIL_LINES
            marker = _TRUNCATION_MARKER.format(freed=f"~{dropped} lines")
            truncated = head + marker + tail

            new_tokens = token_counter.count(truncated)
            freed = max(0, original_tokens - new_tokens)
            return truncated, freed

        head_chars = max(200, len(content) // 5)
        tail_chars = max(100, len(content) // 10)
        min_content = head_chars + tail_chars + 100
        if len(content) <= min_content:
            return content, 0

        head = content[:head_chars]
        tail = content[-tail_chars:]
        dropped_chars = len(content) - head_chars - tail_chars
        marker = _TRUNCATION_MARKER.format(freed=f"~{dropped_chars} chars")
        truncated = head + marker + tail

        new_tokens = token_counter.count(truncated)
        freed = max(0, original_tokens - new_tokens)
        return truncated, freed

    @staticmethod
    def _offload(
        content: str,
        original_tokens: int,
        tool_name: str,
        token_counter: TokenCounter,
        store: ContentStore,
    ) -> tuple[str, int]:
        """Store full content, return preview with rehydration marker.

        The preview is capped at ~10% of the original content length
        (minimum 200 chars) to guarantee meaningful space savings even
        for dense content with few line breaks.
        """
        key = f"{tool_name}:{uuid.uuid4().hex[:12]}"
        preview_char_budget = max(200, len(content) // 10)
        ref = store.store(
            key=key,
            content=content,
            original_tokens=original_tokens,
            preview_lines=_PREVIEW_HEAD_LINES,
            preview_max_chars=preview_char_budget,
        )
        marker_text = ref.to_marker()
        new_tokens = token_counter.count(marker_text)
        freed = max(0, original_tokens - new_tokens)
        return marker_text, freed
