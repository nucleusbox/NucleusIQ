"""
ContextWindowPlugin — trim messages when approaching context window limits.

Prevents token overflow by trimming older messages while preserving the
system message and most recent conversation turns.

Usage::

    agent = Agent(
        ...,
        plugins=[
            ContextWindowPlugin(max_messages=50, keep_recent=10),
        ],
    )

    # Token-based trimming (approximate)
    agent = Agent(
        ...,
        plugins=[
            ContextWindowPlugin(max_tokens=8000, keep_recent=5),
        ],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from nucleusiq.plugins.base import BasePlugin, ModelRequest

logger = logging.getLogger(__name__)


def _approximate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _message_token_count(
    msg: Any, counter: Callable[[str], int] = _approximate_tokens
) -> int:
    """Extract text content from a message and count tokens."""
    content = ""
    if hasattr(msg, "content"):
        content = getattr(msg, "content", "") or ""
    elif isinstance(msg, dict):
        content = msg.get("content", "") or ""
    if isinstance(content, str):
        return counter(content)
    return 0


class ContextWindowPlugin(BasePlugin):
    """Trims messages to stay within context window limits.

    Supports two modes (use one or both):
    - **Message count**: Trim when message count exceeds ``max_messages``
    - **Token count**: Trim when approximate token count exceeds ``max_tokens``

    When trimming, the plugin:
    1. Always preserves the first message (typically the system prompt)
    2. Keeps the most recent ``keep_recent`` messages
    3. Optionally inserts a summary placeholder for removed messages

    Args:
        max_messages: Maximum number of messages allowed. ``None`` = no limit.
        max_tokens: Maximum approximate token count. ``None`` = no limit.
        keep_recent: Number of recent messages to always keep.
        token_counter: Custom function ``(str) -> int`` for token counting.
        placeholder: Text inserted where messages are trimmed. ``None`` = no placeholder.
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        keep_recent: int = 10,
        token_counter: Optional[Callable[[str], int]] = None,
        placeholder: Optional[str] = "[Earlier messages trimmed for context window]",
    ) -> None:
        if max_messages is None and max_tokens is None:
            raise ValueError("At least one of max_messages or max_tokens is required")
        if max_messages is not None and max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        if keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")

        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._keep_recent = keep_recent
        self._counter = token_counter or _approximate_tokens
        self._placeholder = placeholder

    @property
    def name(self) -> str:
        return "context_window"

    def _trim_by_messages(self, messages: List[Any]) -> List[Any]:
        if self._max_messages is None or len(messages) <= self._max_messages:
            return messages

        keep_start = 1
        keep_end = self._keep_recent
        if keep_start + keep_end >= len(messages):
            return messages

        trimmed_count = len(messages) - keep_start - keep_end
        logger.info(
            "ContextWindowPlugin: trimming %d messages (had %d, limit %d)",
            trimmed_count,
            len(messages),
            self._max_messages,
        )

        result = list(messages[:keep_start])
        if self._placeholder:
            result.append({"role": "system", "content": self._placeholder})
        result.extend(messages[-keep_end:])
        return result

    def _trim_by_tokens(self, messages: List[Any]) -> List[Any]:
        if self._max_tokens is None:
            return messages

        total = sum(_message_token_count(m, self._counter) for m in messages)
        if total <= self._max_tokens:
            return messages

        keep_start = 1
        keep_end = min(self._keep_recent, len(messages) - 1)

        head = messages[:keep_start]
        tail = messages[-keep_end:]

        head_tokens = sum(_message_token_count(m, self._counter) for m in head)
        tail_tokens = sum(_message_token_count(m, self._counter) for m in tail)
        budget = self._max_tokens - head_tokens - tail_tokens

        if budget <= 0:
            result = head + tail
        else:
            middle = messages[keep_start:-keep_end] if keep_end > 0 else messages[keep_start:]
            kept_middle: List[Any] = []
            used = 0
            for msg in reversed(middle):
                cost = _message_token_count(msg, self._counter)
                if used + cost > budget:
                    break
                kept_middle.insert(0, msg)
                used += cost
            result = head + kept_middle + tail

        trimmed = len(messages) - len(result)
        if trimmed > 0:
            logger.info(
                "ContextWindowPlugin: trimmed %d messages by token limit "
                "(had ~%d tokens, limit %d)",
                trimmed,
                total,
                self._max_tokens,
            )
            if self._placeholder and len(result) > keep_start:
                result.insert(keep_start, {"role": "system", "content": self._placeholder})

        return result

    async def before_model(self, request: ModelRequest) -> Optional[ModelRequest]:
        messages = list(request.messages)

        if self._max_messages is not None:
            messages = self._trim_by_messages(messages)

        if self._max_tokens is not None:
            messages = self._trim_by_tokens(messages)

        if len(messages) != len(request.messages):
            return request.with_(messages=messages)
        return None
