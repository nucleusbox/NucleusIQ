"""
TokenBudgetMemory — keeps messages within a token limit.

Best for: strict token-budget constraints, mixed message sizes.
Trade-off: loses oldest messages when budget is exceeded.

Token counting uses a simple heuristic (chars / 4) by default.
Pass a custom ``token_counter`` callable for accurate counts.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List

from nucleusiq.memory.base import BaseMemory
from pydantic import ConfigDict, Field


def _default_token_counter(text: str) -> int:
    """Rough approximation: 1 token ~ 4 characters."""
    return max(1, len(text) // 4)


class TokenBudgetMemory(BaseMemory):
    """Retains messages that fit within ``max_tokens``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum token budget for stored messages.",
    )
    token_counter: Callable[[str], int] = Field(
        default=_default_token_counter,
        description="Function mapping text to an integer token count.",
    )

    _messages: deque = deque()
    _token_counts: deque = deque()
    _total_tokens: int = 0

    def model_post_init(self, __context: Any) -> None:
        self._messages: deque = deque()
        self._token_counts: deque = deque()
        self._total_tokens: int = 0

    @property
    def strategy_name(self) -> str:
        return "token_budget"

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        tokens = self.token_counter(content)
        self._messages.append({"role": role, "content": content})
        self._token_counts.append(tokens)
        self._total_tokens += tokens
        self._evict()

    def get_context(
        self, query: str | None = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._token_counts.clear()
        self._total_tokens = 0

    def export_state(self) -> Dict[str, Any]:
        return {
            "messages": list(self._messages),
            "token_counts": list(self._token_counts),
            "max_tokens": self.max_tokens,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        self._messages = deque(state.get("messages", []))
        self._token_counts = deque(state.get("token_counts", []))
        self._total_tokens = sum(self._token_counts)

    def _evict(self) -> None:
        while self._total_tokens > self.max_tokens and self._messages:
            self._messages.popleft()
            removed = self._token_counts.popleft()
            self._total_tokens -= removed
