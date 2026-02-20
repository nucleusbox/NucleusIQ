"""
SlidingWindowMemory — keeps the last *k* messages.

Best for: chatbots, most common use case, predictable token usage.
Trade-off: loses older context.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from pydantic import Field

from nucleusiq.memory.base import BaseMemory


class SlidingWindowMemory(BaseMemory):
    """Retains only the most recent ``window_size`` messages."""

    window_size: int = Field(
        default=10,
        gt=0,
        description="Maximum number of messages to retain.",
    )

    _messages: deque = deque()

    def model_post_init(self, __context: Any) -> None:
        self._messages = deque(maxlen=self.window_size)

    @property
    def strategy_name(self) -> str:
        return "sliding_window"

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self._messages.append({"role": role, "content": content})

    def get_context(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def export_state(self) -> Dict[str, Any]:
        return {
            "messages": list(self._messages),
            "window_size": self.window_size,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        ws = state.get("window_size", self.window_size)
        self._messages = deque(state.get("messages", []), maxlen=ws)
