"""
FullHistoryMemory — stores every message verbatim.

Best for: short conversations, debugging, audit trails.
Trade-off: unbounded token growth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from nucleusiq.memory.base import BaseMemory


class FullHistoryMemory(BaseMemory):
    """Keeps the complete conversation history in memory."""

    _messages: List[Dict[str, str]] = []

    def model_post_init(self, __context: Any) -> None:
        self._messages = []

    @property
    def strategy_name(self) -> str:
        return "full_history"

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self._messages.append({"role": role, "content": content})

    def get_context(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def export_state(self) -> Dict[str, Any]:
        return {"messages": list(self._messages)}

    def import_state(self, state: Dict[str, Any]) -> None:
        self._messages = list(state.get("messages", []))
