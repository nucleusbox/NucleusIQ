"""
BaseMemory — abstract interface for all memory strategies.

Subclasses implement the sync core methods. The agent calls async
variants (a-prefixed) which delegate to sync by default.

For external providers with native async SDKs (Mem0, aiohttp-based),
override the async methods directly.

    Built-in (in-memory)    -> implement sync only
    Sync provider (Redis)   -> implement sync; optionally override async
    Async provider (Mem0)   -> override async; implement sync as fallback
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class BaseMemory(BaseModel, ABC):
    """Abstract base for all memory strategies.

    First-class citizen in NucleusIQ: pass to ``Agent(memory=...)``
    to activate.  No ``enable_memory`` flag needed — presence of the
    object is the signal.

    Subclass contract:
        - MUST implement ``strategy_name``, ``add_message``,
          ``get_context``, ``clear``
        - MAY override async (a-prefixed) variants for true async I/O
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for multi-user memory isolation.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for conversation scoping.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata passed to provider backends.",
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Short identifier for this strategy (e.g. 'full_history')."""
        ...

    # ------------------------------------------------------------------
    # Sync core — subclass MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Store a single message.

        Args:
            role: Message role ('user', 'assistant', 'system', 'tool').
            content: Message text.
            **kwargs: Provider-specific extras (e.g. ``tool_call_id``).
        """
        ...

    @abstractmethod
    def get_context(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        """Retrieve relevant conversation context.

        Args:
            query: Optional semantic query.  In-memory strategies may
                   ignore it; vector/search providers use it for
                   relevance ranking.
            **kwargs: Provider-specific extras (e.g. ``top_k``).

        Returns:
            List of message dicts ``{"role": ..., "content": ...}``.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Erase all stored messages / state."""
        ...

    # ------------------------------------------------------------------
    # Sync optional — sensible defaults
    # ------------------------------------------------------------------

    def get_relevant_context(
        self, query: Any = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        """Retrieve context relevant to a specific query or task.

        By default delegates to ``get_context()`` — suitable for
        in-memory strategies.  Override in RAG-backed or search-based
        memories to perform relevance ranking.

        Args:
            query: Semantic query, task dict, or ``None``.
            **kwargs: Provider-specific extras.

        Returns:
            List of message dicts ``{"role": ..., "content": ...}``.
        """
        q = query if isinstance(query, str) else None
        return self.get_context(query=q, **kwargs)

    def initialize(self) -> None:
        """One-time setup (connect to DB, warm cache, etc.)."""

    def export_state(self) -> Dict[str, Any]:
        """Serialize internal state for persistence."""
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        """Restore internal state from a previous export."""

    # ------------------------------------------------------------------
    # Async — delegates to sync by default.
    # Override ONLY if your backend has a native async SDK.
    # ------------------------------------------------------------------

    async def aget_relevant_context(
        self, query: Any = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        return self.get_relevant_context(query, **kwargs)

    async def aadd_message(
        self, role: str, content: str, **kwargs: Any
    ) -> None:
        self.add_message(role, content, **kwargs)

    async def aget_context(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        return self.get_context(query, **kwargs)

    async def aclear(self) -> None:
        self.clear()

    async def ainitialize(self) -> None:
        self.initialize()

    async def aexport_state(self) -> Dict[str, Any]:
        return self.export_state()

    async def aimport_state(self, state: Dict[str, Any]) -> None:
        self.import_state(state)
