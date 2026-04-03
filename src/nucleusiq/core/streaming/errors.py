"""Streaming pipeline error hierarchy.

Hierarchy::

    NucleusIQError
    └── StreamingError
        ├── StreamInterruptedError      — stream broke mid-generation
        └── StreamOrchestrationError    — tool/LLM loop failed during streaming
"""

from __future__ import annotations

from nucleusiq.errors.base import NucleusIQError

__all__ = [
    "StreamingError",
    "StreamInterruptedError",
    "StreamOrchestrationError",
]


class StreamingError(NucleusIQError):
    """Base exception for streaming pipeline errors.

    Attributes:
        event_type: Stream event type when the error occurred.
        round: Tool-loop round number, if applicable.
    """

    def __init__(
        self,
        message: str = "",
        *,
        event_type: str | None = None,
        round: int | None = None,
    ) -> None:
        self.event_type = event_type
        self.round = round
        super().__init__(message)


class StreamInterruptedError(StreamingError):
    """Stream broke mid-generation (connection lost, provider timeout)."""


class StreamOrchestrationError(StreamingError):
    """Tool/LLM loop failed during a streaming execution."""
