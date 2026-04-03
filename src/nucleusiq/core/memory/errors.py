"""Memory subsystem error hierarchy.

Hierarchy::

    NucleusIQError
    └── NucleusMemoryError
        ├── MemoryWriteError     — failed to store message
        ├── MemoryReadError      — failed to retrieve context
        ├── MemoryImportError    — failed to import/deserialize state
        └── MemoryCapacityError  — token budget / window exceeded

.. note::

    The base class is named ``NucleusMemoryError`` (not ``MemoryError``)
    to avoid shadowing Python's built-in ``MemoryError`` (out-of-memory).
"""

from __future__ import annotations

from nucleusiq.errors.base import NucleusIQError

__all__ = [
    "NucleusMemoryError",
    "MemoryWriteError",
    "MemoryReadError",
    "MemoryImportError",
    "MemoryCapacityError",
]


class NucleusMemoryError(NucleusIQError):
    """Base exception for memory subsystem errors.

    Attributes:
        strategy: Memory strategy name (e.g. "sliding_window").
        operation: Operation that failed (e.g. "write", "get_context").
    """

    def __init__(
        self,
        message: str = "",
        *,
        strategy: str | None = None,
        operation: str | None = None,
    ) -> None:
        self.strategy = strategy
        self.operation = operation
        super().__init__(message)


class MemoryWriteError(NucleusMemoryError):
    """Failed to store a message in memory."""


class MemoryReadError(NucleusMemoryError):
    """Failed to retrieve context from memory."""


class MemoryImportError(NucleusMemoryError):
    """Failed to import or deserialize memory state."""


class MemoryCapacityError(NucleusMemoryError):
    """Token budget or window capacity exceeded (informational)."""
