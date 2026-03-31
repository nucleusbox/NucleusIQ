"""Tool execution error hierarchy.

Hierarchy::

    NucleusIQError
    └── ToolError
        ├── ToolExecutionError    — tool.execute() raised at runtime
        ├── ToolTimeoutError      — tool exceeded timeout
        ├── ToolValidationError   — invalid arguments (JSON parse, type mismatch)
        ├── ToolPermissionError   — blocked by guard plugin whitelist/blacklist
        └── ToolNotFoundError     — tool name not registered in executor
"""

from __future__ import annotations

from typing import Any

from nucleusiq.errors import NucleusIQError

__all__ = [
    "ToolError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
    "ToolPermissionError",
    "ToolNotFoundError",
]


class ToolError(NucleusIQError):
    """Base exception for all tool-related errors.

    Attributes:
        tool_name: Name of the tool that failed.
        original_error: The underlying exception, if available.
        args_snapshot: Arguments that were passed to the tool (for diagnostics).
    """

    def __init__(
        self,
        message: str = "",
        *,
        tool_name: str = "unknown",
        original_error: BaseException | None = None,
        args_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        self.args_snapshot = args_snapshot
        super().__init__(message)

    def __repr__(self) -> str:
        parts = [f"{type(self).__name__}({self!s})"]
        if self.tool_name != "unknown":
            parts.append(f"tool_name={self.tool_name!r}")
        return " ".join(parts)


class ToolExecutionError(ToolError):
    """Tool's ``execute()`` method raised an unhandled exception at runtime."""


class ToolTimeoutError(ToolError):
    """Tool execution exceeded the configured timeout."""


class ToolValidationError(ToolError):
    """Invalid arguments passed to a tool.

    Raised when JSON argument parsing fails, required parameters are
    missing, or argument types don't match the tool's schema.
    """


class ToolPermissionError(ToolError):
    """Tool invocation blocked by a guard plugin (whitelist/blacklist)."""


class ToolNotFoundError(ToolError):
    """Requested tool name is not registered in the executor."""
