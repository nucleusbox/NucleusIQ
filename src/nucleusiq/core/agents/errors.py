"""Agent-level and attachment error hierarchies.

Hierarchy::

    NucleusIQError
    ├── AgentError
    │   ├── AgentConfigError      — invalid config at execute time
    │   ├── AgentExecutionError   — mode-level failure that can't be recovered
    │   └── AgentTimeoutError     — overall execution timeout exceeded
    │
    └── AttachmentError
        ├── AttachmentValidationError    — size/type/count policy violation
        ├── AttachmentProcessingError    — failed to read/encode/decode
        └── AttachmentUnsupportedError   — provider doesn't support type
"""

from __future__ import annotations

from nucleusiq.errors import NucleusIQError

__all__ = [
    "AgentError",
    "AgentConfigError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "AttachmentError",
    "AttachmentValidationError",
    "AttachmentProcessingError",
    "AttachmentUnsupportedError",
]


class AgentError(NucleusIQError):
    """Base exception for agent-level execution errors.

    Attributes:
        mode: Execution mode when the error occurred (direct/standard/autonomous).
        task_id: Task ID if available.
        original_error: The underlying exception, if available.
    """

    def __init__(
        self,
        message: str = "",
        *,
        mode: str | None = None,
        task_id: str | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        self.mode = mode
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(message)


class AgentConfigError(AgentError):
    """Invalid agent configuration detected at execute time.

    Examples: missing LLM, tool count exceeding mode limit,
    conflicting config values.
    """


class AgentExecutionError(AgentError):
    """Mode-level failure that cannot be recovered by retries."""


class AgentTimeoutError(AgentError):
    """Overall agent execution timeout exceeded."""


# ------------------------------------------------------------------ #
# Attachment errors                                                    #
# ------------------------------------------------------------------ #


class AttachmentError(NucleusIQError):
    """Base exception for attachment processing errors.

    Attributes:
        attachment_type: Type of attachment (image_url, file, etc.).
        file_name: File name if available.
    """

    def __init__(
        self,
        message: str = "",
        *,
        attachment_type: str | None = None,
        file_name: str | None = None,
    ) -> None:
        self.attachment_type = attachment_type
        self.file_name = file_name
        super().__init__(message)


class AttachmentValidationError(AttachmentError):
    """Attachment violates size, type, or count policy."""


class AttachmentProcessingError(AttachmentError):
    """Failed to read, encode, or decode an attachment."""


class AttachmentUnsupportedError(AttachmentError):
    """Provider does not support this attachment type."""
