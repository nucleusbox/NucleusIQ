"""Streaming event types for NucleusIQ framework.

``StreamEvent`` is the **public contract** that consumers iterate over
when calling ``agent.execute_stream()`` or ``llm.call_stream()``.

This module lives at ``nucleusiq.streaming`` (framework-level) so that
both the LLM layer and the agent/execution-mode layer can import it
without circular dependencies.

Providers yield ``TOKEN`` events; execution modes add orchestration
events (``TOOL_CALL_START``, ``LLM_CALL_START``, etc.); consumers
process the unified stream.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class StreamEventType(str, Enum):
    """Discriminator for streaming events.

    LLM-level events:
        TOKEN          – partial text chunk from the model
        COMPLETE       – final accumulated text (non-streaming fallback)
        ERROR          – unrecoverable error

    Orchestration events (emitted by execution modes):
        LLM_CALL_START – a new LLM round-trip is beginning
        LLM_CALL_END   – an LLM round-trip finished
        TOOL_CALL_START – tool invocation is starting
        TOOL_CALL_END   – tool invocation finished
        THINKING       – intermediate reasoning / agent notes
    """

    TOKEN = "token"
    TOOL_CALL_START = "tool_start"
    TOOL_CALL_END = "tool_end"
    LLM_CALL_START = "llm_start"
    LLM_CALL_END = "llm_end"
    THINKING = "thinking"
    COMPLETE = "complete"
    ERROR = "error"


class StreamEvent(BaseModel):
    """A single event emitted during streamed execution.

    Only the fields relevant to ``type`` are populated; the rest
    remain ``None``.
    """

    model_config = ConfigDict(use_enum_values=True)

    type: StreamEventType

    # TOKEN
    token: str | None = None

    # COMPLETE
    content: str | None = None

    # TOOL_CALL_START / TOOL_CALL_END
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None

    # LLM_CALL_START / LLM_CALL_END
    call_count: int | None = None

    # THINKING / ERROR
    message: str | None = None

    # Attached to any event (provider-specific extras)
    metadata: dict[str, Any] | None = None

    def to_sse(self) -> str:
        """Serialize to a Server-Sent Events ``data:`` line (JSON)."""
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def token_event(
        cls, text: str, *, metadata: dict[str, Any] | None = None
    ) -> StreamEvent:
        return cls(type=StreamEventType.TOKEN, token=text, metadata=metadata)

    @classmethod
    def complete_event(
        cls, full_text: str, *, metadata: dict[str, Any] | None = None
    ) -> StreamEvent:
        return cls(type=StreamEventType.COMPLETE, content=full_text, metadata=metadata)

    @classmethod
    def tool_start_event(
        cls, name: str, args: dict[str, Any] | None = None
    ) -> StreamEvent:
        return cls(type=StreamEventType.TOOL_CALL_START, tool_name=name, tool_args=args)

    @classmethod
    def tool_end_event(cls, name: str, result: str) -> StreamEvent:
        return cls(
            type=StreamEventType.TOOL_CALL_END, tool_name=name, tool_result=result
        )

    @classmethod
    def llm_start_event(cls, call_count: int) -> StreamEvent:
        return cls(type=StreamEventType.LLM_CALL_START, call_count=call_count)

    @classmethod
    def llm_end_event(cls, call_count: int) -> StreamEvent:
        return cls(type=StreamEventType.LLM_CALL_END, call_count=call_count)

    @classmethod
    def thinking_event(cls, text: str) -> StreamEvent:
        return cls(type=StreamEventType.THINKING, message=text)

    @classmethod
    def error_event(cls, text: str) -> StreamEvent:
        return cls(type=StreamEventType.ERROR, message=text)
