"""NoOpTracer — Null Object implementation of ExecutionTracerProtocol.

All ``record_*`` methods are no-ops.  All read-only accessors return
pre-allocated empty singletons, giving **zero** runtime overhead for
users who don't need tracing.
"""

from __future__ import annotations

from typing import Any

from nucleusiq.agents.agent_result import (
    LLMCallRecord,
    MemorySnapshot,
    PluginEvent,
    ToolCallRecord,
    ValidationRecord,
)

_EMPTY_TOOL: tuple[ToolCallRecord, ...] = ()
_EMPTY_LLM: tuple[LLMCallRecord, ...] = ()
_EMPTY_PLUGIN: tuple[PluginEvent, ...] = ()
_EMPTY_VAL: tuple[ValidationRecord, ...] = ()
_EMPTY_WARN: tuple[str, ...] = ()


class NoOpTracer:
    """Tracer that discards all records; read-only views are empty singletons."""

    __slots__ = ()

    def record_tool_call(self, record: ToolCallRecord) -> None:
        pass

    def record_llm_call(self, record: LLMCallRecord) -> None:
        pass

    def record_plugin_event(self, record: PluginEvent) -> None:
        pass

    def record_validation(self, record: ValidationRecord) -> None:
        pass

    def record_warning(self, message: str) -> None:
        pass

    def set_memory_snapshot(self, snapshot: MemorySnapshot | None) -> None:
        pass

    def set_autonomous_detail(self, **kwargs: Any) -> None:
        pass

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]:
        return _EMPTY_TOOL

    @property
    def llm_calls(self) -> tuple[LLMCallRecord, ...]:
        return _EMPTY_LLM

    @property
    def plugin_events(self) -> tuple[PluginEvent, ...]:
        return _EMPTY_PLUGIN

    @property
    def validations(self) -> tuple[ValidationRecord, ...]:
        return _EMPTY_VAL

    @property
    def warnings(self) -> tuple[str, ...]:
        return _EMPTY_WARN

    @property
    def memory_snapshot(self) -> MemorySnapshot | None:
        return None

    @property
    def autonomous_detail(self) -> dict[str, Any] | None:
        return None

    def reset(self) -> None:
        pass
