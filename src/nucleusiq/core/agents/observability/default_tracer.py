"""DefaultExecutionTracer — in-memory implementation of ExecutionTracerProtocol.

Stores records in mutable lists internally; exposes immutable tuple views via
read-only properties.  Designed for single-threaded agent execution (one
tracer per ``Agent.execute()`` call).
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


class DefaultExecutionTracer:
    """Mutable tracer: lists internally, tuple views via properties."""

    __slots__ = (
        "_tool_calls",
        "_llm_calls",
        "_plugin_events",
        "_validations",
        "_warnings",
        "_memory_snapshot",
        "_autonomous_detail",
    )

    def __init__(self) -> None:
        self._tool_calls: list[ToolCallRecord] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._plugin_events: list[PluginEvent] = []
        self._validations: list[ValidationRecord] = []
        self._warnings: list[str] = []
        self._memory_snapshot: MemorySnapshot | None = None
        self._autonomous_detail: dict[str, Any] = {}

    # -- Record methods ------------------------------------------------- #

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self._tool_calls.append(record)

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self._llm_calls.append(record)

    def record_plugin_event(self, record: PluginEvent) -> None:
        self._plugin_events.append(record)

    def record_validation(self, record: ValidationRecord) -> None:
        self._validations.append(record)

    def record_warning(self, message: str) -> None:
        self._warnings.append(message)

    def set_memory_snapshot(self, snapshot: MemorySnapshot | None) -> None:
        self._memory_snapshot = snapshot

    def set_autonomous_detail(self, **kwargs: Any) -> None:
        self._autonomous_detail.update(kwargs)

    # -- Read-only accessors -------------------------------------------- #

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]:
        return tuple(self._tool_calls)

    @property
    def llm_calls(self) -> tuple[LLMCallRecord, ...]:
        return tuple(self._llm_calls)

    @property
    def plugin_events(self) -> tuple[PluginEvent, ...]:
        return tuple(self._plugin_events)

    @property
    def validations(self) -> tuple[ValidationRecord, ...]:
        return tuple(self._validations)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def memory_snapshot(self) -> MemorySnapshot | None:
        return self._memory_snapshot

    @property
    def autonomous_detail(self) -> dict[str, Any] | None:
        return dict(self._autonomous_detail) if self._autonomous_detail else None

    # -- Lifecycle ------------------------------------------------------ #

    def reset(self) -> None:
        self._tool_calls.clear()
        self._llm_calls.clear()
        self._plugin_events.clear()
        self._validations.clear()
        self._warnings.clear()
        self._memory_snapshot = None
        self._autonomous_detail.clear()
