"""ExecutionTracerProtocol — the observability contract.

Any tracer implementation (in-memory, OpenTelemetry, LangSmith, file-based,
or a commercial plugin) must satisfy this Protocol.

Designed as a ``@runtime_checkable`` Protocol so users can verify compliance
with ``isinstance(obj, ExecutionTracerProtocol)`` at runtime.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nucleusiq.agents.agent_result import (
    LLMCallRecord,
    MemorySnapshot,
    PluginEvent,
    ToolCallRecord,
    ValidationRecord,
)


@runtime_checkable
class ExecutionTracerProtocol(Protocol):
    """Per-execution audit trail (tools, LLM, plugins, validation, warnings).

    Record methods:
        record_tool_call      — after every tool invocation (success or failure)
        record_llm_call       — after every LLM call (streaming or non-streaming)
        record_plugin_event   — after every plugin hook fires
        record_validation     — after every validation attempt (Layer 1/2, Critic)
        record_warning        — non-fatal issues (memory write failure, etc.)
        set_memory_snapshot   — snapshot of memory state at end of execution
        set_autonomous_detail — decomposition, critique, and refinement metadata

    Read-only accessors:
        tool_calls, llm_calls, plugin_events, validations, warnings,
        memory_snapshot, autonomous_detail

    Lifecycle:
        reset() — clear all recorded data (called at start of each execution)
    """

    def record_tool_call(self, record: ToolCallRecord) -> None: ...
    def record_llm_call(self, record: LLMCallRecord) -> None: ...
    def record_plugin_event(self, record: PluginEvent) -> None: ...
    def record_validation(self, record: ValidationRecord) -> None: ...
    def record_warning(self, message: str) -> None: ...
    def set_memory_snapshot(self, snapshot: MemorySnapshot | None) -> None: ...
    def set_autonomous_detail(self, **kwargs: Any) -> None: ...

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]: ...
    @property
    def llm_calls(self) -> tuple[LLMCallRecord, ...]: ...
    @property
    def plugin_events(self) -> tuple[PluginEvent, ...]: ...
    @property
    def validations(self) -> tuple[ValidationRecord, ...]: ...
    @property
    def warnings(self) -> tuple[str, ...]: ...
    @property
    def memory_snapshot(self) -> MemorySnapshot | None: ...
    @property
    def autonomous_detail(self) -> dict[str, Any] | None: ...

    def reset(self) -> None: ...
