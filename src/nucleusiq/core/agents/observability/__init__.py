"""Observability package — execution tracing for NucleusIQ agents.

Public API:
    ExecutionTracerProtocol  — interface contract for any tracer backend
    DefaultExecutionTracer   — in-memory implementation (default)
    NoOpTracer               — null-object implementation (zero overhead)
    build_tool_call_record   — factory: raw tool execution → ToolCallRecord
    build_llm_call_record    — factory: non-streaming LLM response → LLMCallRecord
    build_llm_call_record_from_stream — factory: streaming metadata → LLMCallRecord

This package is designed to be composable and self-contained so it
can be extracted as a standalone library or replaced by a commercial
observability backend (OpenTelemetry, LangSmith, Datadog, etc.)
without modifying agent internals.
"""

from nucleusiq.agents.observability.default_tracer import DefaultExecutionTracer
from nucleusiq.agents.observability.noop_tracer import NoOpTracer
from nucleusiq.agents.observability.protocol import ExecutionTracerProtocol
from nucleusiq.agents.observability.record_builders import (
    build_llm_call_record,
    build_llm_call_record_from_stream,
    build_tool_call_record,
)

__all__ = [
    "ExecutionTracerProtocol",
    "DefaultExecutionTracer",
    "NoOpTracer",
    "build_tool_call_record",
    "build_llm_call_record",
    "build_llm_call_record_from_stream",
]
