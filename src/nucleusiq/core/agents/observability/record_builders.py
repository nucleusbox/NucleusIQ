"""Factory functions for building tracer records from raw execution data.

Each builder centralises the conversion from provider-specific response
shapes into framework-standard record types.  This keeps provider
coupling out of the tracer itself and out of the mode implementations.
"""

from __future__ import annotations

import json
from typing import Any

from nucleusiq.agents.agent_result import LLMCallRecord, ToolCallRecord
from nucleusiq.agents.chat_models import ToolCallRequest
from nucleusiq.agents.observability._response_parser import (
    extract_tool_calls,
    safe_int,
    usage_dict_from_response,
)


def build_tool_call_record(
    tc: ToolCallRequest,
    *,
    result: Any = None,
    success: bool = True,
    error: str | None = None,
    error_type: str | None = None,
    duration_ms: float = 0.0,
    round: int = 1,
    args: dict[str, Any] | None = None,
) -> ToolCallRecord:
    """Build a :class:`ToolCallRecord` for the tracer."""
    parsed: dict[str, Any] = {}
    if args is not None:
        parsed = dict(args)
    elif tc.arguments:
        try:
            loaded = json.loads(tc.arguments)
            if isinstance(loaded, dict):
                parsed = loaded
        except (json.JSONDecodeError, TypeError):
            pass
    return ToolCallRecord(
        tool_name=tc.name or "",
        tool_call_id=tc.id,
        args=parsed,
        result=result,
        success=success,
        error=error,
        error_type=error_type,
        duration_ms=duration_ms,
        round=round,
    )


def build_llm_call_record(
    response: Any,
    *,
    call_round: int,
    purpose: str,
    duration_ms: float = 0.0,
    model: str | None = None,
) -> LLMCallRecord:
    """Build an :class:`LLMCallRecord` from a non-streaming provider response."""
    usage = usage_dict_from_response(response) or {}
    prompt = safe_int(usage, "prompt_tokens")
    completion = safe_int(usage, "completion_tokens")
    total = safe_int(usage, "total_tokens") or (prompt + completion)
    reasoning = safe_int(usage, "reasoning_tokens")

    tcs = extract_tool_calls(response)
    return LLMCallRecord(
        round=call_round,
        purpose=purpose,
        model=model
        or getattr(response, "model", None)
        or (response.get("model") if isinstance(response, dict) else None),
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
        has_tool_calls=bool(tcs),
        tool_call_count=len(tcs),
        duration_ms=duration_ms,
    )


def build_llm_call_record_from_stream(
    metadata: dict[str, Any] | None,
    *,
    call_round: int,
    purpose: str,
    duration_ms: float = 0.0,
    model: str | None = None,
) -> LLMCallRecord:
    """Build an :class:`LLMCallRecord` from streaming COMPLETE metadata."""
    usage: dict[str, Any] = {}
    if metadata:
        u = metadata.get("usage")
        if isinstance(u, dict):
            usage = u
        elif u is not None and hasattr(u, "model_dump"):
            usage = u.model_dump()

    prompt = safe_int(usage, "prompt_tokens")
    completion = safe_int(usage, "completion_tokens")
    total = safe_int(usage, "total_tokens") or (prompt + completion)
    reasoning = safe_int(usage, "reasoning_tokens")

    raw_tcs: list[Any] = []
    if metadata:
        raw_tcs = metadata.get("tool_calls") or []
    if not isinstance(raw_tcs, list):
        raw_tcs = []

    has_tools = bool(raw_tcs)
    return LLMCallRecord(
        round=call_round,
        purpose=purpose,
        model=model or (metadata.get("model") if metadata else None),
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
        has_tool_calls=has_tools,
        tool_call_count=len(raw_tcs),
        duration_ms=duration_ms,
    )
