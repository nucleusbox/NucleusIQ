"""Internal helpers for normalising LLM response shapes into tracer data.

These functions handle the variance across providers (OpenAI, Gemini,
streaming metadata) so that record builders receive clean, uniform inputs.
"""

from __future__ import annotations

import json
from typing import Any


def safe_int(d: dict[str, Any], key: str) -> int:
    """Extract an int from *d*, returning 0 on missing or invalid values."""
    val = d.get(key)
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def usage_dict_from_response(response: Any) -> dict[str, Any] | None:
    """Normalize ``response.usage`` to a plain dict (mirrors UsageTracker)."""
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None and isinstance(response, dict):
        usage_obj = response.get("usage")
    if usage_obj is None:
        return None
    if isinstance(usage_obj, dict):
        return usage_obj
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
        "total_tokens": getattr(usage_obj, "total_tokens", 0),
        "reasoning_tokens": getattr(usage_obj, "reasoning_tokens", 0),
    }


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Collect tool-call wire dicts from OpenAI ``choices`` or Gemini-style parts.

    Handles:
    - OpenAI: ``message.tool_calls``
    - Legacy: ``message.function_call``
    - Gemini-style: ``message.content`` as list of parts with ``function_call``
    """
    if response is None:
        return []
    choices = getattr(response, "choices", None)
    if not choices:
        if isinstance(response, dict):
            choices = response.get("choices") or []
        else:
            return []
    if not choices:
        return []

    choice0 = choices[0]
    msg = getattr(choice0, "message", None)
    if msg is None and isinstance(choice0, dict):
        msg = choice0.get("message")

    if msg is None:
        return []

    raw_tcs = getattr(msg, "tool_calls", None)
    if raw_tcs is None and isinstance(msg, dict):
        raw_tcs = msg.get("tool_calls")
    if raw_tcs and isinstance(raw_tcs, list):
        out: list[dict[str, Any]] = []
        for tc in raw_tcs:
            if isinstance(tc, dict):
                out.append(tc)
        if out:
            return out

    fc = getattr(msg, "function_call", None)
    if fc is None and isinstance(msg, dict):
        fc = msg.get("function_call")
    if isinstance(fc, dict) and fc.get("name"):
        args = fc.get("arguments", "{}")
        if not isinstance(args, str):
            try:
                args = json.dumps(args)
            except (TypeError, ValueError):
                args = "{}"
        return [
            {
                "id": None,
                "type": "function",
                "function": {"name": fc.get("name", ""), "arguments": args},
            }
        ]

    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if not isinstance(content, list):
        return []

    gemini_calls: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        fn = part.get("function_call")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "") or ""
        args = fn.get("args", fn.get("arguments", {}))
        if isinstance(args, str):
            arg_str = args
        else:
            try:
                arg_str = json.dumps(args)
            except (TypeError, ValueError):
                arg_str = "{}"
        entry: dict[str, Any] = {
            "type": "function",
            "function": {"name": name, "arguments": arg_str},
        }
        cid = fn.get("id")
        if cid is not None:
            entry["id"] = cid
        gemini_calls.append(entry)
    return gemini_calls
