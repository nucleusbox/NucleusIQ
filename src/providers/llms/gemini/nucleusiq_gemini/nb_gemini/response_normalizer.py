"""Normalize Gemini SDK responses to ``GeminiLLMResponse``.

**Single Responsibility**: Only handles response normalization — no SDK
calls, no tool conversion, no streaming.

Converts the raw Gemini ``GenerateContentResponse`` object into the
provider's Pydantic ``GeminiLLMResponse`` model which matches the
``BaseLLM`` contract shape (choices → message → content / tool_calls).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from nucleusiq_gemini._shared.response_models import (
    AssistantMessage,
    GeminiLLMResponse,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
    _Choice,
)

logger = logging.getLogger(__name__)


def normalize_response(raw_response: Any) -> GeminiLLMResponse:
    """Convert a raw Gemini SDK response to ``GeminiLLMResponse``.

    Args:
        raw_response: ``GenerateContentResponse`` from the Gemini SDK.

    Returns:
        Normalized ``GeminiLLMResponse`` with choices, usage, etc.
    """
    candidates = getattr(raw_response, "candidates", None) or []
    usage_meta = getattr(raw_response, "usage_metadata", None)
    model_version = getattr(raw_response, "model_version", None)

    choices = [_normalize_candidate(c) for c in candidates]
    if not choices:
        choices = [_Choice(message=AssistantMessage(content=""))]

    usage = _extract_usage(usage_meta) if usage_meta else None

    return GeminiLLMResponse(
        choices=choices,
        usage=usage,
        model=model_version,
    )


def _normalize_candidate(candidate: Any) -> _Choice:
    """Convert a single Gemini candidate to a ``_Choice``."""
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or [] if content else []

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    native_outputs: list[dict[str, Any]] = []

    for part in parts:
        thought = getattr(part, "thought", None)
        if thought:
            native_outputs.append({"type": "thinking", "text": str(thought)})
            continue

        text = getattr(part, "text", None)
        if text:
            text_parts.append(text)

        fn_call = getattr(part, "function_call", None)
        if fn_call:
            tool_calls.append(_normalize_function_call(fn_call))

        executable_code = getattr(part, "executable_code", None)
        if executable_code:
            native_outputs.append(
                {
                    "type": "code_execution",
                    "code": getattr(executable_code, "code", ""),
                    "language": getattr(executable_code, "language", "PYTHON"),
                }
            )

        code_result = getattr(part, "code_execution_result", None)
        if code_result:
            native_outputs.append(
                {
                    "type": "code_execution_result",
                    "output": getattr(code_result, "output", ""),
                    "outcome": getattr(code_result, "outcome", "OUTCOME_OK"),
                }
            )

    combined_text = "".join(text_parts) if text_parts else None

    message = AssistantMessage(
        content=combined_text,
        tool_calls=tool_calls if tool_calls else None,
        native_outputs=native_outputs if native_outputs else None,
    )
    return _Choice(message=message)


def _normalize_function_call(fn_call: Any) -> ToolCall:
    """Convert a Gemini function call part to a ``ToolCall``."""
    name = getattr(fn_call, "name", "") or ""
    args = getattr(fn_call, "args", None) or {}
    call_id = getattr(fn_call, "id", None) or str(uuid.uuid4())

    if isinstance(args, dict):
        args_str = json.dumps(args)
    else:
        args_str = str(args)

    return ToolCall(
        id=call_id,
        type="function",
        function=ToolCallFunction(name=name, arguments=args_str),
    )


def _extract_usage(usage_meta: Any) -> UsageInfo:
    """Extract token usage from Gemini's ``usage_metadata``."""
    return UsageInfo(
        prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
        thoughts_tokens=getattr(usage_meta, "thoughts_token_count", 0) or 0,
        cached_tokens=getattr(usage_meta, "cached_content_token_count", 0) or 0,
    )


def messages_to_gemini_contents(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert BaseLLM message format to Gemini contents format.

    Extracts the system instruction from the first message if its role
    is ``"system"``, then converts the remaining messages to Gemini's
    ``contents`` format (role: user/model, parts: [...]).

    Args:
        messages: Standard BaseLLM messages list.

    Returns:
        Tuple of (system_instruction, contents).
    """
    system_instruction: str | None = None
    contents: list[dict[str, Any]] = []

    for msg_index, msg in enumerate(messages):
        role = msg.get("role", "user")

        if role == "system":
            system_instruction = msg.get("content", "")
            continue

        gemini_role = _map_role(role)

        if role == "tool":
            parts = _build_tool_result_parts(
                msg, all_messages=messages, msg_index=msg_index
            )
        else:
            parts = _build_content_parts(msg)

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    return system_instruction, contents


def _map_role(role: str) -> str:
    """Map standard roles to Gemini roles."""
    role_map = {
        "user": "user",
        "assistant": "model",
        "model": "model",
        "tool": "user",
        "function": "user",
    }
    return role_map.get(role, "user")


def _build_content_parts(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Gemini content parts from a standard message."""
    parts: list[dict[str, Any]] = []
    content = msg.get("content")

    if isinstance(content, str) and content:
        parts.append({"text": content})
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                parts.extend(_convert_content_item(item))

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            part: dict[str, Any] = {
                "function_call": {
                    "name": fn.get("name", ""),
                    "args": args,
                }
            }
            call_id = tc.get("id")
            if call_id:
                part["function_call"]["id"] = call_id
            parts.append(part)

    return parts


def _tool_call_entry_name(tc: Any) -> str:
    """Best-effort function name from an OpenAI-style or SDK-style tool call entry."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return (fn.get("name") or "").strip()
        return (tc.get("name") or "").strip()
    fn = getattr(tc, "function", None)
    if fn is not None:
        nm = getattr(fn, "name", None)
        if isinstance(nm, str) and nm.strip():
            return nm.strip()
    nm2 = getattr(tc, "name", None)
    if isinstance(nm2, str):
        return nm2.strip()
    return ""


def _tool_call_entry_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return str(tc.get("id") or "")
    v = getattr(tc, "id", None)
    return str(v) if v is not None else ""


def _infer_tool_name_from_prior_assistant(
    messages: list[dict[str, Any]],
    tool_msg_index: int,
    tool_call_id: str,
) -> str:
    """Match ``tool_call_id`` to a prior assistant ``tool_calls[]`` entry (Gemini needs non-empty name)."""
    want = str(tool_call_id)
    for j in range(tool_msg_index - 1, -1, -1):
        prev = messages[j]
        if prev.get("role") not in ("assistant", "model"):
            continue
        for tc in prev.get("tool_calls") or []:
            if _tool_call_entry_id(tc) != want:
                continue
            n = _tool_call_entry_name(tc)
            if n:
                return n
    return ""


def _infer_tool_name_single_call_prior_assistant(
    messages: list[dict[str, Any]],
    tool_msg_index: int,
) -> str:
    """If the prior assistant message has exactly one tool call, use its function name."""
    for j in range(tool_msg_index - 1, -1, -1):
        prev = messages[j]
        if prev.get("role") not in ("assistant", "model"):
            continue
        tcs = list(prev.get("tool_calls") or [])
        if len(tcs) != 1:
            return ""
        n = _tool_call_entry_name(tcs[0])
        return n
    return ""


def _infer_first_tool_name_immediate_prior_assistant(
    messages: list[dict[str, Any]],
    tool_msg_index: int,
) -> str:
    """Last resort: first tool name on the message immediately before this tool result."""
    if tool_msg_index < 1:
        return ""
    prev = messages[tool_msg_index - 1]
    if prev.get("role") not in ("assistant", "model"):
        return ""
    tcs = list(prev.get("tool_calls") or [])
    if not tcs:
        return ""
    if len(tcs) > 1:
        logger.warning(
            "Using first of %d tool_calls to fill function_response.name "
            "(tool message had no name; call ids may not have matched).",
            len(tcs),
        )
    return _tool_call_entry_name(tcs[0])


def _build_tool_result_parts(
    msg: dict[str, Any],
    *,
    all_messages: list[dict[str, Any]] | None = None,
    msg_index: int | None = None,
) -> list[dict[str, Any]]:
    """Build Gemini function response parts from a tool result message."""
    content = msg.get("content", "")
    tool_call_id = msg.get("tool_call_id", "")
    name = (msg.get("name") or "").strip()
    if not name and all_messages is not None and msg_index is not None:
        if tool_call_id:
            name = _infer_tool_name_from_prior_assistant(
                all_messages, msg_index, str(tool_call_id)
            )
        if not name:
            name = _infer_tool_name_single_call_prior_assistant(
                all_messages, msg_index
            )
        if not name:
            name = _infer_first_tool_name_immediate_prior_assistant(
                all_messages, msg_index
            )

    try:
        response_data = json.loads(content) if content else {}
    except json.JSONDecodeError:
        response_data = {"result": content}
    # google-genai requires function_response.response to be a mapping, not a bare str/list.
    if not isinstance(response_data, dict):
        response_data = {"result": response_data}

    if not name:
        logger.warning(
            "Gemini function_response requires a non-empty function name; "
            "tool message had no name and none could be inferred from prior assistant tool_calls."
        )

    part: dict[str, Any] = {
        "function_response": {
            "name": name,
            "response": response_data,
        }
    }
    if tool_call_id:
        part["function_response"]["id"] = tool_call_id

    return [part]


def _convert_content_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a multimodal content item to Gemini parts."""
    item_type = item.get("type", "")

    if item_type == "text":
        text = item.get("text", "")
        return [{"text": text}] if text else []

    if item_type == "image_url":
        image_url = item.get("image_url", {})
        url = (
            image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
        )
        if url.startswith("data:"):
            mime, _, b64 = url.partition(";base64,")
            mime = mime.replace("data:", "")
            return [{"inline_data": {"mime_type": mime, "data": b64}}]
        return [{"text": f"[Image: {url}]"}]

    if item_type == "file":
        file_data = item.get("file", {})
        file_data_str = file_data.get("file_data", "")
        if file_data_str.startswith("data:"):
            mime, _, b64 = file_data_str.partition(";base64,")
            mime = mime.replace("data:", "")
            return [{"inline_data": {"mime_type": mime, "data": b64}}]

    return []
