"""Chat Completions API backend.

Builds the payload, dispatches the call (sync or async), and returns
a normalised ``_LLMResponse``.  Retry logic is delegated to
``_shared.retry.call_with_retry``.
"""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq_openai._shared.models import ChatCompletionsPayload
from nucleusiq_openai._shared.response_models import (
    AssistantMessage,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
    _Choice,
    _LLMResponse,
)
from nucleusiq_openai._shared.retry import call_with_retry


def _merge_tool_call_delta(
    accumulated: dict[int, dict[str, Any]],
    delta_tool_calls: list[Any],
) -> None:
    """Merge incremental tool-call deltas into the accumulated map.

    OpenAI streams tool calls as partial fragments spread across multiple
    chunks, keyed by ``index``.  Each chunk may add to the function name
    or arguments string.
    """
    for tc in delta_tool_calls:
        idx = getattr(tc, "index", 0)
        if idx not in accumulated:
            accumulated[idx] = {
                "id": getattr(tc, "id", "") or "",
                "type": getattr(tc, "type", "function") or "function",
                "function": {"name": "", "arguments": ""},
            }
        entry = accumulated[idx]
        if getattr(tc, "id", None):
            entry["id"] = tc.id
        fn_delta = getattr(tc, "function", None)
        if fn_delta:
            if getattr(fn_delta, "name", None):
                entry["function"]["name"] += fn_delta.name
            if getattr(fn_delta, "arguments", None):
                entry["function"]["arguments"] += fn_delta.arguments


def _build_accumulated_response(
    content_parts: list[str],
    tool_call_map: dict[int, dict[str, Any]],
) -> _LLMResponse:
    """Build a complete ``_LLMResponse`` from accumulated stream data."""
    raw: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
        "tool_calls": (
            [tool_call_map[i] for i in sorted(tool_call_map)] if tool_call_map else None
        ),
    }
    msg = _sdk_message_to_assistant(raw)
    return _LLMResponse(choices=[_Choice(message=msg)])


async def _accumulate_stream_async(stream_iter: Any) -> _LLMResponse:
    """Consume an async streaming response and return the full accumulated result."""
    content_parts: list[str] = []
    tool_call_map: dict[int, dict[str, Any]] = {}
    async for chunk in stream_iter:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        if getattr(delta, "tool_calls", None):
            _merge_tool_call_delta(tool_call_map, delta.tool_calls)
    return _build_accumulated_response(content_parts, tool_call_map)


def _accumulate_stream_sync(stream_iter: Any) -> _LLMResponse:
    """Consume a sync streaming response and return the full accumulated result."""
    content_parts: list[str] = []
    tool_call_map: dict[int, dict[str, Any]] = {}
    for chunk in stream_iter:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        if getattr(delta, "tool_calls", None):
            _merge_tool_call_delta(tool_call_map, delta.tool_calls)
    return _build_accumulated_response(content_parts, tool_call_map)


def _extract_usage(resp: Any) -> UsageInfo | None:
    """Extract token usage from an OpenAI SDK response object."""
    usage = getattr(resp, "usage", None)
    if not usage:
        return None
    comp_details = getattr(usage, "completion_tokens_details", None)
    return UsageInfo(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        reasoning_tokens=(
            getattr(comp_details, "reasoning_tokens", 0) or 0 if comp_details else 0
        ),
    )


def _extract_response_metadata(resp: Any) -> dict[str, Any]:
    """Extract top-level metadata (id, model, created, etc.) from SDK response.

    Only includes values that are primitive types to avoid passing through
    mock objects or unexpected SDK wrapper types.
    """
    raw = {
        "response_id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "created": getattr(resp, "created", None),
        "service_tier": getattr(resp, "service_tier", None),
        "system_fingerprint": getattr(resp, "system_fingerprint", None),
    }
    return {k: v for k, v in raw.items() if isinstance(v, (str, int, float))}


def _sdk_message_to_assistant(raw: dict[str, Any]) -> AssistantMessage:
    """Convert an OpenAI SDK ``message.model_dump()`` dict to ``AssistantMessage``."""
    raw_tcs = raw.get("tool_calls")
    typed_tcs = None
    if raw_tcs:
        typed_tcs = [
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=ToolCallFunction(
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", "{}"),
                ),
            )
            for tc in raw_tcs
        ]
    return AssistantMessage(
        role=raw.get("role", "assistant"),
        content=raw.get("content"),
        tool_calls=typed_tcs,
    )


async def call_chat_completions(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    default_temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: list[str] | None = None,
    stream: bool = False,
    response_format: dict[str, Any] | None = None,
    logit_bias: dict[str, float] | None = None,
    max_retries: int = 3,
    async_mode: bool = True,
    logger: logging.Logger | None = None,
    **extra: Any,
) -> _LLMResponse:
    """Call OpenAI **Chat Completions API** (``chat.completions.create``).

    Returns:
        Normalised ``_LLMResponse`` with the first choice.
    """
    _log = logger or logging.getLogger(__name__)

    payload_model = ChatCompletionsPayload.build(
        model=model,
        messages=messages,
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
        default_temperature=default_temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        response_format=response_format,
        logit_bias=logit_bias,
        tools=tools,
        tool_choice=tool_choice,
        **extra,
    )
    payload = payload_model.to_api_kwargs()

    # Mutable state for the bad-request retry callback
    _tool_choice_holder = {"value": tool_choice}

    def _on_bad_request() -> bool:
        if _tool_choice_holder["value"] is not None and "tool_choice" in payload:
            _log.warning("Invalid request — retrying without tool_choice")
            payload.pop("tool_choice", None)
            _tool_choice_holder["value"] = None
            return True
        return False

    if async_mode:

        async def _call() -> _LLMResponse:
            if stream:
                return await _accumulate_stream_async(
                    client.chat.completions.create(**payload)
                )
            resp = await client.chat.completions.create(**payload)
            msg = _sdk_message_to_assistant(resp.choices[0].message.model_dump())
            return _LLMResponse(
                choices=[_Choice(message=msg)],
                usage=_extract_usage(resp),
                **_extract_response_metadata(resp),
            )

        return await call_with_retry(
            _call,
            max_retries=max_retries,
            async_mode=True,
            logger=_log,
            on_bad_request=_on_bad_request,
        )
    else:

        def _call_sync() -> _LLMResponse:
            if stream:
                return _accumulate_stream_sync(
                    client.chat.completions.create(**payload)
                )
            resp = client.chat.completions.create(**payload)
            msg = _sdk_message_to_assistant(resp.choices[0].message.model_dump())
            return _LLMResponse(
                choices=[_Choice(message=msg)],
                usage=_extract_usage(resp),
                **_extract_response_metadata(resp),
            )

        return await call_with_retry(
            _call_sync,
            max_retries=max_retries,
            async_mode=False,
            logger=_log,
            on_bad_request=_on_bad_request,
        )
