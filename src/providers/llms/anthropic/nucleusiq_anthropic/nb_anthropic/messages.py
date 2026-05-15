"""Messages backend — normalize SDK payloads and invoke ``messages.create``."""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import NOT_GIVEN

from nucleusiq_anthropic._shared.response_models import (
    AnthropicLLMResponse,
    AssistantMessage,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
    _Choice,
)
from nucleusiq_anthropic._shared.retry import call_with_retry
from nucleusiq_anthropic._shared.wire import (
    anthropic_tool_choice,
    drop_unsupported_sampling,
    flatten_tools,
    translate_messages,
)

logger = logging.getLogger(__name__)


def _drop_conflicting_sampling(kw: dict[str, Any]) -> None:
    """Claude rejects ``messages.create`` when both ``temperature`` and ``top_p`` are set."""

    temperature = kw.get("temperature")
    top_p = kw.get("top_p")
    has_temp = temperature is not NOT_GIVEN and temperature is not None
    has_top_p = top_p is not NOT_GIVEN and top_p is not None
    if has_temp and has_top_p:
        kw["top_p"] = NOT_GIVEN


def normalize_message_response(raw: Any) -> AnthropicLLMResponse:
    """Map Claude ``Message`` → :class:`AnthropicLLMResponse` (Choices contract)."""

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in getattr(raw, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            t = getattr(block, "text", None)
            if t:
                text_parts.append(str(t))
        elif btype == "tool_use":
            bid = getattr(block, "id", "") or ""
            name = getattr(block, "name", "") or ""
            inp = getattr(block, "input", None)
            if isinstance(inp, dict):
                payload = inp
            else:
                payload = {"value": inp}
            try:
                arguments = json.dumps(payload, default=str)
            except (TypeError, ValueError):
                arguments = "{}"
            tool_calls.append(
                ToolCall(
                    id=bid,
                    type="function",
                    function=ToolCallFunction(name=name, arguments=arguments),
                ),
            )

    merged_text = "\n".join(text_parts).strip() or None
    assistant = AssistantMessage(
        content=merged_text,
        tool_calls=tool_calls if tool_calls else None,
    )

    usage_out: UsageInfo | None = None
    usage = getattr(raw, "usage", None)
    if usage is not None:
        inp = getattr(usage, "input_tokens", None) or 0
        out_t = getattr(usage, "output_tokens", None) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", None) or 0
        usage_out = UsageInfo(
            prompt_tokens=int(inp) + int(cache_create) + int(cache_read),
            completion_tokens=int(out_t),
            total_tokens=int(inp) + int(out_t) + int(cache_read) + int(cache_create),
            cached_tokens=int(cache_read),
        )

    return AnthropicLLMResponse(
        choices=[_Choice(message=assistant)],
        usage=usage_out,
        model=str(getattr(raw, "model", None) or "") or None,
        response_id=getattr(raw, "id", None),
    )


def build_create_kwargs(
    *,
    model: str,
    framework_messages: list[dict[str, Any]],
    max_output_tokens: int,
    temperature: float | None,
    top_p: float | None,
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    merged_extras: dict[str, Any],
    extra_headers: dict[str, str] | None,
    stream: bool = False,
) -> dict[str, Any]:
    """Assemble keyword arguments for ``Anthropic.messages.create``."""

    system, msgs = translate_messages(framework_messages)
    claude_tools = flatten_tools(tools)
    mapped_choice = anthropic_tool_choice(tool_choice)

    clean_extras = drop_unsupported_sampling(merged_extras)
    beta_str = clean_extras.pop("anthropic_beta", None)

    kw: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "messages": msgs,
        "temperature": temperature if temperature is not None else NOT_GIVEN,
        "top_p": top_p if top_p is not None else NOT_GIVEN,
        "stop_sequences": stop or NOT_GIVEN,
        "tools": claude_tools or NOT_GIVEN,
        "tool_choice": mapped_choice if mapped_choice is not None else NOT_GIVEN,
        "stream": stream,
    }

    if system is not None:
        kw["system"] = system

    headers: dict[str, str] | None = None
    if extra_headers:
        headers = dict(extra_headers)

    if isinstance(beta_str, str) and beta_str.strip():
        hs = beta_str.strip()
        headers = dict(headers or {})
        headers.setdefault("anthropic-beta", hs)

    for key in ("model", "max_tokens", "messages", "stream"):
        clean_extras.pop(key, None)

    for key, val in clean_extras.items():
        if val is NOT_GIVEN or val is None:
            continue
        kw[key] = val

    _drop_conflicting_sampling(kw)

    if headers:
        kw["extra_headers"] = headers

    return kw


async def create_message_response(
    client: Any,
    *,
    async_mode: bool,
    max_retries: int,
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    temperature: float | None,
    top_p: float | None,
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    merged_extras: dict[str, Any],
    extra_headers: dict[str, str] | None,
) -> AnthropicLLMResponse:
    """Non-stream call with retries."""

    kw = build_create_kwargs(
        model=model,
        framework_messages=messages,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        merged_extras=dict(merged_extras),
        extra_headers=extra_headers,
        stream=False,
    )

    async def api_call() -> Any:
        return await client.messages.create(**kw)

    def api_call_sync() -> Any:
        return client.messages.create(**kw)

    factory = api_call if async_mode else api_call_sync

    raw = await call_with_retry(
        factory,
        max_retries=max_retries,
        async_mode=async_mode,
        logger=logger,
    )
    return normalize_message_response(raw)
