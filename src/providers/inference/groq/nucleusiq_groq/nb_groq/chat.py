"""Chat Completions backend — single responsibility: call API + normalize."""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq_groq._shared.response_models import (
    AssistantMessage,
    GroqLLMResponse,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
    _Choice,
)
from nucleusiq_groq._shared.retry import call_with_retry
from nucleusiq_groq._shared.wire import build_chat_completion_payload

logger = logging.getLogger(__name__)


def normalize_chat_response(raw: Any) -> GroqLLMResponse:
    """Map an OpenAI SDK chat completion object to :class:`GroqLLMResponse`."""
    choices_out: list[_Choice] = []
    choices_raw = getattr(raw, "choices", None) or []
    for ch in choices_raw:
        msg = getattr(ch, "message", None)
        if msg is None:
            continue
        content = getattr(msg, "content", None)
        tool_calls_out: list[ToolCall] | None = None
        raw_tc = getattr(msg, "tool_calls", None)
        if raw_tc:
            tool_calls_out = []
            for tc in raw_tc:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                tool_calls_out.append(
                    ToolCall(
                        id=getattr(tc, "id", "") or "",
                        type=getattr(tc, "type", None) or "function",
                        function=ToolCallFunction(
                            name=getattr(fn, "name", "") or "",
                            arguments=getattr(fn, "arguments", "") or "",
                        ),
                    )
                )
        choices_out.append(
            _Choice(
                message=AssistantMessage(
                    content=content,
                    tool_calls=tool_calls_out,
                )
            )
        )

    usage_out: UsageInfo | None = None
    usage = getattr(raw, "usage", None)
    if usage:
        comp_details = getattr(usage, "completion_tokens_details", None)
        usage_out = UsageInfo(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            reasoning_tokens=(
                getattr(comp_details, "reasoning_tokens", 0) or 0 if comp_details else 0
            ),
        )

    return GroqLLMResponse(
        choices=choices_out,
        usage=usage_out,
        model=getattr(raw, "model", None),
        response_id=getattr(raw, "id", None),
    )


async def create_chat_completion(
    client: Any,
    *,
    async_mode: bool,
    max_retries: int,
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    temperature: float | None,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    stop: list[str] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    response_format: dict[str, Any] | None,
    parallel_tool_calls: bool | None,
    seed: int | None,
    user: str | None,
    extra: dict[str, Any],
) -> GroqLLMResponse:
    """Run ``chat.completions.create`` with retries and return normalised result."""
    payload = build_chat_completion_payload(
        model=model,
        messages=messages,
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        parallel_tool_calls=parallel_tool_calls,
        seed=seed,
        user=user,
        extra=extra,
    )

    async def api_call() -> Any:
        if async_mode:
            return await client.chat.completions.create(**payload)
        return client.chat.completions.create(**payload)

    raw = await call_with_retry(
        api_call,
        max_retries=max_retries,
        async_mode=async_mode,
        logger=logger,
    )
    return normalize_chat_response(raw)
