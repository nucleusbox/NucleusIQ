"""Ollama ``/api/chat`` — call SDK and normalise to :class:`OllamaLLMResponse`."""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq_ollama._shared.response_models import (
    AssistantMessage,
    OllamaLLMResponse,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
    _Choice,
)
from nucleusiq_ollama._shared.retry import call_with_retry
from nucleusiq_ollama._shared.wire import (
    ThinkLevel,
    build_chat_kwargs,
    build_options,
    tool_arguments_to_json_string,
)

logger = logging.getLogger(__name__)


def normalize_chat_response(raw: Any) -> OllamaLLMResponse:
    """Map an Ollama :class:`ollama.ChatResponse` to :class:`OllamaLLMResponse`."""
    msg = getattr(raw, "message", None)
    choices_out: list[_Choice] = []
    if msg is not None:
        content = getattr(msg, "content", None)
        thinking = getattr(msg, "thinking", None)
        tool_calls_out: list[ToolCall] | None = None
        raw_tc = getattr(msg, "tool_calls", None)
        if raw_tc:
            tool_calls_out = []
            for i, tc in enumerate(raw_tc):
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args = getattr(fn, "arguments", None)
                tool_calls_out.append(
                    ToolCall(
                        id=str(getattr(tc, "id", None) or i),
                        type="function",
                        function=ToolCallFunction(
                            name=getattr(fn, "name", "") or "",
                            arguments=tool_arguments_to_json_string(args),
                        ),
                    )
                )
        choices_out.append(
            _Choice(
                message=AssistantMessage(
                    content=content,
                    thinking=thinking,
                    tool_calls=tool_calls_out,
                )
            )
        )

    usage_out: UsageInfo | None = None
    pe = getattr(raw, "prompt_eval_count", None)
    ev = getattr(raw, "eval_count", None)
    if pe is not None or ev is not None:
        pt = int(pe or 0)
        ct = int(ev or 0)
        usage_out = UsageInfo(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            reasoning_tokens=0,
        )

    return OllamaLLMResponse(
        choices=choices_out,
        usage=usage_out,
        model=getattr(raw, "model", None),
        response_id=getattr(raw, "created_at", None),
    )


async def create_ollama_chat(
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
    format_payload: str | dict[str, Any] | None,
    think: bool | ThinkLevel | None,
    keep_alive: float | str | None,
    seed: int | None,
) -> OllamaLLMResponse:
    """Execute a non-stream chat with retries."""
    options = build_options(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        seed=seed,
    )
    kwargs = build_chat_kwargs(
        model=model,
        messages=messages,
        tools=tools,
        format_payload=format_payload,
        options=options,
        think=think,
        keep_alive=keep_alive,
        stream=False,
        tool_choice=tool_choice,
    )

    async def api_call() -> Any:
        if async_mode:
            return await client.chat(**kwargs)
        return client.chat(**kwargs)

    raw = await call_with_retry(
        api_call,
        max_retries=max_retries,
        async_mode=async_mode,
        logger=logger,
    )
    return normalize_chat_response(raw)
