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
    _Choice,
    _LLMResponse,
)
from nucleusiq_openai._shared.retry import call_with_retry


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
                async for chunk in client.chat.completions.create(**payload):
                    first = chunk.choices[0].delta
                    msg = _sdk_message_to_assistant(first.model_dump())
                    return _LLMResponse(choices=[_Choice(message=msg)])
            resp = await client.chat.completions.create(**payload)
            msg = _sdk_message_to_assistant(resp.choices[0].message.model_dump())
            return _LLMResponse(choices=[_Choice(message=msg)])

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
                chunks = []
                for chunk in client.chat.completions.create(**payload):
                    chunks.append(chunk)
                if chunks:
                    first = chunks[0].choices[0].delta
                    msg = _sdk_message_to_assistant(first.model_dump())
                    return _LLMResponse(choices=[_Choice(message=msg)])
            resp = client.chat.completions.create(**payload)
            msg = _sdk_message_to_assistant(resp.choices[0].message.model_dump())
            return _LLMResponse(choices=[_Choice(message=msg)])

        return await call_with_retry(
            _call_sync,
            max_retries=max_retries,
            async_mode=False,
            logger=_log,
            on_bad_request=_on_bad_request,
        )
