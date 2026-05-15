"""Open a streaming Claude Messages transport with Anthropic-aligned retries."""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq_anthropic._shared.retry import call_with_retry
from nucleusiq_anthropic.nb_anthropic.messages import build_create_kwargs

logger = logging.getLogger(__name__)


async def open_messages_stream(
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
) -> Any:
    """Return SDK stream object — async iterable or synchronous iterator."""

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
        stream=True,
    )

    async def api_call_async() -> Any:
        return await client.messages.create(**kw)

    def api_call_sync() -> Any:
        return client.messages.create(**kw)

    factory = api_call_async if async_mode else api_call_sync

    return await call_with_retry(
        factory,
        max_retries=max_retries,
        async_mode=async_mode,
        logger=logger,
    )
