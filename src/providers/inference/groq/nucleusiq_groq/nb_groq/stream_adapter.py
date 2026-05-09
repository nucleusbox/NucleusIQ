"""Groq Chat Completions streaming → :class:`nucleusiq.streaming.events.StreamEvent`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from nucleusiq.llms.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    ModelNotFoundError,
    PermissionDeniedError,
    ProviderConnectionError,
    ProviderServerError,
    RateLimitError,
)
from nucleusiq.streaming.events import StreamEvent

from nucleusiq_groq._shared.stream_create import open_streaming_completion

logger = logging.getLogger(__name__)

_STREAM_OPEN_RERAISE: tuple[type[BaseException], ...] = (
    RateLimitError,
    AuthenticationError,
    PermissionDeniedError,
    ModelNotFoundError,
    InvalidRequestError,
    ContentFilterError,
    ContextLengthError,
    ProviderServerError,
    ProviderConnectionError,
)

_STREAM_END = object()


class _StreamError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


async def _sync_iter_to_async(
    iterable_factory: Any,
) -> AsyncGenerator[Any, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def _worker() -> None:
        try:
            for item in iterable_factory:
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as exc:
            loop.call_soon_threadsafe(queue.put_nowait, _StreamError(exc))
        loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)

    fut = loop.run_in_executor(None, _worker)
    try:
        while True:
            item = await queue.get()
            if item is _STREAM_END:
                break
            if isinstance(item, _StreamError):
                raise item.exc
            yield item
    finally:
        await fut


def _merge_tool_call_delta(
    accumulated: dict[int, dict[str, Any]],
    delta_tool_calls: list[Any],
) -> None:
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


def _extract_usage_dict(usage: Any) -> dict[str, int]:
    result: dict[str, int] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    comp_details = getattr(usage, "completion_tokens_details", None)
    if comp_details:
        result["reasoning_tokens"] = getattr(comp_details, "reasoning_tokens", 0) or 0
    return result


async def _process_chat_chunks(
    chunk_iter: AsyncGenerator[Any, None],
) -> AsyncGenerator[StreamEvent, None]:
    content_parts: list[str] = []
    tool_call_map: dict[int, dict[str, Any]] = {}
    usage: dict[str, int] | None = None

    async for chunk in chunk_iter:
        if getattr(chunk, "usage", None):
            usage = _extract_usage_dict(chunk.usage)

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        text = getattr(delta, "content", None)
        if text:
            content_parts.append(text)
            yield StreamEvent.token_event(text)

        tc_deltas = getattr(delta, "tool_calls", None)
        if tc_deltas:
            _merge_tool_call_delta(tool_call_map, tc_deltas)

    full_content = "".join(content_parts) or None
    metadata: dict[str, Any] = {}
    if tool_call_map:
        metadata["tool_calls"] = [tool_call_map[i] for i in sorted(tool_call_map)]
    if usage:
        metadata["usage"] = usage

    yield StreamEvent.complete_event(full_content or "", metadata=metadata or None)


async def stream_chat_completions(
    client: Any,
    payload: dict[str, Any],
    *,
    async_mode: bool,
    max_retries: int = 3,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream ``chat.completions.create`` as ``StreamEvent`` objects.

    Opening the stream uses the same ``call_with_retry`` policy as non-streaming
    chat (including **429** + ``Retry-After``). Selected framework errors
    (**auth**, **rate limit after retries**, **model missing**, etc.) **propagate**
    to match ``call()`` semantics; other :class:`LLMError` cases (e.g. unexpected
    SDK wrap) and bare exceptions become ``StreamEvent`` error events.
    """
    try:
        raw = await open_streaming_completion(
            client,
            payload,
            async_mode=async_mode,
            max_retries=max_retries,
        )
        if async_mode:
            async for event in _process_chat_chunks(raw):
                yield event
        else:
            async_stream = _sync_iter_to_async(raw)
            async for event in _process_chat_chunks(async_stream):
                yield event
    except _STREAM_OPEN_RERAISE:
        raise
    except LLMError as exc:
        logger.error("Groq stream error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))
    except Exception as exc:
        logger.error("Groq stream error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))
