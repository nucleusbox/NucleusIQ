"""Streaming adapters for OpenAI APIs.

Converts OpenAI-specific streaming chunks/events into the framework's
``StreamEvent`` objects.

**Responsibility** (Single Responsibility):
    Convert raw OpenAI streaming data → ``StreamEvent`` objects.

**What this module does NOT do:**
    - Build API payloads (caller's job)
    - Handle retries (streaming is not retryable mid-stream)
    - Manage conversation state (``BaseOpenAI``'s job)

**Two adapters, one per API surface:**
    ``stream_chat_completions`` — Chat Completions API (``chat.completions.create``)
    ``stream_responses_api``   — Responses API (``responses.create``)

Both are async generators that yield ``StreamEvent`` objects regardless
of whether the underlying OpenAI client is async or sync.  The sync ↔
async bridging is handled by ``_sync_iter_to_async``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from nucleusiq.streaming.events import StreamEvent

logger = logging.getLogger(__name__)


# ====================================================================== #
# Sync → Async bridge                                                     #
# ====================================================================== #

_STREAM_END = object()


class _StreamError:
    """Sentinel wrapper to distinguish exceptions from real chunks."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


async def _sync_iter_to_async(
    iterable_factory: Any,
) -> AsyncGenerator[Any, None]:
    """Bridge a sync iterable into an async generator via a background thread.

    The sync iterable is consumed in a thread-pool worker; items are
    pushed onto an ``asyncio.Queue`` so the caller can ``async for``
    over them without blocking the event loop.
    """
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


# ====================================================================== #
# Chat Completions streaming adapter                                      #
# ====================================================================== #


def _merge_tool_call_delta(
    accumulated: dict[int, dict[str, Any]],
    delta_tool_calls: list[Any],
) -> None:
    """Merge incremental tool-call deltas into *accumulated*.

    OpenAI streams tool calls as partial fragments spread across
    multiple chunks, keyed by ``index``.
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


def _extract_usage_dict(usage: Any) -> dict[str, int]:
    """Extract token counts from an OpenAI SDK usage object."""
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
    """Core processor: convert Chat Completions chunks → StreamEvent.

    Separated from the iterator creation so it works identically for
    async and (bridged) sync streams.
    """
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
) -> AsyncGenerator[StreamEvent, None]:
    """Stream a Chat Completions API call as ``StreamEvent`` objects.

    Args:
        client: ``openai.AsyncOpenAI`` or ``openai.OpenAI`` instance.
        payload: Ready-to-send kwargs for ``chat.completions.create``.
        async_mode: Whether *client* is async.

    Yields:
        ``StreamEvent`` — ``TOKEN`` events followed by one ``COMPLETE``.
    """
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    try:
        if async_mode:
            raw_stream = await client.chat.completions.create(**payload)
            async for event in _process_chat_chunks(raw_stream):
                yield event
        else:
            sync_stream = client.chat.completions.create(**payload)
            async_stream = _sync_iter_to_async(sync_stream)
            async for event in _process_chat_chunks(async_stream):
                yield event
    except Exception as exc:
        logger.error("Chat Completions stream error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))


# ====================================================================== #
# Responses API streaming adapter                                         #
# ====================================================================== #


async def _process_responses_events(
    event_iter: AsyncGenerator[Any, None],
) -> AsyncGenerator[StreamEvent, None]:
    """Core processor: convert Responses API events → StreamEvent.

    Responses API event types handled:
        response.output_text.delta             → TOKEN
        response.output_item.added (fn_call)   → accumulate tool call
        response.function_call_arguments.delta → accumulate arguments
        response.function_call_arguments.done  → finalize tool call
        response.completed                     → extract usage / resp_id
    """
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    current_fn: dict[str, Any] | None = None
    usage: dict[str, int] | None = None
    resp_id: str | None = None

    async for event in event_iter:
        event_type = getattr(event, "type", "")

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                content_parts.append(delta)
                yield StreamEvent.token_event(delta)

        elif event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", None) == "function_call":
                current_fn = {
                    "call_id": getattr(item, "call_id", ""),
                    "name": getattr(item, "name", ""),
                    "arguments": "",
                }

        elif event_type == "response.function_call_arguments.delta":
            delta = getattr(event, "delta", "")
            if current_fn and delta:
                current_fn["arguments"] += delta

        elif event_type == "response.function_call_arguments.done":
            if current_fn:
                tool_calls.append(current_fn)
                current_fn = None

        elif event_type == "response.completed":
            resp = getattr(event, "response", None)
            if resp:
                resp_id = getattr(resp, "id", None)
                raw_usage = getattr(resp, "usage", None)
                if raw_usage:
                    usage = {
                        "input_tokens": getattr(raw_usage, "input_tokens", 0) or 0,
                        "output_tokens": getattr(raw_usage, "output_tokens", 0) or 0,
                        "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
                    }

    full_content = "".join(content_parts) or None
    metadata: dict[str, Any] = {}

    if tool_calls:
        metadata["tool_calls"] = [
            {
                "id": tc["call_id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"] or "{}",
                },
            }
            for tc in tool_calls
        ]
    if usage:
        metadata["usage"] = usage
    if resp_id:
        metadata["response_id"] = resp_id

    yield StreamEvent.complete_event(full_content or "", metadata=metadata or None)


async def stream_responses_api(
    client: Any,
    payload: dict[str, Any],
    *,
    async_mode: bool,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream a Responses API call as ``StreamEvent`` objects.

    Args:
        client: ``openai.AsyncOpenAI`` or ``openai.OpenAI`` instance.
        payload: Ready-to-send kwargs for ``responses.create``.
        async_mode: Whether *client* is async.

    Yields:
        ``StreamEvent`` — ``TOKEN`` events followed by one ``COMPLETE``.
    """
    payload["stream"] = True

    try:
        if async_mode:
            raw_stream = await client.responses.create(**payload)
            async for event in _process_responses_events(raw_stream):
                yield event
        else:
            sync_stream = client.responses.create(**payload)
            async_stream = _sync_iter_to_async(sync_stream)
            async for event in _process_responses_events(async_stream):
                yield event
    except Exception as exc:
        logger.error("Responses API stream error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))
