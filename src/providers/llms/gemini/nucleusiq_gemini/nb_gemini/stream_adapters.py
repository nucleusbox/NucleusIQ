"""Streaming adapters for the Gemini API.

Converts Gemini-specific streaming chunks into the framework's
``StreamEvent`` objects.

**Responsibility** (Single Responsibility):
    Convert raw Gemini streaming data → ``StreamEvent`` objects.

**What this module does NOT do:**
    - Build API payloads (caller's job)
    - Handle retries (streaming is not retryable mid-stream)
    - Manage conversation state (``BaseGemini``'s job)

The adapter is an async generator that yields ``StreamEvent`` objects.
Since the Gemini SDK's ``generate_content_stream`` returns a sync
iterator, we bridge it to async via ``_sync_iter_to_async``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
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
    iterable: Any,
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
            for item in iterable:
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
# Gemini streaming adapter                                                 #
# ====================================================================== #


async def _process_gemini_chunks(
    chunk_iter: AsyncGenerator[Any, None],
) -> AsyncGenerator[StreamEvent, None]:
    """Core processor: convert Gemini streaming chunks → StreamEvent.

    Handles:
    - Text deltas → TOKEN events
    - Function calls → accumulated and emitted in COMPLETE metadata
    - Thinking parts → THINKING events
    - Usage metadata → included in COMPLETE metadata
    """
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None

    async for chunk in chunk_iter:
        candidates = getattr(chunk, "candidates", None) or []
        usage_meta = getattr(chunk, "usage_metadata", None)

        if usage_meta:
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(usage_meta, "candidates_token_count", 0)
                or 0,
                "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
            }

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            parts = getattr(content, "parts", None) or []
            for part in parts:
                thought = getattr(part, "thought", None)
                if thought:
                    yield StreamEvent.thinking_event(str(thought))
                    continue

                text = getattr(part, "text", None)
                if text:
                    content_parts.append(text)
                    yield StreamEvent.token_event(text)

                fn_call = getattr(part, "function_call", None)
                if fn_call:
                    name = getattr(fn_call, "name", "") or ""
                    args = getattr(fn_call, "args", None) or {}
                    call_id = getattr(fn_call, "id", None) or str(uuid.uuid4())

                    args_str = json.dumps(args) if isinstance(args, dict) else str(args)

                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str,
                            },
                        }
                    )

    full_content = "".join(content_parts) or None
    metadata: dict[str, Any] = {}

    if tool_calls:
        metadata["tool_calls"] = tool_calls
    if usage:
        metadata["usage"] = usage

    yield StreamEvent.complete_event(full_content or "", metadata=metadata or None)


async def stream_gemini(
    sync_stream: Any,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream a Gemini API call as ``StreamEvent`` objects.

    Args:
        sync_stream: Sync iterator from ``client.models.generate_content_stream()``.

    Yields:
        ``StreamEvent`` — ``TOKEN`` events followed by one ``COMPLETE``.
    """
    try:
        async_stream = _sync_iter_to_async(sync_stream)
        async for event in _process_gemini_chunks(async_stream):
            yield event
    except Exception as exc:
        logger.error("Gemini stream error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))
