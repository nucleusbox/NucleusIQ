"""Claude incremental streaming → :class:`~nucleusiq.streaming.events.StreamEvent`."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from nucleusiq.llms.errors import LLMError
from nucleusiq.streaming.events import StreamEvent

from nucleusiq_anthropic.nb_anthropic.stream_create import open_messages_stream

logger = logging.getLogger(__name__)

_STREAM_END = object()


class _StreamError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def _rerasable_exceptions() -> tuple[type[BaseException], ...]:
    from nucleusiq.llms.errors import (
        AuthenticationError,
        ContentFilterError,
        ContextLengthError,
        InvalidRequestError,
        ModelNotFoundError,
        PermissionDeniedError,
        ProviderConnectionError,
        ProviderServerError,
        RateLimitError,
    )

    return (
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


async def _sync_iter_to_async(sync_iterable: Any) -> AsyncGenerator[Any, None]:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def _worker() -> None:
        try:
            for item in sync_iterable:
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


async def _unify_stream(
    raw_stream: Any,
    *,
    async_mode: bool,
) -> AsyncGenerator[Any, None]:
    """Normalise async + sync Anthropic stream iterators."""

    if async_mode:
        async for ev in raw_stream:
            yield ev
    else:
        async for ev in _sync_iter_to_async(raw_stream):
            yield ev


def _usage_payload(usage_obj: Any) -> dict[str, int]:
    inp = int(getattr(usage_obj, "input_tokens", None) or 0)
    outp = int(getattr(usage_obj, "output_tokens", None) or 0)
    cre = int(getattr(usage_obj, "cache_creation_input_tokens", None) or 0)
    crd = int(getattr(usage_obj, "cache_read_input_tokens", None) or 0)
    return {
        "prompt_tokens": inp + cre + crd,
        "completion_tokens": outp,
        "total_tokens": inp + outp + cre + crd,
    }


def _finalize_tool_state(tool_state: dict[str, Any]) -> dict[str, Any]:
    raw_js = "".join(tool_state.get("chunks") or [])
    args = raw_js.strip() if raw_js.strip() else "{}"
    tc_id = str(tool_state.get("id") or "")
    name = str(tool_state.get("name") or "")
    try:
        json.loads(args)
    except json.JSONDecodeError:
        args = "{}"
    return {"id": tc_id, "name": name, "arguments": args}


async def _process_raw_events(
    chunk_iter: AsyncGenerator[Any, None],
) -> AsyncGenerator[StreamEvent, None]:
    block_state: dict[int, dict[str, Any]] = {}
    text_parts: list[str] = []
    usage_dict: dict[str, int] | None = None
    response_id: str | None = None
    model_label: str | None = None

    async for event in chunk_iter:
        etype = getattr(event, "type", None)

        if etype == "message_start":
            msg = getattr(event, "message", None)
            if msg is not None:
                response_id = getattr(msg, "id", None)
                msg_model_name = getattr(msg, "model", None)
                model_label = (
                    str(msg_model_name) if msg_model_name is not None else None
                )

        elif etype == "content_block_start":
            idx = int(getattr(event, "index", 0))
            blk = getattr(event, "content_block", None)
            b_kind = getattr(blk, "type", None) if blk is not None else None
            if b_kind == "text":
                block_state[idx] = {"kind": "text"}
            elif b_kind == "tool_use":
                block_state[idx] = {
                    "kind": "tool",
                    "id": getattr(blk, "id", "") or "",
                    "name": getattr(blk, "name", "") or "",
                    "chunks": [],
                }

        elif etype == "content_block_delta":
            idx = int(getattr(event, "index", 0))
            delta = getattr(event, "delta", None)
            if delta is None:
                continue

            dt = getattr(delta, "type", None)
            if dt == "text_delta":
                piece = getattr(delta, "text", "") or ""
                if piece:
                    text_parts.append(piece)
                    yield StreamEvent.token_event(piece)

            elif dt == "thinking_delta":
                think = getattr(delta, "thinking", "") or ""
                if think.strip():
                    yield StreamEvent.thinking_event(think)

            elif dt == "input_json_delta":
                chunk = getattr(delta, "partial_json", "") or ""
                slot = block_state.get(idx)
                if isinstance(slot, dict) and chunk:
                    lst = slot.setdefault("chunks", [])
                    lst.append(chunk)

        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                usage_dict = _usage_payload(usage)

    tool_calls_sorted: list[dict[str, Any]] = []

    def _numeric_index(key: Any) -> int:
        try:
            return int(key)
        except (TypeError, ValueError):
            return 0

    for bi in sorted(block_state, key=_numeric_index):
        st = block_state[bi]
        if str(st.get("kind", "")) == "tool":
            tool_calls_sorted.append(_finalize_tool_state(st))

    full_text = "".join(text_parts)
    md: dict[str, Any] = {
        "usage": usage_dict or {},
        "tool_calls": tool_calls_sorted,
        "response_id": response_id,
        "model": model_label,
    }
    yield StreamEvent.complete_event(full_text, metadata=md)


async def stream_messages(
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
) -> AsyncGenerator[StreamEvent, None]:
    """Yield ``TOKEN`` chunks and a terminating ``COMPLETE``."""

    rer = _rerasable_exceptions()

    try:
        raw_stream = await open_messages_stream(
            client,
            async_mode=async_mode,
            max_retries=max_retries,
            model=model,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            merged_extras=merged_extras,
            extra_headers=extra_headers,
        )

        unified = _unify_stream(raw_stream, async_mode=async_mode)
        async for evt in _process_raw_events(unified):
            yield evt

    except rer:
        raise
    except LLMError as exc:
        logger.error("Anthropic streaming LLM error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))
    except Exception as exc:
        logger.error("Anthropic streaming error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))
