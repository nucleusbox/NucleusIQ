"""Ollama streaming ``/api/chat`` → :class:`nucleusiq.streaming.events.StreamEvent`."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Mapping

from nucleusiq.llms.errors import LLMError
from nucleusiq.streaming.events import StreamEvent
from ollama import ResponseError

from nucleusiq_ollama._shared.errors import map_ollama_response_error
from nucleusiq_ollama._shared.wire import ThinkLevel, build_chat_kwargs, build_options

logger = logging.getLogger(__name__)

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
            for item in iterable_factory():
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


def _merge_tool_calls_incremental(
    acc: list[dict[str, Any]],
    new_tcs: Sequence[Any] | None,
) -> None:
    if not new_tcs:
        return
    for i, tc in enumerate(new_tcs):
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", None) or ""
        args = getattr(fn, "arguments", None)
        while len(acc) <= i:
            acc.append(
                {
                    "id": str(i),
                    "type": "function",
                    "function": {"name": "", "arguments": {}},
                }
            )
        entry = acc[i]
        if name:
            entry["function"]["name"] = name
        if isinstance(args, Mapping):
            cur = entry["function"]["arguments"]
            if not isinstance(cur, dict):
                cur = {}
            merged = {**cur, **dict(args)}
            entry["function"]["arguments"] = merged


def _tool_calls_for_metadata(acc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in acc:
        fn = dict(entry.get("function") or {})
        args = fn.get("arguments")
        if isinstance(args, Mapping):
            fn = {
                **fn,
                "arguments": json.dumps(args, ensure_ascii=False),
            }
        out.append({**entry, "function": fn})
    return out


async def _open_chat_stream(
    client: Any, kwargs: dict[str, Any], *, async_mode: bool
) -> Any:
    if async_mode:
        return await client.chat(**kwargs)

    def _open() -> Any:
        return client.chat(**kwargs)

    return await asyncio.get_event_loop().run_in_executor(None, _open)


async def _process_ollama_chunks(
    chunk_iter: AsyncGenerator[Any, None],
) -> AsyncGenerator[StreamEvent, None]:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_acc: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None

    async for chunk in chunk_iter:
        msg = getattr(chunk, "message", None)
        if msg is not None:
            piece = getattr(msg, "content", None)
            if piece:
                content_parts.append(piece)
                yield StreamEvent.token_event(piece)
            think_piece = getattr(msg, "thinking", None)
            if think_piece:
                thinking_parts.append(think_piece)
                yield StreamEvent.thinking_event(think_piece)
            _merge_tool_calls_incremental(tool_acc, getattr(msg, "tool_calls", None))

        pe = getattr(chunk, "prompt_eval_count", None)
        ev = getattr(chunk, "eval_count", None)
        if getattr(chunk, "done", False) and (pe is not None or ev is not None):
            pt = int(pe or 0)
            ct = int(ev or 0)
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
                "reasoning_tokens": 0,
            }

    full_content = "".join(content_parts) or None
    metadata: dict[str, Any] = {}
    if thinking_parts:
        metadata["thinking"] = "".join(thinking_parts)
    if tool_acc:
        metadata["tool_calls"] = _tool_calls_for_metadata(tool_acc)
    if usage:
        metadata["usage"] = usage

    yield StreamEvent.complete_event(full_content or "", metadata=metadata or None)


async def stream_ollama_chat(
    client: Any,
    *,
    async_mode: bool,
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
    think: bool | ThinkLevel | None,
    keep_alive: float | str | None,
    seed: int | None,
) -> AsyncGenerator[StreamEvent, None]:
    """Stream chat as :class:`StreamEvent` values."""
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
        format_payload=None,
        options=options,
        think=think,
        keep_alive=keep_alive,
        stream=True,
        tool_choice=tool_choice,
    )

    try:
        stream = await _open_chat_stream(client, kwargs, async_mode=async_mode)
        if async_mode:
            chunk_iter = _async_chunk_iter(stream)
        else:
            chunk_iter = _sync_iter_to_async(lambda: stream)

        async for event in _process_ollama_chunks(chunk_iter):
            yield event

    except ResponseError as exc:
        logger.error("Ollama stream error: %s", exc, exc_info=True)
        raise map_ollama_response_error(exc) from exc
    except LLMError as exc:
        logger.error("Ollama stream error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))
    except Exception as exc:
        logger.error("Ollama stream error: %s", exc, exc_info=True)
        yield StreamEvent.error_event(str(exc))


async def _async_chunk_iter(stream: Any) -> AsyncGenerator[Any, None]:
    async for chunk in stream:
        yield chunk
