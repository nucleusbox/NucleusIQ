"""Tests for streaming chat completions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq.streaming.events import StreamEventType
from nucleusiq_groq.nb_groq.stream_adapter import stream_chat_completions


def _chunk(
    *,
    content: str | None = None,
    tool_deltas: list | None = None,
    usage=None,
):
    delta = SimpleNamespace(content=content, tool_calls=tool_deltas)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.asyncio
async def test_stream_async_mode_tokens_and_complete() -> None:
    u = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
    )

    async def fake_stream():
        yield _chunk(content="hel")
        yield _chunk(content="lo")
        yield _chunk(content=None, usage=u)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=fake_stream())

    events: list = []
    async for ev in stream_chat_completions(
        client, {"model": "m", "messages": []}, async_mode=True
    ):
        events.append(ev)

    assert len(events) == 3
    assert events[0].type == StreamEventType.TOKEN
    assert events[0].token == "hel"
    assert events[1].type == StreamEventType.TOKEN
    assert events[2].type == StreamEventType.COMPLETE
    assert events[2].content == "hello"
    assert events[2].metadata is not None
    assert events[2].metadata["usage"]["total_tokens"] == 3

    call_kw = client.chat.completions.create.await_args.kwargs
    assert call_kw["stream"] is True
    assert call_kw["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_merges_tool_call_deltas() -> None:
    fn1 = SimpleNamespace(name="get_", arguments=None)
    tc1 = SimpleNamespace(index=0, id="t1", type="function", function=fn1)
    fn2 = SimpleNamespace(name=None, arguments='{"x":')
    tc2 = SimpleNamespace(index=0, function=fn2)
    fn3 = SimpleNamespace(name=None, arguments="1}")
    tc3 = SimpleNamespace(index=0, function=fn3)

    async def fake_stream():
        yield _chunk(content=None, tool_deltas=[tc1])
        yield _chunk(content=None, tool_deltas=[tc2])
        yield _chunk(content=None, tool_deltas=[tc3])

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=fake_stream())

    events: list = []
    async for ev in stream_chat_completions(client, {"model": "m"}, async_mode=True):
        events.append(ev)

    last = events[-1]
    assert last.type == StreamEventType.COMPLETE
    assert last.metadata["tool_calls"][0]["function"]["name"] == "get_"
    assert last.metadata["tool_calls"][0]["function"]["arguments"] == '{"x":1}'


@pytest.mark.asyncio
async def test_stream_sync_mode_via_executor() -> None:
    def fake_stream():
        yield _chunk(content="a")

    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=fake_stream())

    events: list = []
    async for ev in stream_chat_completions(client, {"model": "m"}, async_mode=False):
        events.append(ev)

    assert events[0].token == "a"
    assert events[-1].type == StreamEventType.COMPLETE


@pytest.mark.asyncio
async def test_stream_error_yields_error_event() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    events: list = []
    async for ev in stream_chat_completions(client, {"model": "m"}, async_mode=True):
        events.append(ev)

    assert len(events) == 1
    assert events[0].type == StreamEventType.ERROR
    assert "boom" in (events[0].message or "")
