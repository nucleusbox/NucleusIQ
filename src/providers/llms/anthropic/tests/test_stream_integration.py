"""Streaming pathway coverage (adapter + sync bridge)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.llms.errors import ProviderError
from nucleusiq.streaming.events import StreamEvent
from nucleusiq_anthropic.nb_anthropic.stream_adapter import (
    _sync_iter_to_async,
    stream_messages,
)


@pytest.fixture
def trivial_events() -> list[SimpleNamespace]:

    return [
        SimpleNamespace(type="content_block_delta", index=0, delta=None),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="thinking_delta", thinking=" hmm "),
        ),
        SimpleNamespace(type="message_delta", usage=None),
    ]


@pytest.mark.asyncio
async def test_stream_async_mode_full_path(
    trivial_events: list[SimpleNamespace],
) -> None:

    async def raw_iter():

        for ev in trivial_events:
            yield ev

    async def fake_open(*_a: object, **_kw: object):
        return raw_iter()

    with patch(
        "nucleusiq_anthropic.nb_anthropic.stream_adapter.open_messages_stream",
        new=fake_open,
    ):
        out: list[StreamEvent] = []
        async for ev in stream_messages(
            MagicMock(),
            async_mode=True,
            max_retries=2,
            model="claude-test",
            messages=[{"role": "user", "content": "yo"}],
            max_output_tokens=10,
            temperature=0,
            top_p=None,
            stop=None,
            tools=None,
            tool_choice=None,
            merged_extras={},
            extra_headers=None,
        ):
            out.append(ev)

    kinds = [e.type for e in out]

    assert "thinking" in kinds
    assert out[-1].type == "complete"


@pytest.mark.asyncio
async def test_stream_sync_bridge_with_mock_sync_stream(
    monkeypatch,
    trivial_events: list[SimpleNamespace],
) -> None:
    monkeypatch.setattr(
        "nucleusiq_anthropic.nb_anthropic.stream_create.call_with_retry",
        AsyncMock(return_value=iter(trivial_events)),
    )

    fake_client = SimpleNamespace(messages=MagicMock())

    out: list[StreamEvent] = []
    async for ev in stream_messages(
        fake_client,
        async_mode=False,
        max_retries=1,
        model="claude-test",
        messages=[{"role": "user", "content": "yo"}],
        max_output_tokens=10,
        temperature=None,
        top_p=1.0,
        stop=["x"],
        tools=None,
        tool_choice=None,
        merged_extras={},
        extra_headers=None,
    ):
        out.append(ev)

    assert out[-1].type == "complete"


@pytest.mark.asyncio
async def test_sync_bridge_propagates_iterator_error() -> None:

    class BadIter:
        def __iter__(self):
            yield SimpleNamespace(type="message_delta", usage=None)

            raise ValueError("iter fail")

    with pytest.raises(ValueError):

        async def _drain() -> None:
            async for _ in _sync_iter_to_async(BadIter()):
                pass

        await _drain()


@pytest.mark.asyncio
async def test_stream_open_raises_llm_error_event() -> None:

    async def boom(*_a: object, **_kw: object) -> None:

        raise ProviderError("bad stream", provider="anthropic", status_code=None)

    with patch(
        "nucleusiq_anthropic.nb_anthropic.stream_adapter.open_messages_stream",
        new=boom,
    ):
        out: list[StreamEvent] = []
        async for ev in stream_messages(
            MagicMock(),
            async_mode=True,
            max_retries=1,
            model="m",
            messages=[{"role": "user", "content": "x"}],
            max_output_tokens=2,
            temperature=0,
            top_p=None,
            stop=None,
            tools=None,
            tool_choice=None,
            merged_extras={},
            extra_headers=None,
        ):
            out.append(ev)

    assert out and out[0].type == "error"


@pytest.mark.asyncio
async def test_stream_generic_exception_event() -> None:

    async def boom(*_a: object, **_kw: object):

        raise RuntimeError("generic")

    with patch(
        "nucleusiq_anthropic.nb_anthropic.stream_adapter.open_messages_stream",
        new=boom,
    ):
        out: list[StreamEvent] = []
        async for ev in stream_messages(
            MagicMock(),
            async_mode=True,
            max_retries=1,
            model="m",
            messages=[{"role": "user", "content": "x"}],
            max_output_tokens=2,
            temperature=0,
            top_p=None,
            stop=None,
            tools=None,
            tool_choice=None,
            merged_extras={},
            extra_headers=None,
        ):
            out.append(ev)

    assert out[0].type == "error"


@pytest.mark.asyncio
async def test_process_raw_events_malformed_tool_json() -> None:

    from nucleusiq_anthropic.nb_anthropic.stream_adapter import _process_raw_events

    events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="t", name="n"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json="not json"),
        ),
        SimpleNamespace(type="message_delta", usage=None),
    ]

    async def gen():

        for e in events:
            yield e

    out: list[StreamEvent] = []
    async for ev in _process_raw_events(gen()):
        out.append(ev)

    complete = out[-1]
    assert complete.metadata["tool_calls"][0]["arguments"] == "{}"
