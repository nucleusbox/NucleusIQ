"""Streaming adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nucleusiq.streaming.events import StreamEventType
from nucleusiq_ollama.nb_ollama.stream_adapter import stream_ollama_chat
from ollama import ResponseError


async def _collect(gen):
    return [e async for e in gen]


@pytest.mark.asyncio
async def test_stream_async_tokens_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_open(client, kwargs, *, async_mode):
        assert async_mode is True

        async def chunks():
            msg1 = SimpleNamespace(
                content="hel",
                thinking=None,
                tool_calls=None,
                role="assistant",
            )
            msg2 = SimpleNamespace(
                content="lo",
                thinking=None,
                tool_calls=None,
                role="assistant",
            )
            yield SimpleNamespace(
                message=msg1,
                done=False,
                prompt_eval_count=None,
                eval_count=None,
            )
            yield SimpleNamespace(
                message=msg2,
                done=True,
                prompt_eval_count=1,
                eval_count=2,
            )

        return chunks()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        fake_open,
    )

    client = object()
    events = await _collect(
        stream_ollama_chat(
            client,
            async_mode=True,
            model="m",
            messages=[{"role": "user", "content": "x"}],
            max_output_tokens=64,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            tools=None,
            tool_choice=None,
            think=None,
            keep_alive=None,
            seed=None,
        )
    )
    assert events[0].type == StreamEventType.TOKEN
    assert events[1].type == StreamEventType.TOKEN
    assert events[-1].type == StreamEventType.COMPLETE
    assert events[-1].content == "hello"
    assert events[-1].metadata is not None
    assert events[-1].metadata.get("usage", {}).get("total_tokens") == 3


@pytest.mark.asyncio
async def test_stream_thinking_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fn:
        name = "add"
        arguments = {"a": 1}

    class TC:
        function = Fn()

    async def fake_open(client, kwargs, *, async_mode):
        async def chunks():
            yield SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    thinking="step1",
                    tool_calls=None,
                    role="assistant",
                ),
                done=False,
                prompt_eval_count=None,
                eval_count=None,
            )
            yield SimpleNamespace(
                message=SimpleNamespace(
                    content="done",
                    thinking=None,
                    tool_calls=[TC()],
                    role="assistant",
                ),
                done=True,
                prompt_eval_count=1,
                eval_count=1,
            )

        return chunks()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        fake_open,
    )

    events = await _collect(
        stream_ollama_chat(
            MagicMock(),
            async_mode=True,
            model="m",
            messages=[],
            max_output_tokens=8,
            temperature=None,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            tools=None,
            tool_choice=None,
            think=None,
            keep_alive=None,
            seed=None,
        )
    )
    types = [e.type for e in events]
    assert StreamEventType.THINKING in types
    assert events[-1].metadata is not None
    assert events[-1].metadata.get("tool_calls")

    async def boom(client, kwargs, *, async_mode):
        raise ResponseError("nope", status_code=404)

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        boom,
    )

    with pytest.raises(Exception):
        await _collect(
            stream_ollama_chat(
                MagicMock(),
                async_mode=True,
                model="m",
                messages=[],
                max_output_tokens=8,
                temperature=None,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=None,
                tools=None,
                tool_choice=None,
                think=None,
                keep_alive=None,
                seed=None,
            )
        )


@pytest.mark.asyncio
async def test_stream_generic_error_becomes_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(client, kwargs, *, async_mode):
        raise ValueError("bad stream")

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        boom,
    )

    events = await _collect(
        stream_ollama_chat(
            MagicMock(),
            async_mode=True,
            model="m",
            messages=[],
            max_output_tokens=8,
            temperature=None,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            tools=None,
            tool_choice=None,
            think=None,
            keep_alive=None,
            seed=None,
        )
    )
    assert len(events) == 1
    assert events[0].type == StreamEventType.ERROR


@pytest.mark.asyncio
async def test_stream_sync_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_open(client, kwargs, *, async_mode):
        assert async_mode is False

        def chunks():
            msg = SimpleNamespace(
                content="z",
                thinking=None,
                tool_calls=None,
                role="assistant",
            )
            yield SimpleNamespace(
                message=msg,
                done=True,
                prompt_eval_count=0,
                eval_count=1,
            )

        return chunks()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        fake_open,
    )

    events = await _collect(
        stream_ollama_chat(
            MagicMock(),
            async_mode=False,
            model="m",
            messages=[],
            max_output_tokens=8,
            temperature=None,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            tools=None,
            tool_choice=None,
            think=None,
            keep_alive=None,
            seed=None,
        )
    )
    assert events[0].type == StreamEventType.TOKEN
    assert events[-1].content == "z"
