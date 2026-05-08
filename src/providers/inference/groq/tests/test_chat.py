"""Tests for chat completion normalisation and API wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq_groq._shared.response_models import GroqLLMResponse
from nucleusiq_groq.nb_groq.chat import create_chat_completion, normalize_chat_response


def test_normalize_chat_response_minimal() -> None:
    raw = SimpleNamespace(
        choices=[],
        usage=None,
        model="llama-x",
        id="resp-1",
    )
    out = normalize_chat_response(raw)
    assert isinstance(out, GroqLLMResponse)
    assert out.model == "llama-x"
    assert out.response_id == "resp-1"
    assert out.choices == []


def test_normalize_chat_response_with_message_and_usage() -> None:
    usage = SimpleNamespace(
        prompt_tokens=3,
        completion_tokens=5,
        total_tokens=8,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    fn = SimpleNamespace(name="foo", arguments='{"a":1}')
    tc = SimpleNamespace(id="tc1", type="function", function=fn)
    msg = SimpleNamespace(content="hello", tool_calls=[tc])
    ch = SimpleNamespace(message=msg)
    raw = SimpleNamespace(choices=[ch], usage=usage, model="m", id="r2")
    out = normalize_chat_response(raw)
    assert len(out.choices) == 1
    m0 = out.choices[0].message
    assert m0.content == "hello"
    assert m0.tool_calls is not None
    assert m0.tool_calls[0].function.name == "foo"
    assert out.usage is not None
    assert out.usage.total_tokens == 8
    assert out.usage.reasoning_tokens == 2


@pytest.mark.asyncio
async def test_create_chat_completion_async(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def fake_retry(api_call, **kw):
        return await api_call()

    monkeypatch.setattr("nucleusiq_groq.nb_groq.chat.call_with_retry", fake_retry)

    async def api_create(**payload):
        captured["payload"] = payload
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))
            ],
            usage=None,
            model="m",
            id="id",
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=api_create))
        )
    )

    out = await create_chat_completion(
        client,
        async_mode=True,
        max_retries=2,
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=100,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        parallel_tool_calls=None,
        seed=None,
        user=None,
        extra={},
    )
    assert out.choices[0].message.content == "ok"
    assert "stream" not in captured["payload"]
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_chat_completion_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_retry(api_call, **kw):
        return await api_call()

    monkeypatch.setattr("nucleusiq_groq.nb_groq.chat.call_with_retry", fake_retry)

    def api_create(**payload):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="sync", tool_calls=None)
                )
            ],
            usage=None,
            model="m",
            id="id",
        )

    client = MagicMock()
    client.chat.completions.create = api_create

    out = await create_chat_completion(
        client,
        async_mode=False,
        max_retries=0,
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=50,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        parallel_tool_calls=None,
        seed=None,
        user=None,
        extra={},
    )
    assert out.choices[0].message.content == "sync"
