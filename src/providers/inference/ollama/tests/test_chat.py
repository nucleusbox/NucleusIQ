"""Chat normalisation and create_ollama_chat."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nucleusiq_ollama._shared.response_models import OllamaLLMResponse
from nucleusiq_ollama.nb_ollama.chat import create_ollama_chat, normalize_chat_response


def test_normalize_minimal() -> None:
    raw = SimpleNamespace(message=None, model="m", created_at="t0")
    out = normalize_chat_response(raw)
    assert isinstance(out, OllamaLLMResponse)
    assert out.choices == []
    assert out.model == "m"


def test_normalize_full() -> None:
    fn = SimpleNamespace(name="foo", arguments={"a": 1})
    tc = SimpleNamespace(id="0", function=fn)
    msg = SimpleNamespace(
        content="hi",
        thinking="...",
        tool_calls=[tc],
    )
    raw = SimpleNamespace(
        message=msg,
        model="llama",
        created_at="t1",
        prompt_eval_count=3,
        eval_count=5,
    )
    out = normalize_chat_response(raw)
    assert len(out.choices) == 1
    m0 = out.choices[0].message
    assert m0.content == "hi"
    assert m0.thinking == "..."
    assert m0.tool_calls is not None
    assert m0.tool_calls[0].function.name == "foo"
    assert m0.tool_calls[0].function.arguments == '{"a": 1}'
    assert out.usage is not None
    assert out.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_create_ollama_chat_async(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_retry(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        fake_retry,
    )

    async def chat(**kwargs):
        assert kwargs["stream"] is False
        assert kwargs["model"] == "m1"
        return SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None, thinking=None),
            model="m1",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    client = SimpleNamespace(chat=AsyncMock(side_effect=chat))
    out = await create_ollama_chat(
        client,
        async_mode=True,
        max_retries=1,
        model="m1",
        messages=[{"role": "user", "content": "x"}],
        max_output_tokens=128,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        tools=None,
        tool_choice=None,
        format_payload=None,
        think=None,
        keep_alive=None,
        seed=None,
    )
    assert out.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_create_ollama_chat_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_retry(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        fake_retry,
    )

    def chat(**kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(content="sync", tool_calls=None, thinking=None),
            model="m1",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    client = SimpleNamespace(chat=chat)
    out = await create_ollama_chat(
        client,
        async_mode=False,
        max_retries=1,
        model="m1",
        messages=[{"role": "user", "content": "x"}],
        max_output_tokens=128,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        tools=None,
        tool_choice=None,
        format_payload=None,
        think=None,
        keep_alive=None,
        seed=None,
    )
    assert out.choices[0].message.content == "sync"
