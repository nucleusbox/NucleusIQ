"""BaseOllama wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq_ollama._shared.response_models import OllamaLLMResponse
from nucleusiq_ollama.nb_ollama.base import BaseOllama
from pydantic import BaseModel


class Answer(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_base_call_uses_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passthrough(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        passthrough,
    )

    async def chat(**kwargs):
        assert kwargs["model"] == "qwen"
        return SimpleNamespace(
            message=SimpleNamespace(content="yo", tool_calls=None, thinking=None),
            model="qwen",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace(chat=AsyncMock(side_effect=chat))),
    )

    llm = BaseOllama(model_name="qwen", async_mode=True, max_retries=1)
    out = await llm.call(
        model="qwen",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert isinstance(out, OllamaLLMResponse)
    assert out.choices[0].message.content == "yo"


@pytest.mark.asyncio
async def test_base_call_parses_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passthrough(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        passthrough,
    )

    async def chat(**kwargs):
        assert kwargs.get("format") is not None
        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"value": 42}',
                tool_calls=None,
                thinking=None,
            ),
            model="m",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace(chat=AsyncMock(side_effect=chat))),
    )

    llm = BaseOllama(async_mode=True, max_retries=1)
    out = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "?"}],
        response_format=Answer,
    )
    assert isinstance(out, Answer)
    assert out.value == 42


@pytest.mark.asyncio
async def test_base_call_tools_drop_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passthrough(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        passthrough,
    )

    fmt_set = {"v": False}

    async def chat(**kwargs):
        fmt_set["v"] = kwargs.get("format") is not None
        return SimpleNamespace(
            message=SimpleNamespace(
                content="plain",
                tool_calls=None,
                thinking=None,
            ),
            model="m",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace(chat=AsyncMock(side_effect=chat))),
    )

    tool = {"type": "function", "function": {"name": "t", "parameters": {}}}
    llm = BaseOllama(async_mode=True, max_retries=1)
    out = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "?"}],
        tools=[tool],
        response_format=Answer,
    )
    assert isinstance(out, OllamaLLMResponse)
    assert fmt_set["v"] is False
    assert out.choices[0].message.content == "plain"


@pytest.mark.asyncio
async def test_base_tuple_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passthrough(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        passthrough,
    )

    schema_dict = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    async def chat(**kwargs):
        assert kwargs["format"] == schema_dict
        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"value": 7}',
                tool_calls=None,
                thinking=None,
            ),
            model="m",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace(chat=AsyncMock(side_effect=chat))),
    )

    llm = BaseOllama(async_mode=True, max_retries=1)
    out = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "?"}],
        response_format=(schema_dict, Answer),
    )
    assert isinstance(out, Answer)
    assert out.value == 7


@pytest.mark.asyncio
async def test_base_tuple_openai_json_schema_wrapper_normalizes_for_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent structured output uses OpenAI-shaped provider_format in the tuple."""

    async def passthrough(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        passthrough,
    )

    schema_dict = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    openai_wrapper = {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "strict": True,
            "schema": schema_dict,
        },
    }

    async def chat(**kwargs):
        fmt = kwargs["format"]
        assert isinstance(fmt, dict)
        assert fmt.get("type") != "json_schema"
        assert "value" in (fmt.get("properties") or {})
        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"value": 99}',
                tool_calls=None,
                thinking=None,
            ),
            model="m",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace(chat=AsyncMock(side_effect=chat))),
    )

    llm = BaseOllama(async_mode=True, max_retries=1)
    out = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "?"}],
        response_format=(openai_wrapper, Answer),
    )
    assert isinstance(out, Answer)
    assert out.value == 99


def test_base_client_host_and_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.local:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    ctor = MagicMock(return_value=SimpleNamespace())
    monkeypatch.setattr("nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient", ctor)
    BaseOllama(async_mode=True)
    ctor.assert_called_once()
    _, kwargs = ctor.call_args
    assert kwargs["host"] == "http://ollama.local:11434"
    assert kwargs["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_call_stream_yields_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_open(client, kwargs, *, async_mode):
        async def gen():
            msg = SimpleNamespace(
                content="x",
                thinking=None,
                tool_calls=None,
                role="assistant",
            )
            yield SimpleNamespace(
                message=msg,
                done=True,
                prompt_eval_count=0,
                eval_count=0,
            )

        return gen()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        fake_open,
    )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace()),
    )

    llm = BaseOllama(async_mode=True, max_retries=1)
    evts: list = []
    async for e in llm.call_stream(
        model="m",
        messages=[{"role": "user", "content": "h"}],
    ):
        evts.append(e)
    assert len(evts) == 2
