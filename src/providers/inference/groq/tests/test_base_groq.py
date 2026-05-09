"""BaseGroq integration tests (mocked chat layer)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from nucleusiq_groq._shared.response_models import (
    AssistantMessage,
    GroqLLMResponse,
    _Choice,
)
from nucleusiq_groq.llm_params import GroqLLMParams
from nucleusiq_groq.nb_groq.base import BaseGroq
from pydantic import BaseModel


class _Out(BaseModel):
    v: str


@pytest.fixture
def groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def test_base_groq_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from nucleusiq.llms.errors import AuthenticationError

    with pytest.raises(AuthenticationError):
        BaseGroq(api_key=None)


def test_get_context_window_known_model(groq_key: None) -> None:
    llm = BaseGroq(model_name="llama-3.3-70b-versatile")
    assert llm.get_context_window() == 131_072


def test_get_context_window_default(groq_key: None) -> None:
    llm = BaseGroq(model_name="unknown-model-xyz")
    assert llm.get_context_window() == 128_000


def test_convert_tool_spec(groq_key: None) -> None:
    llm = BaseGroq()
    spec = {"name": "f", "description": "d", "parameters": {"type": "object"}}
    conv = llm._convert_tool_spec(spec)
    assert conv["type"] == "function"
    assert conv["function"]["name"] == "f"


def test_convert_tool_specs_with_get_spec(groq_key: None) -> None:
    llm = BaseGroq()

    class T:
        def get_spec(self):
            return {"name": "t", "description": "", "parameters": {"type": "object"}}

    out = llm.convert_tool_specs([T()])
    assert out[0]["function"]["name"] == "t"


@pytest.mark.asyncio
async def test_call_returns_structured_parsed(groq_key: None) -> None:
    llm = BaseGroq(async_mode=True)

    async def fake_create(*a, **kw):
        resp = GroqLLMResponse(
            choices=[
                _Choice(
                    message=AssistantMessage(
                        content='{"v": "ok"}',
                        tool_calls=None,
                    )
                )
            ],
            model="m",
            response_id="r",
        )
        return resp

    with patch("nucleusiq_groq.nb_groq.base.create_chat_completion", new=fake_create):
        result = await llm.call(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Out,
        )
    assert isinstance(result, _Out)
    assert result.v == "ok"


@pytest.mark.asyncio
async def test_call_returns_raw_when_tools_and_format_skipped(
    groq_key: None, caplog
) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    llm = BaseGroq(async_mode=True)
    resp = GroqLLMResponse(
        choices=[_Choice(message=AssistantMessage(content="x", tool_calls=None))],
        model="m",
        response_id="r",
    )

    async def fake_create(*a, **kw):
        return resp

    with patch("nucleusiq_groq.nb_groq.base.create_chat_completion", new=fake_create):
        out = await llm.call(
            model="m",
            messages=[],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            response_format=_Out,
        )
    assert out is resp
    assert "dropping response_format" in caplog.text


@pytest.mark.asyncio
async def test_call_tuple_response_format(groq_key: None) -> None:
    llm = BaseGroq(async_mode=True)
    fmt = {
        "type": "json_schema",
        "json_schema": {"name": "n", "schema": {}, "strict": True},
    }

    async def fake_create(*a, **kw):
        return GroqLLMResponse(
            choices=[
                _Choice(message=AssistantMessage(content='{"v":"t"}', tool_calls=None))
            ],
            model="m",
            response_id="r",
        )

    with patch("nucleusiq_groq.nb_groq.base.create_chat_completion", new=fake_create):
        result = await llm.call(
            model="m",
            messages=[],
            response_format=(fmt, _Out),
        )
    assert isinstance(result, _Out)


@pytest.mark.asyncio
async def test_call_stream_yields_events(groq_key: None) -> None:
    from nucleusiq.streaming.events import StreamEvent, StreamEventType

    async def fake_stream(*a, **kw):
        yield StreamEvent.token_event("x")
        yield StreamEvent.complete_event("x", metadata={"usage": {"total_tokens": 1}})

    llm = BaseGroq(async_mode=True)
    with patch(
        "nucleusiq_groq.nb_groq.base.stream_chat_completions",
        new=fake_stream,
    ):
        evts = [e async for e in llm.call_stream(model="m", messages=[])]
    assert len(evts) == 2
    assert evts[0].type == StreamEventType.TOKEN


@pytest.mark.asyncio
async def test_temperature_from_llm_params_when_call_passes_none(
    groq_key: None,
) -> None:
    llm = BaseGroq(llm_params=GroqLLMParams(temperature=0.33))
    captured: dict = {}

    async def fake_create(_client, *, temperature, **kw):
        captured["temperature"] = temperature
        return GroqLLMResponse(
            choices=[_Choice(message=AssistantMessage(content="z", tool_calls=None))],
            model="m",
            response_id="r",
        )

    with patch(
        "nucleusiq_groq.nb_groq.base.create_chat_completion",
        new=fake_create,
    ):
        await llm.call(model="m", messages=[], temperature=None)

    assert captured["temperature"] == 0.33


@pytest.mark.asyncio
async def test_call_strict_parallel_unknown_model_raises(
    groq_key: None,
) -> None:
    from nucleusiq.llms.errors import InvalidRequestError

    llm = BaseGroq(
        llm_params=GroqLLMParams(strict_model_capabilities=True),
        async_mode=True,
    )

    async def boom(*a, **kw):
        raise AssertionError("API should not be called when validation fails")

    with patch(
        "nucleusiq_groq.nb_groq.base.create_chat_completion",
        new=boom,
    ):
        with pytest.raises(InvalidRequestError):
            await llm.call(
                model="unknown-parallel-model",
                messages=[],
                parallel_tool_calls=True,
            )


@pytest.mark.asyncio
async def test_call_warns_parallel_unknown_model_when_not_strict(
    groq_key: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    llm = BaseGroq(async_mode=True)

    async def fake_create(*a, **kw):
        return GroqLLMResponse(
            choices=[_Choice(message=AssistantMessage(content="z", tool_calls=None))],
            model="m",
            response_id="r",
        )

    with patch(
        "nucleusiq_groq.nb_groq.base.create_chat_completion",
        new=fake_create,
    ):
        await llm.call(
            model="unknown-parallel-model",
            messages=[],
            parallel_tool_calls=True,
        )
    assert "capability allowlist" in caplog.text


def test_sync_client_initialization(groq_key: None) -> None:
    llm = BaseGroq(async_mode=False)
    assert llm.async_mode is False
