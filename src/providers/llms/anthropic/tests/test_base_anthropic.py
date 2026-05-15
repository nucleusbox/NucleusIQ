"""``BaseAnthropic`` constructor + call wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from nucleusiq_anthropic.llm_params import AnthropicLLMParams
from nucleusiq_anthropic.nb_anthropic.base import BaseAnthropic


@pytest.mark.asyncio
async def test_missing_key_raises() -> None:

    from nucleusiq.llms.errors import AuthenticationError

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AuthenticationError):
            BaseAnthropic(api_key=None, async_mode=True)


@pytest.mark.asyncio
async def test_call_routes_to_messages_create(monkeypatch) -> None:

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    llm = BaseAnthropic(model_name="claude-test", async_mode=True)

    fake_msg = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hi")],
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="claude-test",
        id="id1",
    )

    llm._client.messages.create = AsyncMock(return_value=fake_msg)

    out = await llm.call(
        model="claude-test",
        messages=[{"role": "user", "content": "yo"}],
        max_output_tokens=100,
    )

    assert out.choices[0].message.content == "Hi"

    llm._client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_extra_headers_from_llm_params(monkeypatch) -> None:

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    params = AnthropicLLMParams(
        extra_headers={"X-Unit": "1"},
        anthropic_beta="beta-test",
    )

    llm = BaseAnthropic(
        llm_params=params,
        async_mode=True,
    )

    captured: dict = {}

    async def _spy(**kw):

        captured.update(kw)

        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="")],
            usage=SimpleNamespace(
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="m",
            id="i",
        )

    llm._client.messages.create = _spy

    await llm.call(model="m", messages=[{"role": "user", "content": "x"}])

    assert captured["extra_headers"]["X-Unit"] == "1"

    assert "anthropic-beta" in captured["extra_headers"]


def test_context_window_known_model(monkeypatch) -> None:

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    llm = BaseAnthropic(model_name="claude-3-5-sonnet-20241022")
    assert llm.get_context_window() == 200_000


def test_context_window_prefix_and_default(monkeypatch) -> None:

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    llm = BaseAnthropic(model_name="claude-sonnet-4-custom")
    assert llm.get_context_window() == 200_000

    unknown = BaseAnthropic(model_name="not-claude")
    assert unknown.get_context_window() == 128_000


@pytest.mark.asyncio
async def test_call_uses_sync_client(monkeypatch) -> None:

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    from unittest.mock import MagicMock

    llm = BaseAnthropic(async_mode=False)

    fake_msg = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="sync")],
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="m",
        id="i",
    )

    llm._client.messages.create = MagicMock(return_value=fake_msg)

    out = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        stop=("a", "b"),
    )

    assert out.choices[0].message.content == "sync"

    llm._client.messages.create.assert_called_once()

    assert llm._client.messages.create.call_args.kwargs["stop_sequences"] == ["a", "b"]


@pytest.mark.asyncio
async def test_call_stream_yields_from_adapter(monkeypatch) -> None:

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    from nucleusiq.streaming.events import StreamEvent

    async def fake_stream(*_a: object, **_kw: object):

        yield StreamEvent.token_event("x")

        yield StreamEvent.complete_event("", metadata={"usage": {}})

    llm = BaseAnthropic(async_mode=True)

    with patch(
        "nucleusiq_anthropic.nb_anthropic.base.stream_messages",
        new=fake_stream,
    ):
        events = [
            ev
            async for ev in llm.call_stream(
                model="m",
                messages=[{"role": "user", "content": "y"}],
            )
        ]

    assert events[0].type == "token"

    assert events[-1].type == "complete"


def test_convert_tool_spec_public_wrapper(monkeypatch) -> None:

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    llm = BaseAnthropic()

    spec = {
        "type": "function",
        "function": {
            "name": "roll",
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    assert llm._convert_tool_spec(spec)["name"] == "roll"
