"""Tests for :mod:`nucleusiq_anthropic.nb_anthropic.stream_create`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import NOT_GIVEN
from nucleusiq_anthropic.nb_anthropic import stream_create as sc_mod


@pytest.mark.asyncio
async def test_open_messages_stream_async_path(monkeypatch) -> None:
    client = MagicMock()
    client.messages.create = AsyncMock(return_value="stream-obj")

    async def relay(api_call: object, **kw: object) -> object:

        if kw.get("async_mode"):
            return await api_call()

        return api_call()

    monkeypatch.setattr(sc_mod, "call_with_retry", relay)

    out = await sc_mod.open_messages_stream(
        client,
        async_mode=True,
        max_retries=1,
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=32,
        temperature=0.0,
        top_p=1.0,
        stop=None,
        tools=None,
        tool_choice=None,
        merged_extras={},
        extra_headers=None,
    )
    assert out == "stream-obj"
    client.messages.create.assert_awaited_once()
    ck = client.messages.create.await_args.kwargs
    assert ck["temperature"] == 0.0
    assert ck["top_p"] is NOT_GIVEN


@pytest.mark.asyncio
async def test_open_messages_stream_sync_path(monkeypatch) -> None:

    sentinel = iter(())

    client = MagicMock()
    client.messages.create = MagicMock(return_value=sentinel)

    async def relay(api_call: object, **kw: object) -> object:

        if kw.get("async_mode"):
            return await api_call()
        return api_call()

    monkeypatch.setattr(sc_mod, "call_with_retry", relay)

    out = await sc_mod.open_messages_stream(
        client,
        async_mode=False,
        max_retries=2,
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_output_tokens=32,
        temperature=None,
        top_p=None,
        stop=["."],
        tools=None,
        tool_choice=None,
        merged_extras={"thinking": {}},
        extra_headers={"X-T": "1"},
    )

    assert out is sentinel

    client.messages.create.assert_called_once()
