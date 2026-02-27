"""Additional coverage tests for retry and Responses API streaming."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from nucleusiq_openai._shared.retry import call_with_retry
from nucleusiq_openai.nb_openai import responses_api as resp_mod


class _DummyAPIConnectionError(Exception):
    pass


class _DummyRateLimitError(Exception):
    pass


class _DummyAPIError(Exception):
    pass


@pytest.mark.asyncio
async def test_retry_additional_async_branches(monkeypatch):
    logger = logging.getLogger("retry-extra")
    g = call_with_retry.__globals__
    monkeypatch.setattr(g["openai"], "APIConnectionError", _DummyAPIConnectionError)
    monkeypatch.setattr(g["openai"], "APIError", _DummyAPIError)

    sleeps = []

    async def _sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(g["asyncio"], "sleep", _sleep)

    c1 = {"n": 0}

    async def _conn_once():
        c1["n"] += 1
        if c1["n"] == 1:
            raise _DummyAPIConnectionError("net")
        return "ok"

    assert (
        await call_with_retry(_conn_once, max_retries=2, async_mode=True, logger=logger)
        == "ok"
    )

    c2 = {"n": 0}

    async def _api_once():
        c2["n"] += 1
        if c2["n"] == 1:
            raise _DummyAPIError("server")
        return "ok2"

    assert (
        await call_with_retry(_api_once, max_retries=2, async_mode=True, logger=logger)
        == "ok2"
    )

    c3 = {"n": 0}

    async def _http_once():
        c3["n"] += 1
        if c3["n"] == 1:
            raise httpx.HTTPError("transport")
        return "ok3"

    assert (
        await call_with_retry(_http_once, max_retries=2, async_mode=True, logger=logger)
        == "ok3"
    )
    assert sleeps


@pytest.mark.asyncio
async def test_retry_sync_sleep_and_exhaustion_paths(monkeypatch):
    logger = logging.getLogger("retry-sync-extra")
    g = call_with_retry.__globals__
    monkeypatch.setattr(g["openai"], "RateLimitError", _DummyRateLimitError)
    monkeypatch.setattr(g["openai"], "APIConnectionError", _DummyAPIConnectionError)
    monkeypatch.setattr(g["openai"], "APIError", _DummyAPIError)

    sleeps: list[int] = []
    monkeypatch.setattr(g["time"], "sleep", lambda s: sleeps.append(s))

    # Rate limit sync path should sleep then succeed.
    r = {"n": 0}

    def rate_then_ok():
        r["n"] += 1
        if r["n"] == 1:
            raise _DummyRateLimitError("slow")
        return "ok"

    assert (
        await call_with_retry(
            rate_then_ok,
            max_retries=2,
            async_mode=False,
            logger=logger,
        )
        == "ok"
    )

    # Connection error exhaustion path.
    def conn_always_fail():
        raise _DummyAPIConnectionError("network down")

    with pytest.raises(_DummyAPIConnectionError):
        await call_with_retry(
            conn_always_fail,
            max_retries=0,
            async_mode=False,
            logger=logger,
        )

    # APIError sync branch sleeps on retry.
    a = {"n": 0}

    def api_then_ok():
        a["n"] += 1
        if a["n"] == 1:
            raise _DummyAPIError("server")
        return "ok-api"

    assert (
        await call_with_retry(
            api_then_ok,
            max_retries=2,
            async_mode=False,
            logger=logger,
        )
        == "ok-api"
    )

    # HTTPError sync branch sleeps on retry.
    h = {"n": 0}

    def http_then_ok():
        h["n"] += 1
        if h["n"] == 1:
            raise httpx.HTTPError("transport")
        return "ok-http"

    assert (
        await call_with_retry(
            http_then_ok,
            max_retries=2,
            async_mode=False,
            logger=logger,
        )
        == "ok-http"
    )

    assert sleeps


@pytest.mark.asyncio
async def test_accumulate_responses_stream_async_function_calls():
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="function_call", call_id="c1", name="add"),
        ),
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"a":'),
        SimpleNamespace(type="response.function_call_arguments.delta", delta=' 1}'),
        SimpleNamespace(type="response.function_call_arguments.done"),
        SimpleNamespace(type="response.output_text.delta", delta="done"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(id="resp_1"),
        ),
    ]

    async def _aiter():
        for e in events:
            yield e

    _final, result, rid = await resp_mod._accumulate_responses_stream_async(_aiter())
    assert rid == "resp_1"
    assert result.choices[0].message.content == "done"
    assert result.choices[0].message.tool_calls is not None
    assert result.choices[0].message.tool_calls[0].function.name == "add"


@pytest.mark.asyncio
async def test_call_responses_api_stream_async_returns_response_id_with_tool_calls(monkeypatch):
    class _Responses:
        async def create(self, **kwargs):
            async def _stream():
                yield SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        type="function_call",
                        call_id="c1",
                        name="add",
                    ),
                )
                yield SimpleNamespace(
                    type="response.function_call_arguments.delta",
                    delta='{"a":1}',
                )
                yield SimpleNamespace(type="response.function_call_arguments.done")
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(id="resp_x"),
                )

            return _stream()

    client = SimpleNamespace(responses=_Responses())

    # Keep retry behavior simple for this test.
    async def _retry(api_call, **kwargs):
        return await api_call()

    monkeypatch.setattr(resp_mod, "call_with_retry", _retry)

    out, rid = await resp_mod.call_responses_api(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "add"}],
        stream=True,
        async_mode=True,
    )
    assert out.choices[0].message.tool_calls is not None
    assert rid == "resp_x"


def test_accumulate_responses_stream_sync_edge_paths():
    events = [
        # delta before output_item.added should be ignored
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"a":'),
        # then function call starts
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="function_call", call_id="c2", name="mul"),
        ),
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"a":2'),
        SimpleNamespace(type="response.function_call_arguments.delta", delta=',"b":3}'),
        SimpleNamespace(type="response.function_call_arguments.done"),
        # completed event without response object keeps rid None
        SimpleNamespace(type="response.completed", response=None),
    ]
    _final, result, rid = resp_mod._accumulate_responses_stream(iter(events))
    assert rid is None
    assert result.choices[0].message.tool_calls is not None
    assert result.choices[0].message.tool_calls[0].function.name == "mul"
    assert result.choices[0].message.tool_calls[0].function.arguments == '{"a":2,"b":3}'


@pytest.mark.asyncio
async def test_call_responses_api_non_stream_async_without_tool_calls(monkeypatch):
    class _Responses:
        async def create(self, **kwargs):
            out = [
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="plain answer")],
                    model_dump=lambda: {"type": "message"},
                )
            ]
            return SimpleNamespace(id="resp_final", output=out)

    client = SimpleNamespace(responses=_Responses())

    async def _retry(api_call, **kwargs):
        return await api_call()

    monkeypatch.setattr(resp_mod, "call_with_retry", _retry)

    out, rid = await resp_mod.call_responses_api(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        async_mode=True,
    )
    assert out.choices[0].message.content == "plain answer"
    assert rid is None


@pytest.mark.asyncio
async def test_call_responses_api_non_stream_sync_with_tool_calls(monkeypatch):
    class _Responses:
        def create(self, **kwargs):
            out = [
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="add",
                    arguments='{"a": 1, "b": 2}',
                    model_dump=lambda: {"type": "function_call"},
                )
            ]
            return SimpleNamespace(id="resp_tool", output=out)

    client = SimpleNamespace(responses=_Responses())

    async def _retry(api_call, **kwargs):
        return api_call()

    monkeypatch.setattr(resp_mod, "call_with_retry", _retry)

    out, rid = await resp_mod.call_responses_api(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "add"}],
        stream=False,
        async_mode=False,
    )
    assert out.choices[0].message.tool_calls is not None
    assert rid == "resp_tool"

