"""Targeted tests for branches that were below the coverage bar (95%+ goal)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nucleusiq.llms.errors import (
    InvalidRequestError,
    ProviderConnectionError,
    ProviderError,
)
from nucleusiq.streaming.events import StreamEventType
from nucleusiq_ollama._shared.retry import call_with_retry
from nucleusiq_ollama._shared.wire import (
    build_options,
    sanitize_messages,
    tool_arguments_to_json_string,
)
from nucleusiq_ollama.nb_ollama.base import BaseOllama
from nucleusiq_ollama.nb_ollama.chat import normalize_chat_response
from nucleusiq_ollama.nb_ollama.stream_adapter import (
    _merge_tool_calls_incremental,
    _open_chat_stream,
    _sync_iter_to_async,
    _tool_calls_for_metadata,
    stream_ollama_chat,
)
from nucleusiq_ollama.structured_output.builder import (
    _type_to_schema,
    build_ollama_format,
)
from nucleusiq_ollama.structured_output.cleaner import clean_schema_for_ollama
from nucleusiq_ollama.structured_output.parser import parse_response
from ollama import ResponseError

# --- retry -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_response_error_uses_time_sleep_when_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    def capture_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("nucleusiq_ollama._shared.retry.time.sleep", capture_sleep)
    logger = MagicMock()

    n = {"c": 0}

    def flaky():
        n["c"] += 1
        if n["c"] < 2:
            raise ResponseError("x", status_code=503)
        return "done"

    out = await call_with_retry(flaky, max_retries=3, async_mode=False, logger=logger)
    assert out == "done"
    assert sleeps


@pytest.mark.asyncio
async def test_retry_connection_error_exhausted_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_sleep(_):
        return None

    monkeypatch.setattr(
        "nucleusiq_ollama._shared.retry.asyncio.sleep",
        immediate_sleep,
    )
    logger = MagicMock()

    async def always_down():
        raise ConnectionError("no")

    with pytest.raises(ProviderConnectionError):
        await call_with_retry(
            always_down, max_retries=2, async_mode=True, logger=logger
        )


@pytest.mark.asyncio
async def test_retry_connection_sync_time_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    monkeypatch.setattr(
        "nucleusiq_ollama._shared.retry.time.sleep",
        lambda s: sleeps.append(s),
    )
    logger = MagicMock()
    n = {"c": 0}

    def flaky():
        n["c"] += 1
        if n["c"] < 2:
            raise ConnectionError("down")
        return 1

    out = await call_with_retry(flaky, max_retries=3, async_mode=False, logger=logger)
    assert out == 1
    assert sleeps


@pytest.mark.asyncio
async def test_retry_httpx_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_sleep(_):
        return None

    monkeypatch.setattr(
        "nucleusiq_ollama._shared.retry.asyncio.sleep",
        immediate_sleep,
    )
    logger = MagicMock()

    async def bad():
        raise httpx.ReadTimeout("t")

    with pytest.raises(ProviderConnectionError):
        await call_with_retry(bad, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_retry_httpx_sync_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nucleusiq_ollama._shared.retry.time.sleep",
        lambda s: None,
    )
    logger = MagicMock()
    n = {"c": 0}

    def flaky():
        n["c"] += 1
        if n["c"] < 2:
            raise httpx.ConnectError("e")
        return 9

    out = await call_with_retry(flaky, max_retries=3, async_mode=False, logger=logger)
    assert out == 9


@pytest.mark.asyncio
async def test_retry_unexpected_exception_wraps_provider_error() -> None:
    logger = MagicMock()

    async def boom():
        raise RuntimeError("nope")

    with pytest.raises(ProviderError):
        await call_with_retry(boom, max_retries=0, async_mode=True, logger=logger)


# --- wire ------------------------------------------------------------------


def test_normalize_tool_call_nested_with_id() -> None:
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "n1",
                    "type": "function",
                    "function": {"name": "g", "arguments": {"q": 1}},
                }
            ],
        }
    ]
    out = sanitize_messages(msgs)
    tc = out[0]["tool_calls"][0]
    assert tc["id"] == "n1"
    assert tc["function"]["name"] == "g"


def test_normalize_tool_flat_with_id() -> None:
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c2", "name": "h", "arguments": '{"z":true}'},
            ],
        }
    ]
    out = sanitize_messages(msgs)
    tc = out[0]["tool_calls"][0]
    assert tc.get("id") == "c2"
    assert tc["function"]["name"] == "h"


def test_build_options_presence_only_and_multi_stop() -> None:
    o = build_options(
        max_output_tokens=10,
        temperature=None,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.2,
        stop=[",", "."],
        seed=None,
    )
    assert "presence_penalty" in o
    assert o["stop"] == [",", "."]


def test_tool_arguments_non_serializable_fallback() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird"

    s = tool_arguments_to_json_string(Weird())
    assert "weird" in s


# --- chat normalize: skip tool call when function missing -------------------


def test_normalize_skips_tool_call_without_function() -> None:
    class TC:
        id = "x"

    msg = SimpleNamespace(content="", tool_calls=[TC()], thinking=None)
    raw = SimpleNamespace(
        message=msg, model="m", created_at="t", prompt_eval_count=None, eval_count=None
    )
    out = normalize_chat_response(raw)
    assert out.choices[0].message.tool_calls == []


# --- stream_adapter ---------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_iter_to_async_worker_exception() -> None:
    def factory():
        raise ValueError("iter boom")
        yield 1

    with pytest.raises(ValueError, match="iter boom"):
        async for _ in _sync_iter_to_async(factory):
            pass


@pytest.mark.asyncio
async def test_open_chat_stream_sync_uses_executor() -> None:
    def sync_chat(**kwargs):
        return ("chunk",)

    client = SimpleNamespace(chat=sync_chat)
    stream = await _open_chat_stream(client, {"model": "m"}, async_mode=False)
    assert stream == ("chunk",)


def test_merge_tool_calls_skips_missing_function() -> None:
    acc: list = []

    class TC:
        pass

    _merge_tool_calls_incremental(acc, [TC()])
    assert acc == []


def test_normalize_tool_implicit_type_with_function_object() -> None:
    """Branch where ``type`` is not ``function`` but ``function`` dict has ``name``."""
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "ix",
                    "function": {"name": "api_fn", "arguments": "{}"},
                }
            ],
        }
    ]
    out = sanitize_messages(msgs)
    tc = out[0]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "api_fn"
    assert tc["id"] == "ix"


@pytest.mark.asyncio
async def test_open_chat_stream_async_await_client() -> None:
    client = MagicMock()
    client.chat = AsyncMock(return_value="stream-result")

    out = await _open_chat_stream(
        client, {"model": "m", "stream": True}, async_mode=True
    )
    assert out == "stream-result"
    client.chat.assert_awaited_once()


def test_merge_tool_calls_resets_non_dict_then_merges() -> None:
    class Fn1:
        name = "g"
        arguments = {"a": 1}

    class TC1:
        function = Fn1()

    acc: list = []
    _merge_tool_calls_incremental(acc, [TC1()])

    class Fn2:
        name = ""
        arguments = {"b": 2}

    class TC2:
        function = Fn2()

    acc[0]["function"]["arguments"] = "not-a-dict"
    _merge_tool_calls_incremental(acc, [TC2()])
    assert acc[0]["function"]["arguments"] == {"b": 2}


@pytest.mark.asyncio
async def test_base_merge_llm_params_into_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def passthrough(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        passthrough,
    )

    async def chat(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None, thinking=None),
            model="m",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace(chat=AsyncMock(side_effect=chat))),
    )

    from nucleusiq_ollama.llm_params import OllamaLLMParams

    llm = BaseOllama(
        async_mode=True,
        max_retries=0,
        llm_params=OllamaLLMParams(think=True, keep_alive="1m"),
    )
    await llm.call(model="x", messages=[{"role": "user", "content": "h"}])
    assert captured.get("think") is True
    assert captured.get("keep_alive") == "1m"


def test_inline_refs_processes_list() -> None:
    from nucleusiq_ollama.structured_output.cleaner import _inline_refs

    out = _inline_refs(
        [{"k": [{"$ref": "#/$defs/T"}]}],
        {"T": {"type": "string"}},
    )
    assert isinstance(out, list)
    assert out[0]["k"][0]["type"] == "string"


def test_merge_tool_calls_string_arguments_not_mapping() -> None:
    class Fn:
        name = "f"
        arguments = "not-a-mapping"

    class TC:
        function = Fn()

    acc: list = []
    _merge_tool_calls_incremental(acc, [TC()])
    assert acc[0]["function"]["arguments"] == {}


def test_tool_calls_for_metadata_non_mapping_args_unchanged() -> None:
    acc = [
        {
            "function": {
                "name": "f",
                "arguments": "already-string",
            }
        }
    ]
    out = _tool_calls_for_metadata(acc)
    assert out[0]["function"]["arguments"] == "already-string"


@pytest.mark.asyncio
async def test_stream_yields_llm_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_open(client, kwargs, *, async_mode):
        async def gen():
            raise InvalidRequestError.from_provider_error(
                provider="ollama",
                message="bad",
                status_code=400,
                original_error=None,
            )
            yield  # becomes async generator; never reached

        return gen()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        fake_open,
    )

    events = [
        e
        async for e in stream_ollama_chat(
            MagicMock(),
            async_mode=True,
            model="m",
            messages=[],
            max_output_tokens=8,
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
    ]  # noqa: E501
    assert len(events) == 1
    assert events[0].type == StreamEventType.ERROR


# --- base ------------------------------------------------------------------


def test_base_uses_sync_client(monkeypatch: pytest.MonkeyPatch) -> None:
    ctor = MagicMock(return_value=SimpleNamespace())
    monkeypatch.setattr("nucleusiq_ollama.nb_ollama.base.ollama.Client", ctor)
    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(),
    )
    llm = BaseOllama(async_mode=False)
    assert llm._client is ctor.return_value
    ctor.assert_called_once()


def test_get_context_and_convert_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace()),
    )
    llm = BaseOllama(async_mode=True)
    assert llm.get_context_window() == 8192
    conv = llm._convert_tool_spec({"name": "z", "description": "", "parameters": {}})
    assert conv["function"]["name"] == "z"


@pytest.mark.asyncio
async def test_call_and_stream_log_extra_kwargs(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passthrough(api_call, **kw):
        return await api_call()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.chat.call_with_retry",
        passthrough,
    )

    async def chat(**kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(content="x", tool_calls=None, thinking=None),
            model="m",
            created_at="t",
            prompt_eval_count=None,
            eval_count=None,
        )

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.base.ollama.AsyncClient",
        MagicMock(return_value=SimpleNamespace(chat=AsyncMock(side_effect=chat))),
    )

    llm = BaseOllama(async_mode=True, max_retries=0)

    with caplog.at_level(logging.DEBUG):
        await llm.call(
            model="m",
            messages=[{"role": "user", "content": "h"}],
            unsupported_extra_field=123,
        )
    assert any("unsupported" in r.message.lower() for r in caplog.records) or any(
        "Ignoring unsupported" in r.message for r in caplog.records
    )

    async def fake_open(*args, **kwargs):
        async def gen():
            if False:
                yield None

        return gen()

    monkeypatch.setattr(
        "nucleusiq_ollama.nb_ollama.stream_adapter._open_chat_stream",
        fake_open,
    )

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        async for _ in llm.call_stream(
            model="m",
            messages=[{"role": "user", "content": "h"}],
            another_extra=456,
        ):
            pass
    assert any("Ignoring unsupported" in r.message for r in caplog.records)


# --- structured output ------------------------------------------------------


def test_build_json_schema_wrapper_missing_inner_schema_dict() -> None:
    wrapped = {"type": "json_schema", "json_schema": "broken"}
    fmt = build_ollama_format(wrapped)
    assert isinstance(fmt, dict)


def test_type_to_schema_list_branch() -> None:
    from typing import List

    assert _type_to_schema(List[int])["type"] == "array"


def test_cleaner_anyof_and_array_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "flex": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ]
            },
            "arr": {"type": "array", "items": {"type": "number"}},
        },
    }
    out = clean_schema_for_ollama(schema)
    assert "anyOf" in out["properties"]["flex"]
    assert out["properties"]["arr"]["items"]["type"] == "number"


def test_inline_refs_unknown_ref_returns_dict_copy() -> None:
    schema = {
        "type": "object",
        "properties": {"r": {"$ref": "#/unknown/X"}},
        "$defs": {},
    }
    out = clean_schema_for_ollama(schema)
    assert "$ref" in out["properties"]["r"]


def test_parse_response_fallback_return_raw() -> None:
    """Schema type that is not BaseModel / dataclass / dict / annotated proto."""
    out = parse_response({"content": '{"k":1}', "role": "assistant"}, 12345)  # type: ignore[arg-type]
    assert out == {"k": 1}


@pytest.mark.asyncio
async def test_call_tuple_format_non_str_dict_raw_uses_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import BaseModel

    class M(BaseModel):
        a: int

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
                content='{"a": 1}',
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

    llm = BaseOllama(async_mode=True, max_retries=0)
    out = await llm.call(
        model="m",
        messages=[{"role": "user", "content": "?"}],
        response_format=(M, M),
    )
    assert isinstance(out, M)
    assert out.a == 1
