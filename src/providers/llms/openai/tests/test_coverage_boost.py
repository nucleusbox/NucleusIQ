# ruff: noqa: E402

import dataclasses
import logging
import sys
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nucleusiq_openai._shared.models import (
    ChatCompletionsPayload,
    MessageInputItem,
    ResponsesPayload,
)
from nucleusiq_openai._shared.response_models import (
    AssistantMessage,
    ToolCall,
    ToolCallFunction,
)
from nucleusiq_openai._shared.retry import call_with_retry
from nucleusiq_openai.nb_openai import chat_completions as chat_mod
from nucleusiq_openai.nb_openai import responses_api as resp_mod
from nucleusiq_openai.nb_openai.base import BaseOpenAI
from nucleusiq_openai.structured_output.builder import (
    _annotations_to_schema,
    _dataclass_to_schema,
    _type_to_schema,
    build_response_format,
)
from nucleusiq_openai.structured_output.cleaner import clean_schema_for_openai
from nucleusiq_openai.structured_output.parser import parse_response


class _DummyRateLimitError(Exception):
    pass


class _DummyAPIConnectionError(Exception):
    pass


class _DummyAuthenticationError(Exception):
    pass


class _DummyPermissionDeniedError(Exception):
    pass


class _DummyBadRequestError(Exception):
    pass


class _DummyUnprocessableEntityError(Exception):
    pass


class _DummyAPIError(Exception):
    pass


@pytest.mark.asyncio
async def test_retry_async_and_sync_paths(monkeypatch):
    logger = logging.getLogger("test-retry")

    async def async_call():
        return "ok-async"

    assert (
        await call_with_retry(
            async_call,
            max_retries=1,
            async_mode=True,
            logger=logger,
        )
        == "ok-async"
    )

    def sync_call():
        return "ok-sync"

    assert (
        await call_with_retry(
            sync_call,
            max_retries=1,
            async_mode=False,
            logger=logger,
        )
        == "ok-sync"
    )


@pytest.mark.asyncio
async def test_retry_error_branches(monkeypatch):
    logger = logging.getLogger("test-retry-errors")
    g = call_with_retry.__globals__
    monkeypatch.setattr(g["openai"], "RateLimitError", _DummyRateLimitError)
    monkeypatch.setattr(g["openai"], "APIConnectionError", _DummyAPIConnectionError)
    monkeypatch.setattr(g["openai"], "AuthenticationError", _DummyAuthenticationError)
    monkeypatch.setattr(
        g["openai"], "PermissionDeniedError", _DummyPermissionDeniedError
    )
    monkeypatch.setattr(g["openai"], "BadRequestError", _DummyBadRequestError)
    monkeypatch.setattr(
        g["openai"], "UnprocessableEntityError", _DummyUnprocessableEntityError
    )
    monkeypatch.setattr(g["openai"], "APIError", _DummyAPIError)

    sleep_calls: list[int] = []

    async def _sleep_async(seconds: int):
        sleep_calls.append(seconds)

    monkeypatch.setattr(g["asyncio"], "sleep", _sleep_async)
    monkeypatch.setattr(g["time"], "sleep", lambda s: sleep_calls.append(s))

    attempts = {"n": 0}

    async def rate_limit_then_ok():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _DummyRateLimitError("slow down")
        return "ok"

    assert (
        await call_with_retry(
            rate_limit_then_ok,
            max_retries=2,
            async_mode=True,
            logger=logger,
        )
        == "ok"
    )
    assert sleep_calls

    with pytest.raises(_DummyRateLimitError):
        n = {"v": 0}

        async def always_rate_limit():
            n["v"] += 1
            raise _DummyRateLimitError("still throttled")

        await call_with_retry(
            always_rate_limit,
            max_retries=0,
            async_mode=True,
            logger=logger,
        )

    with pytest.raises(ValueError, match="Invalid API key"):

        async def auth_err():
            raise _DummyAuthenticationError("bad key")

        await call_with_retry(auth_err, max_retries=0, async_mode=True, logger=logger)

    with pytest.raises(ValueError, match="Permission denied"):

        async def perm_err():
            raise _DummyPermissionDeniedError("forbidden")

        await call_with_retry(perm_err, max_retries=0, async_mode=True, logger=logger)

    patched = {"done": False}

    async def bad_request_once():
        if not patched["done"]:
            raise _DummyBadRequestError("invalid tool_choice")
        return "patched-ok"

    def on_bad_request() -> bool:
        patched["done"] = True
        return True

    assert (
        await call_with_retry(
            bad_request_once,
            max_retries=1,
            async_mode=True,
            logger=logger,
            on_bad_request=on_bad_request,
        )
        == "patched-ok"
    )

    with pytest.raises(ValueError, match="Invalid request parameters"):

        async def unprocessable():
            raise _DummyUnprocessableEntityError("nope")

        await call_with_retry(
            unprocessable,
            max_retries=1,
            async_mode=True,
            logger=logger,
        )

    with pytest.raises(_DummyAPIError):

        async def api_error():
            raise _DummyAPIError("api down")

        await call_with_retry(api_error, max_retries=0, async_mode=True, logger=logger)

    with pytest.raises(httpx.HTTPError):

        async def http_error():
            raise httpx.HTTPError("transport")

        await call_with_retry(http_error, max_retries=0, async_mode=True, logger=logger)

    with pytest.raises(RuntimeError):

        async def unknown_error():
            raise RuntimeError("boom")

        await call_with_retry(
            unknown_error,
            max_retries=0,
            async_mode=True,
            logger=logger,
        )

    # Sync-mode retry branches (covers time.sleep paths)
    attempts_sync = {"n": 0}

    def conn_then_ok_sync():
        attempts_sync["n"] += 1
        if attempts_sync["n"] == 1:
            raise _DummyAPIConnectionError("network")
        return "ok-sync"

    assert (
        await call_with_retry(
            conn_then_ok_sync,
            max_retries=2,
            async_mode=False,
            logger=logger,
        )
        == "ok-sync"
    )

    with pytest.raises(_DummyAPIError):

        def api_err_sync():
            raise _DummyAPIError("api-sync")

        await call_with_retry(
            api_err_sync,
            max_retries=0,
            async_mode=False,
            logger=logger,
        )

    with pytest.raises(httpx.HTTPError):

        def http_err_sync():
            raise httpx.HTTPError("http-sync")

        await call_with_retry(
            http_err_sync,
            max_retries=0,
            async_mode=False,
            logger=logger,
        )


class _Person(BaseModel):
    name: str
    age: int


@dataclasses.dataclass
class _Point:
    x: int
    y: int
    label: str = "pt"


class _Typed:
    value: int
    tags: list[str]
    meta: dict[str, Any]
    maybe: int | None


def test_structured_output_build_clean_parse():
    # builder: dict passthrough
    assert build_response_format({"type": "json_object"}) == {"type": "json_object"}

    # builder: pydantic, dataclass, annotations
    pyd_fmt = build_response_format(_Person)
    dc_fmt = build_response_format(_Point)
    td_fmt = build_response_format(_Typed)
    assert pyd_fmt["type"] == "json_schema"
    assert dc_fmt["json_schema"]["name"] == "_Point"
    assert td_fmt["json_schema"]["schema"]["properties"]["meta"]["type"] == "object"
    assert build_response_format(123) is None

    # direct helpers
    dc_schema = _dataclass_to_schema(_Point)
    assert dc_schema["required"] == ["x", "y"]
    ann_schema = _annotations_to_schema(_Typed)
    assert "value" in ann_schema["properties"]
    assert _type_to_schema(typing.Union[int, None]) == {  # noqa: UP007
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    }
    assert _type_to_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}
    assert _type_to_schema(dict[str, int]) == {"type": "object"}
    assert _type_to_schema(complex) == {"type": "string"}

    raw = {
        "title": "T",
        "$schema": "x",
        "description": "desc",
        "$defs": {
            "Address": {
                "type": "object",
                "title": "Address",
                "properties": {"city": {"type": "string", "description": "c"}},
            }
        },
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "address": {"$ref": "#/$defs/Address"},
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"k": {"type": "integer"}}},
            },
            "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }
    cleaned = clean_schema_for_openai(raw)
    assert "title" not in cleaned and "$defs" not in cleaned
    assert cleaned["additionalProperties"] is False
    assert "address" in cleaned["properties"]
    assert cleaned["properties"]["items"]["items"]["additionalProperties"] is False

    parsed_model = parse_response({"content": '{"name":"A","age":1}'}, _Person)
    assert isinstance(parsed_model, _Person) and parsed_model.age == 1
    parsed_dc = parse_response({"content": '{"x":3,"y":4,"label":"p"}'}, _Point)
    assert isinstance(parsed_dc, _Point) and parsed_dc.x == 3
    parsed_dict = parse_response({"content": '{"value":10}'}, {"type": "json_object"})
    assert parsed_dict["value"] == 10
    parsed_ann = parse_response({"content": '{"value":9}'}, _Typed)
    assert parsed_ann["value"] == 9
    parsed_fallback = parse_response({"content": '{"x":1}'}, 123)
    assert parsed_fallback["x"] == 1

    with pytest.raises(ValueError, match="empty content"):
        parse_response({"content": ""}, _Person)
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_response({"content": "{bad json"}, _Person)


def _mk_chunk_delta(content: str | None = None, tool_calls: list | None = None):
    """Build a mock delta matching the real OpenAI SDK ChatCompletionDelta."""
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _mk_chunk_message(content: str = "hi", with_tool: bool = False):
    msg = {"role": "assistant", "content": content}
    if with_tool:
        msg["tool_calls"] = [
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "add", "arguments": "{}"},
            }
        ]
    return SimpleNamespace(model_dump=lambda: msg)


def _mk_tool_call_delta(index=0, call_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, type="function", function=fn)


@pytest.mark.asyncio
async def test_chat_completions_stream_and_sync_paths(monkeypatch):
    # Async stream: multi-chunk content accumulation
    async def _async_stream_create(**kwargs):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=_mk_chunk_delta(content="Hello"))]
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=_mk_chunk_delta(content=" world"))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_async_stream_create))
    )
    result = await chat_mod.call_chat_completions(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        async_mode=True,
    )
    assert result.choices[0].message.content == "Hello world"

    # Sync stream: tool call accumulation across chunks
    def _sync_stream(**kwargs):
        return iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=_mk_chunk_delta(
                                tool_calls=[
                                    _mk_tool_call_delta(0, "call1", "add", '{"a"')
                                ]
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=_mk_chunk_delta(
                                tool_calls=[
                                    _mk_tool_call_delta(0, None, None, ": 1}")
                                ]
                            )
                        )
                    ]
                ),
            ]
        )

    sync_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_sync_stream))
    )
    out = await chat_mod.call_chat_completions(
        sync_client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        async_mode=False,
    )
    assert out.choices[0].message.tool_calls is not None
    assert out.choices[0].message.tool_calls[0].function.name == "add"
    assert out.choices[0].message.tool_calls[0].function.arguments == '{"a": 1}'

    # Sync stream with empty chunks -> returns response with no content
    sync_client2 = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: iter([]))
        )
    )
    out2 = await chat_mod.call_chat_completions(
        sync_client2,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        stream=True,
        async_mode=False,
    )
    assert out2.choices[0].message.content is None
    assert out2.choices[0].message.tool_calls is None


@pytest.mark.asyncio
async def test_chat_bad_request_tool_choice_patch(monkeypatch):
    captured = {}

    async def fake_retry(api_call, **kwargs):
        assert kwargs["on_bad_request"]() is True
        assert kwargs["on_bad_request"]() is False
        return await api_call()

    monkeypatch.setattr(chat_mod, "call_with_retry", fake_retry)

    async def create_async(**kwargs):
        captured.update(kwargs)
        msg = SimpleNamespace(model_dump=lambda: {"role": "assistant", "content": "ok"})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_async))
    )
    await chat_mod.call_chat_completions(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
        tool_choice="required",
        async_mode=True,
    )
    assert "tool_choice" not in captured


@pytest.mark.asyncio
async def test_responses_api_additional_branches(monkeypatch):
    # no client.responses branch
    result, rid = await resp_mod.call_responses_api(
        SimpleNamespace(),
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result is None and rid is None

    # sync branch + bad request patch + extras + strict model temperature skip
    captured = {}

    class SyncResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            out = [
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="done")],
                    model_dump=lambda: {"type": "message"},
                )
            ]
            return SimpleNamespace(id="resp1", output=out)

    async def fake_retry_async(api_call, **kwargs):
        kwargs["on_bad_request"]()
        return api_call()

    monkeypatch.setattr(resp_mod, "call_with_retry", fake_retry_async)
    client = SimpleNamespace(responses=SyncResponses())
    out, new_id = await resp_mod.call_responses_api(
        client,
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function", "function": {"name": "add", "parameters": {}}}],
        tool_choice="required",
        async_mode=False,
        reasoning_effort="low",
        service_tier="default",
        metadata={"k": "v"},
        store=True,
        truncation="auto",
        max_tool_calls=3,
        parallel_tool_calls=True,
        safety_identifier="sid",
        seed=1,
    )
    assert out.choices[0].message.content == "done"
    assert new_id is None
    assert "tool_choice" not in captured
    assert "temperature" not in captured  # strict-defaults model family
    assert captured["reasoning"]["effort"] == "low"
    assert captured["service_tier"] == "default"

    # Hit instructions + stream payload branches and on_bad_request False path
    captured2 = {}

    class SyncResponses2:
        def create(self, **kwargs):
            captured2.update(kwargs)
            return iter(
                [
                    SimpleNamespace(type="response.output_text.delta", delta="ok2"),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(id="resp2"),
                    ),
                ]
            )

    async def fake_retry_no_patch(api_call, **kwargs):
        assert kwargs["on_bad_request"]() is False
        return api_call()

    monkeypatch.setattr(resp_mod, "call_with_retry", fake_retry_no_patch)
    client2 = SimpleNamespace(responses=SyncResponses2())
    out2, _ = await resp_mod.call_responses_api(
        client2,
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Follow instructions"},
            {"role": "user", "content": "x"},
        ],
        stream=True,
        async_mode=False,
    )
    assert out2.choices[0].message.content == "ok2"
    assert captured2["instructions"] == "Follow instructions"
    assert captured2["stream"] is True

    # responses_call_direct sync path
    raw = await resp_mod.responses_call_direct(
        client,
        model="gpt-4o",
        input="hello",
        tools=[{"type": "web_search_preview"}],
        instructions="do",
        temperature=0.1,
        max_output_tokens=20,
        previous_response_id="r0",
        stream=True,
        include=["x"],
        tool_choice="auto",
        async_mode=False,
        custom="y",
    )
    assert raw.id == "resp1"


@pytest.mark.asyncio
async def test_base_branches_and_helpers(monkeypatch):
    # __init__ missing API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        BaseOpenAI(model_name="gpt-4o", api_key=None)

    # sync client construction branch
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setattr(
        "nucleusiq_openai.nb_openai.base.openai.OpenAI",
        lambda **kwargs: SimpleNamespace(sync=True, kwargs=kwargs),
    )
    llm_sync = BaseOpenAI(model_name="gpt-4o", api_key=None, async_mode=False)
    assert getattr(llm_sync._client, "sync", False) is True
    assert llm_sync.estimate_tokens("hello world") > 0

    # helper delegates and non-dict tool skip
    assert isinstance(llm_sync._uses_max_completion_tokens("gpt-4o"), bool)
    assert isinstance(llm_sync._is_strict_defaults_model("gpt-4o"), bool)
    assert llm_sync._has_native_tools(["not-a-dict", {"type": "function"}]) is False

    # structured-output dict-message branch in call()
    async def fake_chat(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message={"role": "assistant", "content": '{"name":"Bob","age":22}'}
                )
            ]
        )

    monkeypatch.setattr(
        "nucleusiq_openai.nb_openai.base.call_chat_completions", fake_chat
    )
    parsed = await llm_sync.call(
        model="gpt-4o",
        messages=[{"role": "user", "content": "extract"}],
        response_format=_Person,
    )
    assert isinstance(parsed, _Person)

    # structured-output tuple branch + extra param collection branch
    parsed2 = await llm_sync.call(
        model="gpt-4o",
        messages=[{"role": "user", "content": "extract"}],
        response_format=({"type": "json_object"}, _Person),
        seed=123,
    )
    assert isinstance(parsed2, _Person)


def test_shared_models_serialization_paths():
    payload = ResponsesPayload(
        model="gpt-4o",
        input=[MessageInputItem(role="user", content="hi")],
        stream=False,
        temperature=None,
    )
    kwargs = payload.to_api_kwargs()
    assert kwargs["model"] == "gpt-4o"
    assert "temperature" not in kwargs

    chat_payload = ChatCompletionsPayload.build(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        seed=1,
    )
    as_kwargs = chat_payload.to_api_kwargs()
    assert as_kwargs["model"] == "gpt-4o"
    assert "max_tokens" in as_kwargs or "max_completion_tokens" in as_kwargs

    msg = AssistantMessage(
        role="assistant",
        content="ok",
        tool_calls=[
            ToolCall(id="1", function=ToolCallFunction(name="add", arguments="{}"))
        ],
        _native_outputs=[{"type": "web_search_call"}],
    )
    d = msg.to_dict()
    assert "tool_calls" in d
    assert "_native_outputs" in d
