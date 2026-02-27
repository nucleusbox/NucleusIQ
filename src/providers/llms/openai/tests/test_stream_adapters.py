# ruff: noqa: E402
"""Tests for streaming adapters (stream_adapters.py) and BaseOpenAI.call_stream().

Covers:
    1. Chat Completions streaming — text tokens, tool calls, usage telemetry
    2. Responses API streaming — text events, function calls, usage
    3. Sync→async bridge (_sync_iter_to_async)
    4. Error propagation (error events on failure)
    5. BaseOpenAI.call_stream() routing (Chat Completions vs Responses API)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nucleusiq.streaming.events import StreamEvent
from nucleusiq_openai.nb_openai.stream_adapters import (
    _extract_usage_dict,
    _merge_tool_call_delta,
    _process_chat_chunks,
    _process_responses_events,
    _sync_iter_to_async,
    stream_chat_completions,
    stream_responses_api,
)

# ====================================================================== #
# Mock helpers                                                            #
# ====================================================================== #


def _mk_chat_chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: Any | None = None,
) -> SimpleNamespace:
    """Create a mock Chat Completions streaming chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=usage)


def _mk_empty_chunk(usage: Any | None = None) -> SimpleNamespace:
    """Create a chunk with empty choices (usage-only final chunk)."""
    return SimpleNamespace(choices=[], usage=usage)


def _mk_tool_call_delta(
    index: int = 0,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, type="function", function=fn)


def _mk_usage(
    prompt: int = 10, completion: int = 20, total: int = 30, reasoning: int = 0
) -> SimpleNamespace:
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning else None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        completion_tokens_details=details,
    )


def _mk_responses_event(event_type: str, **kwargs: Any) -> SimpleNamespace:
    """Create a mock Responses API streaming event."""
    return SimpleNamespace(type=event_type, **kwargs)


# ====================================================================== #
# _merge_tool_call_delta                                                  #
# ====================================================================== #


class TestMergeToolCallDelta:
    def test_single_tool_call(self):
        acc: dict[int, dict[str, Any]] = {}
        _merge_tool_call_delta(
            acc, [_mk_tool_call_delta(0, "call_1", "search", '{"q":')]
        )
        _merge_tool_call_delta(acc, [_mk_tool_call_delta(0, None, None, '"cats"}')])

        assert len(acc) == 1
        assert acc[0]["id"] == "call_1"
        assert acc[0]["function"]["name"] == "search"
        assert acc[0]["function"]["arguments"] == '{"q":"cats"}'

    def test_multiple_tool_calls(self):
        acc: dict[int, dict[str, Any]] = {}
        _merge_tool_call_delta(acc, [_mk_tool_call_delta(0, "c1", "foo", "a")])
        _merge_tool_call_delta(acc, [_mk_tool_call_delta(1, "c2", "bar", "b")])

        assert len(acc) == 2
        assert acc[0]["function"]["name"] == "foo"
        assert acc[1]["function"]["name"] == "bar"


# ====================================================================== #
# _extract_usage_dict                                                     #
# ====================================================================== #


class TestExtractUsageDict:
    def test_basic_usage(self):
        usage = _mk_usage(10, 20, 30)
        result = _extract_usage_dict(usage)
        assert result["prompt_tokens"] == 10
        assert result["completion_tokens"] == 20
        assert result["total_tokens"] == 30

    def test_with_reasoning_tokens(self):
        usage = _mk_usage(10, 20, 30, reasoning=5)
        result = _extract_usage_dict(usage)
        assert result["reasoning_tokens"] == 5

    def test_none_values_default_zero(self):
        usage = SimpleNamespace(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            completion_tokens_details=None,
        )
        result = _extract_usage_dict(usage)
        assert result["prompt_tokens"] == 0


# ====================================================================== #
# _process_chat_chunks                                                    #
# ====================================================================== #


class TestProcessChatChunks:
    @pytest.mark.asyncio
    async def test_text_only_stream(self):
        async def chunks():
            yield _mk_chat_chunk(content="Hello")
            yield _mk_chat_chunk(content=" world")

        events: list[StreamEvent] = []
        async for e in _process_chat_chunks(chunks()):
            events.append(e)

        tokens = [e for e in events if e.type == "token"]
        complete = [e for e in events if e.type == "complete"]

        assert len(tokens) == 2
        assert tokens[0].token == "Hello"
        assert tokens[1].token == " world"
        assert len(complete) == 1
        assert complete[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_tool_calls_in_metadata(self):
        async def chunks():
            yield _mk_chat_chunk(
                tool_calls=[_mk_tool_call_delta(0, "c1", "search", '{"q":')]
            )
            yield _mk_chat_chunk(
                tool_calls=[_mk_tool_call_delta(0, None, None, '"test"}')]
            )

        events: list[StreamEvent] = []
        async for e in _process_chat_chunks(chunks()):
            events.append(e)

        complete = events[-1]
        assert complete.type == "complete"
        assert complete.metadata is not None
        assert len(complete.metadata["tool_calls"]) == 1
        tc = complete.metadata["tool_calls"][0]
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == '{"q":"test"}'

    @pytest.mark.asyncio
    async def test_usage_in_metadata(self):
        usage = _mk_usage(5, 15, 20)

        async def chunks():
            yield _mk_chat_chunk(content="hi")
            yield _mk_empty_chunk(usage=usage)

        events: list[StreamEvent] = []
        async for e in _process_chat_chunks(chunks()):
            events.append(e)

        complete = events[-1]
        assert complete.metadata is not None
        assert complete.metadata["usage"]["prompt_tokens"] == 5
        assert complete.metadata["usage"]["total_tokens"] == 20

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        async def chunks():
            if False:
                yield None

        events: list[StreamEvent] = []
        async for e in _process_chat_chunks(chunks()):
            events.append(e)

        assert len(events) == 1
        assert events[0].type == "complete"
        assert events[0].content == ""

    @pytest.mark.asyncio
    async def test_chunks_with_no_delta(self):
        """Chunks where delta is None should be skipped."""

        async def chunks():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=None)], usage=None)
            yield _mk_chat_chunk(content="ok")

        events: list[StreamEvent] = []
        async for e in _process_chat_chunks(chunks()):
            events.append(e)

        tokens = [e for e in events if e.type == "token"]
        assert len(tokens) == 1
        assert tokens[0].token == "ok"

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_calls(self):
        async def chunks():
            yield _mk_chat_chunk(content="Thinking...")
            yield _mk_chat_chunk(
                tool_calls=[_mk_tool_call_delta(0, "c1", "calc", '{"x":1}')]
            )

        events: list[StreamEvent] = []
        async for e in _process_chat_chunks(chunks()):
            events.append(e)

        complete = events[-1]
        assert complete.content == "Thinking..."
        assert complete.metadata is not None
        assert len(complete.metadata["tool_calls"]) == 1


# ====================================================================== #
# _process_responses_events                                               #
# ====================================================================== #


class TestProcessResponsesEvents:
    @pytest.mark.asyncio
    async def test_text_delta_events(self):
        async def events():
            yield _mk_responses_event("response.output_text.delta", delta="Hello")
            yield _mk_responses_event("response.output_text.delta", delta=" world")
            yield _mk_responses_event("response.completed", response=None)

        result: list[StreamEvent] = []
        async for e in _process_responses_events(events()):
            result.append(e)

        tokens = [e for e in result if e.type == "token"]
        assert len(tokens) == 2
        assert tokens[0].token == "Hello"

        complete = result[-1]
        assert complete.type == "complete"
        assert complete.content == "Hello world"

    @pytest.mark.asyncio
    async def test_function_call_events(self):
        fn_item = SimpleNamespace(type="function_call", call_id="fc_1", name="search")

        async def events():
            yield _mk_responses_event("response.output_item.added", item=fn_item)
            yield _mk_responses_event(
                "response.function_call_arguments.delta", delta='{"q":'
            )
            yield _mk_responses_event(
                "response.function_call_arguments.delta", delta='"test"}'
            )
            yield _mk_responses_event("response.function_call_arguments.done")
            yield _mk_responses_event("response.completed", response=None)

        result: list[StreamEvent] = []
        async for e in _process_responses_events(events()):
            result.append(e)

        complete = result[-1]
        assert complete.metadata is not None
        tc_list = complete.metadata["tool_calls"]
        assert len(tc_list) == 1
        assert tc_list[0]["id"] == "fc_1"
        assert tc_list[0]["function"]["name"] == "search"
        assert tc_list[0]["function"]["arguments"] == '{"q":"test"}'

    @pytest.mark.asyncio
    async def test_usage_and_response_id(self):
        usage_obj = SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30)
        resp_obj = SimpleNamespace(id="resp_abc", usage=usage_obj)

        async def events():
            yield _mk_responses_event("response.output_text.delta", delta="ok")
            yield _mk_responses_event("response.completed", response=resp_obj)

        result: list[StreamEvent] = []
        async for e in _process_responses_events(events()):
            result.append(e)

        complete = result[-1]
        assert complete.metadata["response_id"] == "resp_abc"
        assert complete.metadata["usage"]["input_tokens"] == 10
        assert complete.metadata["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_empty_delta_ignored(self):
        async def events():
            yield _mk_responses_event("response.output_text.delta", delta="")
            yield _mk_responses_event("response.output_text.delta", delta="hi")
            yield _mk_responses_event("response.completed", response=None)

        result: list[StreamEvent] = []
        async for e in _process_responses_events(events()):
            result.append(e)

        tokens = [e for e in result if e.type == "token"]
        assert len(tokens) == 1
        assert tokens[0].token == "hi"


# ====================================================================== #
# _sync_iter_to_async                                                     #
# ====================================================================== #


class TestSyncIterToAsync:
    @pytest.mark.asyncio
    async def test_basic_bridging(self):
        items: list[int] = []
        async for item in _sync_iter_to_async(iter([1, 2, 3])):
            items.append(item)
        assert items == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_iterable(self):
        items: list[Any] = []
        async for item in _sync_iter_to_async(iter([])):
            items.append(item)
        assert items == []

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        def _failing_iter():
            yield 1
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            async for _ in _sync_iter_to_async(_failing_iter()):
                pass


# ====================================================================== #
# stream_chat_completions (integration)                                   #
# ====================================================================== #


class TestStreamChatCompletionsIntegration:
    @pytest.mark.asyncio
    async def test_async_mode(self):
        """Full async stream with text + usage."""
        usage = _mk_usage(5, 10, 15)

        async def _mock_create(**kwargs):
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}

            async def _gen():
                yield _mk_chat_chunk(content="Hello")
                yield _mk_chat_chunk(content=" world")
                yield _mk_empty_chunk(usage=usage)

            return _gen()

        client = MagicMock()
        client.chat.completions.create = _mock_create

        events: list[StreamEvent] = []
        async for e in stream_chat_completions(
            client, {"model": "gpt-4o"}, async_mode=True
        ):
            events.append(e)

        tokens = [e for e in events if e.type == "token"]
        assert len(tokens) == 2
        assert "".join(e.token for e in tokens) == "Hello world"

        complete = events[-1]
        assert complete.type == "complete"
        assert complete.metadata["usage"]["prompt_tokens"] == 5

    @pytest.mark.asyncio
    async def test_sync_mode(self):
        """Sync stream bridged to async."""

        def _mock_create(**kwargs):
            def _gen():
                yield _mk_chat_chunk(content="sync")
                yield _mk_chat_chunk(content=" stream")

            return _gen()

        client = MagicMock()
        client.chat.completions.create = _mock_create

        events: list[StreamEvent] = []
        async for e in stream_chat_completions(
            client, {"model": "gpt-4o"}, async_mode=False
        ):
            events.append(e)

        complete = events[-1]
        assert complete.type == "complete"
        assert complete.content == "sync stream"

    @pytest.mark.asyncio
    async def test_error_yields_error_event(self):
        async def _mock_create(**kwargs):
            raise ConnectionError("network failure")

        client = MagicMock()
        client.chat.completions.create = _mock_create

        events: list[StreamEvent] = []
        async for e in stream_chat_completions(
            client, {"model": "gpt-4o"}, async_mode=True
        ):
            events.append(e)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "network failure" in events[0].message


# ====================================================================== #
# stream_responses_api (integration)                                      #
# ====================================================================== #


class TestStreamResponsesApiIntegration:
    @pytest.mark.asyncio
    async def test_async_mode_text(self):
        async def _mock_create(**kwargs):
            assert kwargs["stream"] is True

            async def _gen():
                yield _mk_responses_event("response.output_text.delta", delta="Hello")
                yield _mk_responses_event(
                    "response.completed",
                    response=SimpleNamespace(id="r1", usage=None),
                )

            return _gen()

        client = MagicMock()
        client.responses.create = _mock_create

        events: list[StreamEvent] = []
        async for e in stream_responses_api(
            client, {"model": "gpt-4o"}, async_mode=True
        ):
            events.append(e)

        assert events[0].type == "token"
        assert events[0].token == "Hello"
        assert events[-1].type == "complete"
        assert events[-1].metadata["response_id"] == "r1"

    @pytest.mark.asyncio
    async def test_async_mode_with_tool_calls(self):
        fn_item = SimpleNamespace(type="function_call", call_id="fc_1", name="calc")

        async def _mock_create(**kwargs):
            async def _gen():
                yield _mk_responses_event("response.output_item.added", item=fn_item)
                yield _mk_responses_event(
                    "response.function_call_arguments.delta", delta='{"x":1}'
                )
                yield _mk_responses_event("response.function_call_arguments.done")
                yield _mk_responses_event("response.completed", response=None)

            return _gen()

        client = MagicMock()
        client.responses.create = _mock_create

        events: list[StreamEvent] = []
        async for e in stream_responses_api(
            client, {"model": "gpt-4o"}, async_mode=True
        ):
            events.append(e)

        complete = events[-1]
        assert complete.metadata is not None
        tc = complete.metadata["tool_calls"][0]
        assert tc["function"]["name"] == "calc"

    @pytest.mark.asyncio
    async def test_error_yields_error_event(self):
        async def _mock_create(**kwargs):
            raise RuntimeError("API down")

        client = MagicMock()
        client.responses.create = _mock_create

        events: list[StreamEvent] = []
        async for e in stream_responses_api(
            client, {"model": "gpt-4o"}, async_mode=True
        ):
            events.append(e)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "API down" in events[0].message


# ====================================================================== #
# BaseOpenAI.call_stream() routing                                        #
# ====================================================================== #


class TestBaseOpenAICallStreamRouting:
    """Test that call_stream() routes to the correct adapter."""

    @pytest.mark.asyncio
    async def test_routes_to_chat_completions_when_no_native_tools(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            async def _mock_create(**kwargs):
                async def _gen():
                    yield _mk_chat_chunk(content="routed")

                return _gen()

            mock_client = MagicMock()
            mock_client.chat.completions.create = _mock_create

            with (
                patch(
                    "nucleusiq_openai.nb_openai.base.openai.AsyncOpenAI",
                    return_value=mock_client,
                ),
                patch(
                    "nucleusiq_openai.nb_openai.base.ChatCompletionsPayload"
                ) as mock_payload_cls,
            ):
                mock_payload = MagicMock()
                mock_payload.to_api_kwargs.return_value = {"model": "gpt-4o"}
                mock_payload_cls.build.return_value = mock_payload

                from nucleusiq_openai.nb_openai.base import BaseOpenAI

                openai_llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
                openai_llm._client = mock_client

                events: list[StreamEvent] = []
                async for e in openai_llm.call_stream(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                ):
                    events.append(e)

                assert any(e.type == "token" for e in events)
                assert events[-1].type == "complete"

    @pytest.mark.asyncio
    async def test_routes_to_responses_api_when_native_tools(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):

            async def _mock_create(**kwargs):
                async def _gen():
                    yield _mk_responses_event(
                        "response.output_text.delta", delta="searched"
                    )
                    yield _mk_responses_event("response.completed", response=None)

                return _gen()

            mock_client = MagicMock()
            mock_client.responses.create = _mock_create

            with patch(
                "nucleusiq_openai.nb_openai.base.openai.AsyncOpenAI",
                return_value=mock_client,
            ):
                from nucleusiq_openai.nb_openai.base import BaseOpenAI

                openai_llm = BaseOpenAI(model_name="gpt-4o", api_key="test-key")
                openai_llm._client = mock_client

                native_tools = [{"type": "web_search_preview"}]

                events: list[StreamEvent] = []
                async for e in openai_llm.call_stream(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "search cats"}],
                    tools=native_tools,
                ):
                    events.append(e)

                assert events[0].type == "token"
                assert events[0].token == "searched"
                assert events[-1].type == "complete"
