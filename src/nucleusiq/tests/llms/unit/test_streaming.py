"""Tests for the streaming data model and BaseLLM / MockLLM streaming support.

Covers:
    1. StreamEventType enum values
    2. StreamEvent model creation, serialization, and convenience constructors
    3. BaseLLM.call_stream() fallback (non-streaming → single COMPLETE event)
    4. MockLLM.call_stream() real token-by-token streaming
    5. SSE serialization
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.streaming.events import StreamEvent, StreamEventType


# ═══════════════════════════════════════════════════════════════════════════════
# StreamEventType
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamEventType:
    def test_values(self):
        assert StreamEventType.TOKEN == "token"
        assert StreamEventType.COMPLETE == "complete"
        assert StreamEventType.TOOL_CALL_START == "tool_start"
        assert StreamEventType.TOOL_CALL_END == "tool_end"
        assert StreamEventType.LLM_CALL_START == "llm_start"
        assert StreamEventType.LLM_CALL_END == "llm_end"
        assert StreamEventType.THINKING == "thinking"
        assert StreamEventType.ERROR == "error"

    def test_is_string_enum(self):
        assert isinstance(StreamEventType.TOKEN, str)
        assert StreamEventType.TOKEN == "token"

    def test_all_members(self):
        assert len(StreamEventType) == 8


# ═══════════════════════════════════════════════════════════════════════════════
# StreamEvent — model basics
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamEventModel:
    def test_minimal_creation(self):
        evt = StreamEvent(type=StreamEventType.TOKEN, token="hi")
        assert evt.type == "token"
        assert evt.token == "hi"
        assert evt.content is None

    def test_optional_fields_default_none(self):
        evt = StreamEvent(type=StreamEventType.COMPLETE, content="done")
        assert evt.tool_name is None
        assert evt.tool_args is None
        assert evt.tool_result is None
        assert evt.call_count is None
        assert evt.message is None
        assert evt.metadata is None

    def test_metadata_attachment(self):
        evt = StreamEvent(
            type=StreamEventType.TOKEN,
            token="x",
            metadata={"provider": "openai", "model": "gpt-4o"},
        )
        assert evt.metadata["provider"] == "openai"

    def test_model_dump_excludes_none(self):
        evt = StreamEvent.token_event("hi")
        dumped = evt.model_dump(exclude_none=True)
        assert "content" not in dumped
        assert "tool_name" not in dumped
        assert dumped["type"] == "token"
        assert dumped["token"] == "hi"


# ═══════════════════════════════════════════════════════════════════════════════
# StreamEvent — convenience constructors
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamEventConstructors:
    def test_token_event(self):
        evt = StreamEvent.token_event("hello")
        assert evt.type == "token"
        assert evt.token == "hello"

    def test_complete_event(self):
        evt = StreamEvent.complete_event("full text")
        assert evt.type == "complete"
        assert evt.content == "full text"

    def test_tool_start_event(self):
        evt = StreamEvent.tool_start_event("search", {"query": "cats"})
        assert evt.type == "tool_start"
        assert evt.tool_name == "search"
        assert evt.tool_args == {"query": "cats"}

    def test_tool_start_event_no_args(self):
        evt = StreamEvent.tool_start_event("ping")
        assert evt.tool_args is None

    def test_tool_end_event(self):
        evt = StreamEvent.tool_end_event("search", "found 3 results")
        assert evt.type == "tool_end"
        assert evt.tool_name == "search"
        assert evt.tool_result == "found 3 results"

    def test_llm_start_event(self):
        evt = StreamEvent.llm_start_event(1)
        assert evt.type == "llm_start"
        assert evt.call_count == 1

    def test_llm_end_event(self):
        evt = StreamEvent.llm_end_event(2)
        assert evt.type == "llm_end"
        assert evt.call_count == 2

    def test_thinking_event(self):
        evt = StreamEvent.thinking_event("reasoning about the problem")
        assert evt.type == "thinking"
        assert evt.message == "reasoning about the problem"

    def test_error_event(self):
        evt = StreamEvent.error_event("connection failed")
        assert evt.type == "error"
        assert evt.message == "connection failed"


# ═══════════════════════════════════════════════════════════════════════════════
# StreamEvent — SSE serialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamEventSSE:
    def test_to_sse_format(self):
        evt = StreamEvent.token_event("hi")
        sse = evt.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse[len("data: ") : -2])
        assert payload["type"] == "token"
        assert payload["token"] == "hi"
        assert "content" not in payload

    def test_to_sse_complete(self):
        evt = StreamEvent.complete_event("all done")
        payload = json.loads(evt.to_sse()[len("data: ") : -2])
        assert payload["type"] == "complete"
        assert payload["content"] == "all done"


# ═══════════════════════════════════════════════════════════════════════════════
# BaseLLM.call_stream() — non-streaming fallback
# ═══════════════════════════════════════════════════════════════════════════════


class _FallbackLLM(BaseLLM):
    """Concrete LLM that only implements call() — uses default call_stream()."""

    async def call(self, **kwargs: Any) -> Any:
        class _Msg:
            content = "fallback response"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _EmptyLLM(BaseLLM):
    """LLM that returns None content — tests empty fallback path."""

    async def call(self, **kwargs: Any) -> Any:
        class _Msg:
            content = None

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _NoneResponseLLM(BaseLLM):
    """LLM that returns None — tests null response path."""

    async def call(self, **kwargs: Any) -> Any:
        return None


class TestBaseLLMCallStreamFallback:
    @pytest.mark.asyncio
    async def test_fallback_yields_complete_event(self):
        llm = _FallbackLLM()
        events: list[StreamEvent] = []
        async for evt in llm.call_stream(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        ):
            events.append(evt)

        assert len(events) == 1
        assert events[0].type == "complete"
        assert events[0].content == "fallback response"

    @pytest.mark.asyncio
    async def test_fallback_empty_content_yields_nothing(self):
        llm = _EmptyLLM()
        events: list[StreamEvent] = []
        async for evt in llm.call_stream(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        ):
            events.append(evt)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_fallback_none_response_yields_nothing(self):
        llm = _NoneResponseLLM()
        events: list[StreamEvent] = []
        async for evt in llm.call_stream(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        ):
            events.append(evt)

        assert len(events) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BaseLLM._extract_content_from_response()
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractContent:
    def test_none_response(self):
        assert BaseLLM._extract_content_from_response(None) is None

    def test_no_choices(self):
        class R:
            choices = []

        assert BaseLLM._extract_content_from_response(R()) is None

    def test_dict_message(self):
        class R:
            class C:
                message = {"content": "from dict"}

            choices = [C()]

        assert BaseLLM._extract_content_from_response(R()) == "from dict"

    def test_attr_message(self):
        class Msg:
            content = "from attr"

        class R:
            class C:
                message = Msg()

            choices = [C()]

        assert BaseLLM._extract_content_from_response(R()) == "from attr"

    def test_no_message(self):
        class R:
            class C:
                pass

            choices = [C()]

        assert BaseLLM._extract_content_from_response(R()) is None


# ═══════════════════════════════════════════════════════════════════════════════
# MockLLM.call_stream()
# ═══════════════════════════════════════════════════════════════════════════════


class TestMockLLMCallStream:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens_then_complete(self):
        llm = MockLLM(stream_chunk_size=5)
        events: list[StreamEvent] = []
        async for evt in llm.call_stream(
            model="mock",
            messages=[{"role": "user", "content": "hello world"}],
        ):
            events.append(evt)

        token_events = [e for e in events if e.type == "token"]
        complete_events = [e for e in events if e.type == "complete"]

        assert len(token_events) >= 1
        assert len(complete_events) == 1

        reconstructed = "".join(e.token for e in token_events)
        assert reconstructed == complete_events[0].content

    @pytest.mark.asyncio
    async def test_stream_chunk_size_1(self):
        """Each character should be a separate token event."""
        llm = MockLLM(stream_chunk_size=1)
        events: list[StreamEvent] = []
        async for evt in llm.call_stream(
            model="mock",
            messages=[{"role": "user", "content": "AB"}],
        ):
            events.append(evt)

        token_events = [e for e in events if e.type == "token"]
        full_text = "Echo: AB"
        assert len(token_events) == len(full_text)
        for i, e in enumerate(token_events):
            assert e.token == full_text[i]

    @pytest.mark.asyncio
    async def test_stream_with_tools_yields_complete(self):
        """When tools trigger a function call, stream yields a complete event."""
        llm = MockLLM()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                    },
                },
            }
        ]
        events: list[StreamEvent] = []
        async for evt in llm.call_stream(
            model="mock",
            messages=[{"role": "user", "content": "5 + 3"}],
            tools=tools,
        ):
            events.append(evt)

        assert len(events) == 1
        assert events[0].type == "complete"

    @pytest.mark.asyncio
    async def test_stream_with_function_result(self):
        llm = MockLLM()
        llm._call_count = 1
        events: list[StreamEvent] = []
        async for evt in llm.call_stream(
            model="mock",
            messages=[
                {"role": "user", "content": "sum"},
                {"role": "function", "content": "8"},
            ],
        ):
            events.append(evt)

        complete = [e for e in events if e.type == "complete"]
        assert len(complete) == 1
        assert "8" in complete[0].content

    @pytest.mark.asyncio
    async def test_stream_increments_call_count(self):
        llm = MockLLM()
        assert llm._call_count == 0
        async for _ in llm.call_stream(
            model="mock",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
        assert llm._call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Import paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestImports:
    def test_import_from_streaming_package(self):
        from nucleusiq.streaming import StreamEvent, StreamEventType

        assert StreamEventType.TOKEN == "token"
        assert StreamEvent.token_event("x").token == "x"

    def test_import_from_llms_package(self):
        from nucleusiq.llms import StreamEvent, StreamEventType

        assert StreamEventType.COMPLETE == "complete"

    def test_import_from_agents_package(self):
        from nucleusiq.agents import StreamEvent, StreamEventType

        assert StreamEventType.ERROR == "error"
