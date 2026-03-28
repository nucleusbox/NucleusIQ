"""Tests for nucleusiq_gemini.nb_gemini.stream_adapters."""

import pytest
from nucleusiq_gemini.nb_gemini.stream_adapters import (
    _process_gemini_chunks,
    _sync_iter_to_async,
    stream_gemini,
)

from tests.unit._mock_factories import (
    make_candidate,
    make_stream_chunks,
    make_text_part,
    make_usage_metadata,
)


async def _collect_events(async_gen):
    """Helper to collect all events from an async generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


class TestSyncIterToAsync:
    @pytest.mark.asyncio
    async def test_basic(self):
        items = [1, 2, 3]
        result = []
        async for item in _sync_iter_to_async(items):
            result.append(item)
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty(self):
        result = []
        async for item in _sync_iter_to_async([]):
            result.append(item)
        assert result == []

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        def bad_iter():
            yield 1
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            async for _ in _sync_iter_to_async(bad_iter()):
                pass


class TestProcessGeminiChunks:
    @pytest.mark.asyncio
    async def test_text_chunks(self):
        chunks = make_stream_chunks(["Hello", " ", "world!"])

        async def mock_iter():
            for c in chunks:
                yield c

        events = await _collect_events(_process_gemini_chunks(mock_iter()))
        token_events = [e for e in events if e.type == "token"]
        complete_events = [e for e in events if e.type == "complete"]

        assert len(token_events) == 3
        assert token_events[0].token == "Hello"
        assert token_events[1].token == " "
        assert token_events[2].token == "world!"
        assert len(complete_events) == 1
        assert complete_events[0].content == "Hello world!"

    @pytest.mark.asyncio
    async def test_with_usage(self):
        chunks = make_stream_chunks(
            ["Hi"],
            usage=make_usage_metadata(prompt=5, candidates=10, total=15),
        )

        async def mock_iter():
            for c in chunks:
                yield c

        events = await _collect_events(_process_gemini_chunks(mock_iter()))
        complete = [e for e in events if e.type == "complete"][0]
        assert complete.metadata["usage"]["prompt_tokens"] == 5
        assert complete.metadata["usage"]["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_with_function_calls(self):
        chunks = make_stream_chunks(
            [],
            function_calls=[("get_weather", {"location": "SF"})],
        )

        async def mock_iter():
            for c in chunks:
                yield c

        events = await _collect_events(_process_gemini_chunks(mock_iter()))
        complete = [e for e in events if e.type == "complete"][0]
        assert "tool_calls" in complete.metadata
        tc = complete.metadata["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_thinking_events(self):
        from types import SimpleNamespace

        think_part = make_text_part("Thinking...", thought=True)
        text_part = make_text_part("Answer")

        chunks = [
            SimpleNamespace(
                candidates=[make_candidate([think_part])],
                usage_metadata=None,
            ),
            SimpleNamespace(
                candidates=[make_candidate([text_part])],
                usage_metadata=None,
            ),
        ]

        async def mock_iter():
            for c in chunks:
                yield c

        events = await _collect_events(_process_gemini_chunks(mock_iter()))
        thinking = [e for e in events if e.type == "thinking"]
        tokens = [e for e in events if e.type == "token"]
        assert len(thinking) == 1
        assert thinking[0].message == "Thinking..."
        assert len(tokens) == 1  # only answer text (thought skips text)

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        async def mock_iter():
            return
            yield

        events = await _collect_events(_process_gemini_chunks(mock_iter()))
        assert len(events) == 1
        assert events[0].type == "complete"
        assert events[0].content == ""


class TestStreamGemini:
    @pytest.mark.asyncio
    async def test_basic_stream(self):
        chunks = make_stream_chunks(["Hello", " world"])
        events = await _collect_events(stream_gemini(chunks))

        token_events = [e for e in events if e.type == "token"]
        complete_events = [e for e in events if e.type == "complete"]
        assert len(token_events) == 2
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_error_handling(self):
        def bad_stream():
            raise ValueError("API error")
            yield  # noqa: F401, E741

        events = await _collect_events(stream_gemini(bad_stream()))
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "API error" in error_events[0].message
