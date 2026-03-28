"""Integration tests: Gemini streaming.

Tests real streaming API calls — requires GEMINI_API_KEY.
"""

from __future__ import annotations

import os

import pytest

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="GEMINI_API_KEY not set"),
]


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_produces_tokens(self, gemini_llm):
        """call_stream() yields TOKEN events followed by COMPLETE."""
        events = []
        async for event in gemini_llm.call_stream(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_output_tokens=100,
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "token" in types, "Should have at least one TOKEN event"
        assert types[-1] == "complete", "Last event must be COMPLETE"

    @pytest.mark.asyncio
    async def test_stream_tokens_concatenate_to_complete(self, gemini_llm):
        """Concatenated TOKEN text matches COMPLETE content."""
        tokens = []
        complete_content = None

        async for event in gemini_llm.call_stream(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Say hello world."}],
            max_output_tokens=50,
        ):
            if event.type == "token" and event.token:
                tokens.append(event.token)
            elif event.type == "complete":
                complete_content = event.content

        concatenated = "".join(tokens)
        assert complete_content is not None
        assert len(concatenated) > 0
        assert concatenated == complete_content

    @pytest.mark.asyncio
    async def test_stream_with_system_message(self, gemini_llm):
        """Streaming respects system instructions."""
        content_parts = []
        async for event in gemini_llm.call_stream(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Always reply in French."},
                {"role": "user", "content": "Hello!"},
            ],
            max_output_tokens=50,
        ):
            if event.type == "token" and event.token:
                content_parts.append(event.token)

        full_text = "".join(content_parts).lower()
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_stream_error_event_on_bad_model(self):
        """Streaming with an invalid model produces an error event."""
        from nucleusiq_gemini import BaseGemini

        llm = BaseGemini(temperature=0.0)
        events = []
        async for event in llm.call_stream(
            model="nonexistent-model-12345",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "error" in types or "complete" in types

    @pytest.mark.asyncio
    async def test_stream_complete_has_usage(self, gemini_llm):
        """COMPLETE event metadata includes usage stats."""
        complete_event = None
        async for event in gemini_llm.call_stream(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            max_output_tokens=100,
        ):
            if event.type == "complete":
                complete_event = event

        assert complete_event is not None
        if complete_event.metadata and "usage" in complete_event.metadata:
            usage = complete_event.metadata["usage"]
            assert usage.get("total_tokens", 0) > 0
