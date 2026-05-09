"""Smoke tests against the real Groq API.

Run locally::

    cd src/providers/inference/groq
    export GROQ_API_KEY=gsk_...
    uv run pytest tests/integration -m integration -q

These tests are excluded from the default Groq test run (``-m "not integration"``)
and from CI, so no API key is required in automation.
"""

from __future__ import annotations

import os

import pytest
from nucleusiq.streaming.events import StreamEventType
from nucleusiq_groq._shared.response_models import GroqLLMResponse
from nucleusiq_groq.nb_groq.base import BaseGroq

pytestmark = pytest.mark.integration


def _model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@pytest.fixture
def groq_api_key() -> None:
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set (live smoke tests off)")


@pytest.mark.asyncio
async def test_live_chat_completion(groq_api_key: None) -> None:
    llm = BaseGroq(model_name=_model(), async_mode=True)
    result = await llm.call(
        model=_model(),
        messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
        max_output_tokens=32,
        temperature=0,
    )
    assert isinstance(result, GroqLLMResponse)
    assert result.choices
    text = (result.choices[0].message.content or "").strip()
    assert text


@pytest.mark.asyncio
async def test_live_stream(groq_api_key: None) -> None:
    llm = BaseGroq(model_name=_model(), async_mode=True)
    events: list = []
    async for ev in llm.call_stream(
        model=_model(),
        messages=[{"role": "user", "content": "Count 1 2 3 with spaces only."}],
        max_output_tokens=64,
        temperature=0,
    ):
        events.append(ev)
    assert events
    assert any(e.type == StreamEventType.COMPLETE for e in events)
    complete = next(e for e in events if e.type == StreamEventType.COMPLETE)
    assert complete.content or (complete.metadata or {}).get("usage")
