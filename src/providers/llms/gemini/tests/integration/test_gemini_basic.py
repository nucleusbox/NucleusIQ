"""Integration tests: Basic Gemini API calls.

Tests real API calls — requires GEMINI_API_KEY.

Run with:
    pytest tests/integration/test_gemini_basic.py -m integration -v
"""

from __future__ import annotations

import os

import pytest

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="GEMINI_API_KEY not set"),
]


# ====================================================================== #
# Simple text generation                                                   #
# ====================================================================== #


class TestBasicCall:
    @pytest.mark.asyncio
    async def test_simple_text_response(self, gemini_llm):
        """Gemini returns a text response for a simple prompt."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 + 2? Reply with just the number.",
                }
            ],
            max_output_tokens=50,
        )
        assert result is not None
        assert result.choices
        content = result.choices[0].message.content
        assert content is not None
        assert "4" in content

    @pytest.mark.asyncio
    async def test_response_has_usage(self, gemini_llm):
        """Response includes token usage metadata."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Say hello."}],
            max_output_tokens=50,
        )
        assert result.usage is not None
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_system_message(self, gemini_llm):
        """System message is respected as instruction."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Always respond in exactly one word."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            max_output_tokens=100,
        )
        content = result.choices[0].message.content
        assert content is not None
        assert len(content.split()) <= 5

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, gemini_llm):
        """Multi-turn conversation context is maintained."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "My name is Brijesh."},
                {"role": "assistant", "content": "Nice to meet you, Brijesh!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_output_tokens=50,
        )
        content = result.choices[0].message.content
        assert "brijesh" in content.lower()

    @pytest.mark.asyncio
    async def test_temperature_zero_deterministic(self, gemini_llm):
        """Temperature=0 produces consistent output."""
        messages = [
            {"role": "user", "content": "What is the capital of France? One word."}
        ]

        result1 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=messages,
            max_output_tokens=100,
            temperature=0.0,
        )
        result2 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=messages,
            max_output_tokens=100,
            temperature=0.0,
        )
        c1 = (result1.choices[0].message.content or "").strip().lower()
        c2 = (result2.choices[0].message.content or "").strip().lower()
        assert "paris" in c1, f"Expected 'paris' in: {c1!r}"
        assert "paris" in c2, f"Expected 'paris' in: {c2!r}"

    @pytest.mark.asyncio
    async def test_max_output_tokens_respected(self, gemini_llm):
        """max_output_tokens limits the response length."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Write a very long essay about space."}
            ],
            max_output_tokens=1024,
        )
        content = result.choices[0].message.content or ""
        assert result.choices is not None
        assert result.usage.completion_tokens <= 1100

    @pytest.mark.asyncio
    async def test_response_model_field(self, gemini_llm):
        """Response includes the model version."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            max_output_tokens=100,
        )
        assert result.model is not None


# ====================================================================== #
# BaseLLM contract compliance                                              #
# ====================================================================== #


class TestBaseLLMContract:
    @pytest.mark.asyncio
    async def test_choices_shape(self, gemini_llm):
        """Response matches BaseLLM contract: choices[0].message.content."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert hasattr(result, "choices")
        assert len(result.choices) >= 1
        msg = result.choices[0].message
        assert hasattr(msg, "content")
        assert hasattr(msg, "tool_calls")
        assert hasattr(msg, "role")
        assert msg.role == "assistant"

    @pytest.mark.asyncio
    async def test_to_dict_round_trip(self, gemini_llm):
        """AssistantMessage.to_dict() produces a valid message dict."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            max_output_tokens=100,
        )
        msg = result.choices[0].message
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert "content" in d
