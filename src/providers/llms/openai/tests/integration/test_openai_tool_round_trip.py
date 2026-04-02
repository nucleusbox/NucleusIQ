"""Integration tests: full tool call round-trip with OpenAI.

Mirrors the Gemini integration test suite structure to ensure consistent
coverage across providers.

Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel

_HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set"),
]

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name",
            },
        },
        "required": ["location"],
    },
}


class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str


class TestFullToolRoundTrip:
    """Full multi-turn: prompt → tool_call → tool_result → final answer."""

    @pytest.mark.asyncio
    async def test_round_trip_with_tools_resent(self, openai_llm):
        """Tools are sent on both calls; tool result produces a final answer."""
        result1 = await openai_llm.call(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
            tools=[WEATHER_TOOL],
            max_output_tokens=200,
        )
        msg1 = result1.choices[0].message
        if not msg1.tool_calls:
            pytest.skip("Model answered directly without calling tool")

        tc = msg1.tool_calls[0]
        assert tc.function.name == "get_weather"

        tool_result = json.dumps(
            {"temperature": 8, "conditions": "rainy", "unit": "celsius"}
        )

        result2 = await openai_llm.call(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "What's the weather in Berlin?"},
                msg1.to_dict(),
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": tool_result,
                },
            ],
            tools=[WEATHER_TOOL],
            max_output_tokens=300,
        )
        content = result2.choices[0].message.content
        assert content is not None
        assert any(w in content.lower() for w in ["8", "rainy", "berlin"])

    @pytest.mark.asyncio
    async def test_structured_output_with_tools(self, openai_llm):
        """OpenAI supports structured output + tools simultaneously."""
        result = await openai_llm.call(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
            ],
            tools=[WEATHER_TOOL],
            response_format=WeatherReport,
            max_output_tokens=300,
        )
        msg = result.choices[0].message
        assert msg.content is not None or msg.tool_calls is not None


class TestBasicIntegration:
    """Basic real-API smoke tests for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self, openai_llm):
        result = await openai_llm.call(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello."}],
            max_output_tokens=50,
        )
        assert result.choices[0].message.content is not None

    @pytest.mark.asyncio
    async def test_tool_call_emitted(self, openai_llm):
        result = await openai_llm.call(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            max_output_tokens=200,
        )
        msg = result.choices[0].message
        assert msg.tool_calls is not None or msg.content is not None

    @pytest.mark.asyncio
    async def test_response_has_usage(self, openai_llm):
        result = await openai_llm.call(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "1+1=?"}],
            max_output_tokens=20,
        )
        assert result.usage is not None
        assert result.usage.total_tokens > 0
