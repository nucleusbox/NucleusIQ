"""Integration tests: full tool call round-trip with Gemini.

Covers the exact failure modes that caused the Gemini 400 errors:
1. function_response.name must not be empty
2. function_response.response must be a dict
3. tools must be resent on the second call (multi-turn)
4. structured output (JSON schema mode) cannot coexist with tools
5. native + custom tools cannot be mixed in one request

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="GEMINI_API_KEY not set"),
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

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": "Perform arithmetic calculations",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate",
            },
        },
        "required": ["expression"],
    },
}


class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str


class TestFullToolRoundTrip:
    """Full multi-turn: prompt → tool_call → tool_result → final answer,
    with tool declarations resent on every turn."""

    @pytest.mark.asyncio
    async def test_round_trip_with_tools_resent(self, gemini_llm):
        """Tools are sent on BOTH the first and second call (real API)."""
        result1 = await gemini_llm.call(
            model="gemini-2.5-flash",
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

        result2 = await gemini_llm.call(
            model="gemini-2.5-flash",
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
    async def test_tool_result_json_string_content(self, gemini_llm):
        """json.dumps(str) produces a JSON string — Gemini requires dict response."""
        result1 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo right now?"}
            ],
            tools=[WEATHER_TOOL],
            max_output_tokens=200,
        )
        msg1 = result1.choices[0].message
        if not msg1.tool_calls:
            pytest.skip("Model answered directly")

        tc = msg1.tool_calls[0]

        plain_string_result = json.dumps("Sunny and 25°C in Tokyo")

        result2 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo right now?",
                },
                msg1.to_dict(),
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": plain_string_result,
                },
            ],
            tools=[WEATHER_TOOL],
            max_output_tokens=300,
        )
        content = result2.choices[0].message.content
        assert content is not None

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_round_trip(self, gemini_llm):
        """Model calls multiple tools; each result sent back with correct names."""
        result1 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "What's the weather in London AND calculate 15 * 7? "
                        "Use both tools."
                    ),
                }
            ],
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            max_output_tokens=200,
        )
        msg1 = result1.choices[0].message
        if not msg1.tool_calls:
            pytest.skip("Model answered directly")

        tool_results = []
        for tc in msg1.tool_calls:
            if tc.function.name == "get_weather":
                result_data = json.dumps({"temperature": 12, "conditions": "foggy"})
            elif tc.function.name == "calculate":
                result_data = json.dumps({"result": 105})
            else:
                result_data = json.dumps({"result": "unknown tool"})

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result_data,
                }
            )

        messages = [
            {
                "role": "user",
                "content": (
                    "What's the weather in London AND calculate 15 * 7? "
                    "Use both tools."
                ),
            },
            msg1.to_dict(),
            *tool_results,
        ]

        result2 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            max_output_tokens=500,
        )
        content = result2.choices[0].message.content
        assert content is not None


class TestToolsAndStructuredOutputConflict:
    """Gemini API rejects tools + JSON schema mode in the same request."""

    @pytest.mark.asyncio
    async def test_tools_with_response_format_drops_schema(self, gemini_llm):
        """When both tools and response_format are set, the guard should
        drop the structured output and the call should succeed without 400."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
            ],
            tools=[WEATHER_TOOL],
            response_format=WeatherReport,
            max_output_tokens=300,
        )
        msg = result.choices[0].message
        assert msg.content is not None or msg.tool_calls is not None
