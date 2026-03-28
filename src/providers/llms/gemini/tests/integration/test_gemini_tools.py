"""Integration tests: Gemini function calling (tools).

Tests real tool-calling API calls — requires GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os

import pytest

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


class TestFunctionCalling:
    @pytest.mark.asyncio
    async def test_model_calls_tool(self, gemini_llm):
        """Model returns a function call when given a tool and appropriate prompt."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            max_output_tokens=200,
        )
        msg = result.choices[0].message
        assert msg.tool_calls is not None, "Model should call the weather tool"
        assert len(msg.tool_calls) >= 1
        tc = msg.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert "location" in args

    @pytest.mark.asyncio
    async def test_tool_call_has_id(self, gemini_llm):
        """Tool call includes a call ID for response matching."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            max_output_tokens=200,
        )
        msg = result.choices[0].message
        if msg.tool_calls:
            assert msg.tool_calls[0].id is not None
            assert len(msg.tool_calls[0].id) > 0

    @pytest.mark.asyncio
    async def test_tool_result_round_trip(self, gemini_llm):
        """Full tool call cycle: prompt → tool call → tool result → final answer."""
        result1 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            tools=[WEATHER_TOOL],
            max_output_tokens=200,
        )
        msg1 = result1.choices[0].message
        if not msg1.tool_calls:
            pytest.skip("Model didn't call tool (may answer directly)")

        tc = msg1.tool_calls[0]
        tool_result = json.dumps(
            {"temperature": 15, "conditions": "cloudy", "unit": "celsius"}
        )

        result2 = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "What's the weather in London?"},
                msg1.to_dict(),
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": tool_result,
                },
            ],
            max_output_tokens=200,
        )
        content = result2.choices[0].message.content
        assert content is not None
        assert any(w in content.lower() for w in ["15", "cloudy", "celsius", "london"])

    @pytest.mark.asyncio
    async def test_multiple_tools(self, gemini_llm):
        """Model can choose from multiple tools."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What is 42 * 17?"}],
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            max_output_tokens=200,
        )
        msg = result.choices[0].message
        if msg.tool_calls:
            assert msg.tool_calls[0].function.name == "calculate"

    @pytest.mark.asyncio
    async def test_tool_choice_auto(self, gemini_llm):
        """tool_choice='auto' lets model decide."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_output_tokens=200,
        )
        msg = result.choices[0].message
        assert msg.content is not None or msg.tool_calls is not None

    @pytest.mark.asyncio
    async def test_tool_choice_none(self, gemini_llm):
        """tool_choice='none' prevents tool calls."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[WEATHER_TOOL],
            tool_choice="none",
            max_output_tokens=200,
        )
        msg = result.choices[0].message
        assert msg.content is not None
        assert msg.tool_calls is None
