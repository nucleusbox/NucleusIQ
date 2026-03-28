"""Integration tests: Gemini native tools (Google Search, Code Execution).

Tests built-in Gemini tools — requires GEMINI_API_KEY.
"""

from __future__ import annotations

import os

import pytest
from nucleusiq_gemini import BaseGemini, GeminiTool

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="GEMINI_API_KEY not set"),
]


def _make_llm():
    return BaseGemini(model_name="gemini-2.5-flash", temperature=0.0)


class TestGoogleSearchTool:
    @pytest.mark.asyncio
    async def test_google_search_grounding(self):
        """Gemini uses Google Search to ground a factual response."""
        llm = _make_llm()
        search_tool = GeminiTool.google_search()

        result = await llm.call(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": "What is the current population of India?",
                }
            ],
            tools=[search_tool],
            max_output_tokens=200,
        )
        content = result.choices[0].message.content
        assert content is not None
        assert len(content) > 0


class TestCodeExecutionTool:
    @pytest.mark.asyncio
    async def test_code_execution(self):
        """Gemini executes Python code inline."""
        llm = _make_llm()
        code_tool = GeminiTool.code_execution()

        result = await llm.call(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": "Calculate the first 10 Fibonacci numbers using code execution.",
                }
            ],
            tools=[code_tool],
            max_output_tokens=500,
        )
        content = result.choices[0].message.content
        assert content is not None


class TestURLContextTool:
    @pytest.mark.asyncio
    async def test_url_context(self):
        """Gemini uses URL context to answer questions about a webpage."""
        llm = _make_llm()
        url_tool = GeminiTool.url_context()

        result = await llm.call(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarize the main points from this page: "
                        "https://en.wikipedia.org/wiki/Python_(programming_language)"
                    ),
                }
            ],
            tools=[url_tool],
            max_output_tokens=500,
        )
        content = result.choices[0].message.content
        assert content is not None
        assert len(content) > 20
