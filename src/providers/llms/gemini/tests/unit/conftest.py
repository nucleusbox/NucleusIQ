"""Shared pytest fixtures for Gemini provider unit tests.

Factory functions live in ``_mock_factories.py`` (importable by test files).
This conftest wraps them as pytest fixtures for convenient injection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit._mock_factories import (
    make_candidate,
    make_code_execution_part,
    make_function_call_part,
    make_response,
    make_stream_chunks,
    make_text_part,
    make_usage_metadata,
)


@pytest.fixture
def simple_response():
    """A simple text-only Gemini response."""
    return make_response()


@pytest.fixture
def tool_call_response():
    """A response with a function call."""
    return make_response(
        candidates=[
            make_candidate(
                [
                    make_function_call_part(
                        "get_weather",
                        {"location": "San Francisco"},
                        call_id="call_123",
                    )
                ]
            )
        ]
    )


@pytest.fixture
def multi_tool_response():
    """A response with multiple function calls."""
    return make_response(
        candidates=[
            make_candidate(
                [
                    make_function_call_part(
                        "get_weather", {"location": "SF"}, "call_1"
                    ),
                    make_function_call_part("get_time", {"timezone": "PST"}, "call_2"),
                ]
            )
        ]
    )


@pytest.fixture
def thinking_response():
    """A response with thinking parts."""
    return make_response(
        candidates=[
            make_candidate(
                [
                    make_text_part("Let me think...", thought=True),
                    make_text_part("The answer is 42."),
                ]
            )
        ],
        usage=make_usage_metadata(thoughts=50),
    )


@pytest.fixture
def code_exec_response():
    """A response with code execution."""
    return make_response(
        candidates=[
            make_candidate(
                [
                    make_code_execution_part("print(2+2)", "4"),
                    make_text_part("The result is 4."),
                ]
            )
        ]
    )


@pytest.fixture
def empty_response():
    """A response with no candidates."""
    return make_response(candidates=[])


@pytest.fixture
def stream_chunks():
    """Simple streaming chunks."""
    return make_stream_chunks(
        ["Hello", ", ", "world!"],
        usage=make_usage_metadata(),
    )


@pytest.fixture
def mock_gemini_client():
    """A mock GeminiClient that returns canned responses."""
    client = MagicMock()
    client.generate_content = AsyncMock(return_value=make_response())
    client.generate_content_stream = AsyncMock(
        return_value=make_stream_chunks(["Hello", " world"])
    )
    return client
