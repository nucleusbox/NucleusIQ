"""Integration tests: Gemini structured output.

Tests native JSON schema / Pydantic model structured output — requires GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from typing import Optional

import pytest
from pydantic import BaseModel

_HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEY, reason="GEMINI_API_KEY not set"),
]


class CityInfo(BaseModel):
    name: str
    country: str
    population: Optional[int] = None


class MathResult(BaseModel):
    expression: str
    result: float


class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_pydantic_model_response(self, gemini_llm):
        """Gemini returns a validated Pydantic model instance."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me about Tokyo."}],
            response_format=CityInfo,
            max_output_tokens=1024,
        )
        assert isinstance(result, CityInfo)
        assert result.name.lower() == "tokyo"
        assert result.country.lower() == "japan"

    @pytest.mark.asyncio
    async def test_structured_math(self, gemini_llm):
        """Structured output works for math results."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What is 15 * 7?"}],
            response_format=MathResult,
            max_output_tokens=1024,
        )
        assert isinstance(result, MathResult)
        assert result.result == 105.0

    @pytest.mark.asyncio
    async def test_structured_with_optional_field(self, gemini_llm):
        """Optional fields can be present or absent."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Tell me about Paris."}],
            response_format=CityInfo,
            max_output_tokens=1024,
        )
        assert isinstance(result, CityInfo)
        assert result.name.lower() == "paris"

    @pytest.mark.asyncio
    async def test_structured_complex_model(self, gemini_llm):
        """Structured output for a model with string and float fields."""
        result = await gemini_llm.call(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": "Write a brief review of the movie 'Inception'. Rate it out of 10.",
                }
            ],
            response_format=MovieReview,
            max_output_tokens=1024,
        )
        assert isinstance(result, MovieReview)
        assert "inception" in result.title.lower()
        assert 0 <= result.rating <= 10
        assert len(result.summary) > 10
