"""Shared fixtures for Gemini provider integration tests.

These tests make **real API calls** to Google's Gemini API.
They require ``GEMINI_API_KEY`` to be set in the environment
(loaded from ``.env`` by the root conftest).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

_HAS_GEMINI_KEY = bool(os.getenv("GEMINI_API_KEY"))

_repo_root = Path(__file__).resolve().parents[6]
_nucleusiq_core = str(_repo_root / "src" / "nucleusiq")
if _nucleusiq_core not in sys.path:
    sys.path.insert(0, _nucleusiq_core)


def _make_gemini_llm(model: str = "gemini-2.5-flash", temperature: float = 0.0):
    """Create a real BaseGemini instance."""
    from nucleusiq_gemini import BaseGemini

    return BaseGemini(model_name=model, temperature=temperature)


@pytest.fixture
def gemini_llm():
    """A real BaseGemini instance for integration tests."""
    return _make_gemini_llm()


@pytest.fixture
def gemini_llm_creative():
    """A real BaseGemini with higher temperature."""
    return _make_gemini_llm(temperature=0.9)


class AddTool:
    """Simple calculator tool for testing function calling."""

    def __init__(self):
        self.name = "add"
        self.description = "Add two numbers together"

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        }


@pytest.fixture
def add_tool_spec():
    """A converted add tool spec for Gemini."""
    tool = AddTool()
    llm = _make_gemini_llm()
    return llm.convert_tool_specs([tool])
