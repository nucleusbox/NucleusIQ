"""Shared fixtures for OpenAI provider integration tests.

These tests make **real API calls** to OpenAI.
They require ``OPENAI_API_KEY`` to be set in the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

_HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))

_repo_root = Path(__file__).resolve().parents[6]
_nucleusiq_core = str(_repo_root / "src" / "nucleusiq")
if _nucleusiq_core not in sys.path:
    sys.path.insert(0, _nucleusiq_core)


def _make_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Create a real BaseOpenAI instance."""
    from nucleusiq_openai import BaseOpenAI

    return BaseOpenAI(model_name=model, temperature=temperature)


@pytest.fixture
def openai_llm():
    """A real BaseOpenAI instance for integration tests."""
    return _make_openai_llm()


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
