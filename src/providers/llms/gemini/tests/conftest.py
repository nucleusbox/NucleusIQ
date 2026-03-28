"""Root conftest for Gemini provider tests.

Loads ``.env`` from repo root so that GEMINI_API_KEY is available
for integration tests. Provides shared fixtures used by both
unit and integration test suites.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[5]
_env_file = _repo_root / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env_file, override=False)
    except ImportError:
        pass

_nucleusiq_core = str(_repo_root / "src" / "nucleusiq")
if _nucleusiq_core not in sys.path:
    sys.path.insert(0, _nucleusiq_core)

_HAS_GEMINI_KEY = bool(os.getenv("GEMINI_API_KEY"))


@pytest.fixture
def gemini_api_key() -> str | None:
    """Fixture for Gemini API key."""
    return os.getenv("GEMINI_API_KEY")


@pytest.fixture
def skip_if_no_gemini_key(gemini_api_key):
    """Fixture to skip tests if GEMINI_API_KEY is not set."""
    if not gemini_api_key:
        pytest.skip("GEMINI_API_KEY not set, skipping integration test")


def has_gemini_key() -> bool:
    """Helper for ``@pytest.mark.skipif`` decorators."""
    return _HAS_GEMINI_KEY
