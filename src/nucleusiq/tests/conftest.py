"""
Pytest configuration and fixtures for NucleusIQ tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


@pytest.fixture
def openai_api_key() -> str | None:
    """Fixture for OpenAI API key."""
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture
def skip_if_no_openai_key(openai_api_key):
    """Fixture to skip tests if OpenAI API key is not available."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping test")


@pytest.fixture
def mock_openai_client():
    """Fixture for mock OpenAI client (for unit tests)."""
    # This can be extended with a proper mock if needed
    pass
