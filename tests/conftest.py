"""
Pytest configuration and fixtures for NucleusIQ tests.
"""

import pytest
import os
from typing import Optional


@pytest.fixture
def openai_api_key() -> Optional[str]:
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

