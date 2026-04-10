"""Importable helpers for tests (same API as repo ``tests/conftest.py``)."""

from nucleusiq.prompts.zero_shot import ZeroShotPrompt


def make_test_prompt(
    system: str = "You are a test assistant.", user: str = "Help the user."
) -> ZeroShotPrompt:
    """Create a minimal ZeroShotPrompt for tests that need an Agent."""
    return ZeroShotPrompt().configure(system=system, user=user)
