"""Find and load a repo-root ``.env`` for example scripts (e.g. ``ANTHROPIC_API_KEY``)."""

from __future__ import annotations

from pathlib import Path


def load_repo_dotenv() -> None:
    """Load the first ``.env`` found walking up from this file or CWD."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    start = Path(__file__).resolve().parent
    for d in [start, *start.parents]:
        env_file = d / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return
    load_dotenv()
