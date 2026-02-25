"""Model-specific quirks and parameter mapping.

Centralises model-family detection so that backends don't need to
duplicate the same string-matching logic.
"""

from __future__ import annotations


def uses_max_completion_tokens(model: str) -> bool:
    """Some newer models reject ``max_tokens`` and require ``max_completion_tokens``."""
    m = (model or "").lower()
    return m.startswith("gpt-5")


def is_strict_defaults_model(model: str) -> bool:
    """Some models only accept default sampling parameters."""
    m = (model or "").lower()
    return m.startswith("gpt-5")
