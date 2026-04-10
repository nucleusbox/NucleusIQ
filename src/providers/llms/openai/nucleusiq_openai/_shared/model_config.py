"""Model-specific quirks and parameter mapping.

Centralises model-family detection so that backends don't need to
duplicate the same string-matching logic.

OpenAI has three wire-level parameter names for controlling max tokens:

- ``max_tokens`` — Chat Completions, non-reasoning models (GPT-4o, etc.)
- ``max_completion_tokens`` — Chat Completions, reasoning models (o1, o3,
  o4-mini, GPT-5) and Azure OpenAI reasoning deployments.
- ``max_output_tokens`` — Responses API (all models).

The framework exposes a single ``max_output_tokens`` parameter to users;
this module decides which wire name each model actually needs.
"""

from __future__ import annotations

import re

_O_SERIES_RE = re.compile(r"^o[1-9]")


def is_o_series(model: str) -> bool:
    """Return ``True`` for OpenAI o-series reasoning models (o1, o3, o4-mini, ...)."""
    m = (model or "").lower()
    return bool(_O_SERIES_RE.match(m))


def uses_max_completion_tokens(model: str) -> bool:
    """Return ``True`` when Chat Completions requires ``max_completion_tokens``
    instead of ``max_tokens``.

    Applies to o-series reasoning models and GPT-5 family.  These models
    reject the legacy ``max_tokens`` field.
    """
    m = (model or "").lower()
    return m.startswith("gpt-5") or is_o_series(m)


def is_strict_defaults_model(model: str) -> bool:
    """Return ``True`` for models that reject custom sampling parameters.

    o-series models have fixed temperature / top_p;  GPT-5 also
    uses its own sampling defaults.
    """
    m = (model or "").lower()
    return m.startswith("gpt-5") or is_o_series(m)


# ------------------------------------------------------------------ #
# Context window registry                                              #
# ------------------------------------------------------------------ #

_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    "gpt-5": 1_050_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
}

_DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model: str) -> int:
    """Return context window size for an OpenAI model.

    Matches on exact name first, then prefix-matches for versioned
    model names (e.g. ``gpt-4o-2024-08-06`` → ``gpt-4o``).
    """
    m = (model or "").lower()
    if m in _CONTEXT_WINDOWS:
        return _CONTEXT_WINDOWS[m]
    for prefix, size in _CONTEXT_WINDOWS.items():
        if m.startswith(prefix):
            return size
    return _DEFAULT_CONTEXT_WINDOW
