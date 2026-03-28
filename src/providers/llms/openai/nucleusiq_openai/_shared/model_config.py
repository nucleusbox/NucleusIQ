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
