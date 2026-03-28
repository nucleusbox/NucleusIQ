"""Gemini model metadata and capability helpers.

Centralises model-family detection so that the client, normalizer,
and base class don't duplicate the same string-matching logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiModelInfo:
    """Metadata for a Gemini model family."""

    context_window: int
    max_output_tokens: int
    supports_thinking: bool = False
    supports_function_calling: bool = True
    supports_structured_output: bool = True
    supports_code_execution: bool = False
    supports_grounding: bool = False


_MODEL_REGISTRY: dict[str, GeminiModelInfo] = {
    "gemini-2.5-pro": GeminiModelInfo(
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_thinking=True,
        supports_code_execution=True,
        supports_grounding=True,
    ),
    "gemini-2.5-flash": GeminiModelInfo(
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_thinking=True,
        supports_code_execution=True,
        supports_grounding=True,
    ),
    "gemini-2.5-flash-lite": GeminiModelInfo(
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_thinking=True,
        supports_code_execution=True,
        supports_grounding=True,
    ),
    "gemini-2.0-flash": GeminiModelInfo(
        context_window=1_048_576,
        max_output_tokens=8_192,
        supports_code_execution=True,
        supports_grounding=True,
    ),
    "gemini-2.0-flash-lite": GeminiModelInfo(
        context_window=1_048_576,
        max_output_tokens=8_192,
    ),
    "gemini-1.5-pro": GeminiModelInfo(
        context_window=2_097_152,
        max_output_tokens=8_192,
        supports_code_execution=True,
        supports_grounding=True,
    ),
    "gemini-1.5-flash": GeminiModelInfo(
        context_window=1_048_576,
        max_output_tokens=8_192,
        supports_code_execution=True,
        supports_grounding=True,
    ),
}


def _match_model(model: str) -> str | None:
    """Find the best matching registry key for a model string.

    Handles exact matches and prefix-based lookups (e.g.
    ``"gemini-2.5-pro-preview-05-06"`` → ``"gemini-2.5-pro"``).
    """
    m = (model or "").lower().strip()
    if m in _MODEL_REGISTRY:
        return m
    for key in sorted(_MODEL_REGISTRY, key=len, reverse=True):
        if m.startswith(key):
            return key
    return None


def get_model_info(model: str) -> GeminiModelInfo | None:
    """Return metadata for a Gemini model, or ``None`` if unknown."""
    key = _match_model(model)
    return _MODEL_REGISTRY.get(key) if key else None


def get_context_window(model: str) -> int:
    """Return context window size (defaults to 1M if unknown)."""
    info = get_model_info(model)
    return info.context_window if info else 1_048_576


def get_max_output_tokens(model: str) -> int:
    """Return max output tokens (defaults to 8192 if unknown)."""
    info = get_model_info(model)
    return info.max_output_tokens if info else 8_192


def supports_thinking(model: str) -> bool:
    """Check if a model supports the thinking/reasoning feature."""
    info = get_model_info(model)
    return info.supports_thinking if info else False


def supports_function_calling(model: str) -> bool:
    """Check if a model supports function calling."""
    info = get_model_info(model)
    return info.supports_function_calling if info else True


def supports_structured_output(model: str) -> bool:
    """Check if a model supports native structured output."""
    info = get_model_info(model)
    return info.supports_structured_output if info else True
