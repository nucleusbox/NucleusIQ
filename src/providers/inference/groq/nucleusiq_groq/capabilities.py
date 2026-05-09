"""Groq model capability hints (parallel tools, etc.).

Groq’s OpenAI-compatible surface accepts ``parallel_tool_calls`` for models that
support it. The live model matrix can drift; we keep a **conservative allowlist**
for strict validation and emit **warnings** by default for unknown models.
"""

from __future__ import annotations

import logging

from nucleusiq.llms.errors import InvalidRequestError

__all__ = [
    "PARALLEL_TOOL_CALLS_DOCUMENTED_MODELS",
    "check_parallel_tool_calls_capability",
]

# Models aligned with BaseGroq’s known context table / Groq docs examples; expand as verified.
PARALLEL_TOOL_CALLS_DOCUMENTED_MODELS: frozenset[str] = frozenset(
    {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
)


def check_parallel_tool_calls_capability(
    model_id: str,
    parallel_tool_calls: bool | None,
    *,
    strict: bool,
    logger: logging.Logger,
) -> None:
    """If *parallel_tool_calls* is true, warn or reject when *model_id* is unknown.

    When *strict* is false (default via :class:`GroqLLMParams`), only log a warning.
    When *strict* is true, raise :class:`InvalidRequestError` before the HTTP call.
    """
    if not parallel_tool_calls:
        return
    mid = model_id.strip()
    if mid in PARALLEL_TOOL_CALLS_DOCUMENTED_MODELS:
        return
    msg = (
        f"parallel_tool_calls=True for model {mid!r} is not in the built-in "
        "capability allowlist; Groq may reject the request or behave unexpectedly."
    )
    if strict:
        raise InvalidRequestError.from_provider_error(
            provider="groq",
            message=msg + " Set strict_model_capabilities=False to skip this check.",
            status_code=400,
            original_error=None,
        )
    logger.warning("Groq capability: %s", msg)
