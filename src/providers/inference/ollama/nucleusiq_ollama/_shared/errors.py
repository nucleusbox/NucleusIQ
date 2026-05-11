"""Map ``ollama.ResponseError`` to NucleusIQ :mod:`nucleusiq.llms.errors`."""

from __future__ import annotations

from typing import Any

from nucleusiq.llms.errors import (
    InvalidRequestError,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
)
from ollama import ResponseError


def map_ollama_response_error(exc: ResponseError) -> ProviderError:
    """Translate an ``ollama.ResponseError`` into a framework ``LLMError``."""
    code = getattr(exc, "status_code", None) or 0
    msg = str(getattr(exc, "error", None) or exc)
    orig: Any = exc

    if code == 404:
        return ModelNotFoundError.from_provider_error(
            provider="ollama",
            message=msg,
            status_code=404,
            original_error=orig,
        )
    if code in (400, 409):
        return InvalidRequestError.from_provider_error(
            provider="ollama",
            message=msg,
            status_code=code,
            original_error=orig,
        )
    if code == 0 or code in (502, 503, 504):
        return ProviderConnectionError.from_provider_error(
            provider="ollama",
            message=msg or "Could not reach Ollama server.",
            status_code=code or 502,
            original_error=orig,
        )
    if code >= 500:
        return ProviderServerError.from_provider_error(
            provider="ollama",
            message=msg,
            status_code=code,
            original_error=orig,
        )
    return ProviderError.from_provider_error(
        provider="ollama",
        message=msg,
        status_code=code or 500,
        original_error=orig,
    )
