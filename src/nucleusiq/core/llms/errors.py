"""Framework-level LLM error taxonomy.

Every LLM provider (OpenAI, Gemini, Anthropic, ...) has its own SDK
exception types.  This module provides **provider-agnostic** exceptions
so that callers never need to import ``openai.RateLimitError`` or
``google.genai.errors.ClientError`` — they catch framework types instead.

Hierarchy::

    NucleusIQError
    └── LLMError
        ├── AuthenticationError      — invalid API key (401)
        ├── PermissionDeniedError    — access denied (403)
        ├── RateLimitError           — too many requests (429)
        ├── InvalidRequestError      — bad parameters (400)
        ├── ModelNotFoundError       — model does not exist (404)
        ├── ContentFilterError       — content blocked by safety filter
        ├── ContextLengthError       — input exceeds model context window
        ├── ProviderServerError      — provider 5xx error
        ├── ProviderConnectionError  — network / connection failure
        └── ProviderError            — catch-all for other provider errors

Usage in application code::

    from nucleusiq.llms.errors import RateLimitError, AuthenticationError

    try:
        result = await agent.execute("Hello")
    except RateLimitError as e:
        print(f"Hit rate limit on {e.provider}: {e}")
        # retry after backoff
    except AuthenticationError as e:
        print(f"Bad API key for {e.provider}")

Provider implementations use the ``from_provider_error`` classmethod::

    raise RateLimitError.from_provider_error(
        provider="openai",
        message="Rate limit exceeded",
        status_code=429,
        original_error=sdk_error,
    )
"""

from __future__ import annotations

from typing import Any

from nucleusiq.errors.base import NucleusIQError

__all__ = [
    "NucleusIQError",
    "LLMError",
    "AuthenticationError",
    "PermissionDeniedError",
    "RateLimitError",
    "InvalidRequestError",
    "ModelNotFoundError",
    "ContentFilterError",
    "ContextLengthError",
    "ProviderServerError",
    "ProviderConnectionError",
    "ProviderError",
]


class LLMError(NucleusIQError):
    """Base exception for all LLM-related errors.

    Attributes:
        provider: Name of the LLM provider (e.g. ``"openai"``, ``"gemini"``).
        status_code: HTTP status code if applicable, else ``None``.
        original_error: The underlying SDK exception, if available.
    """

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "unknown",
        status_code: int | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.original_error = original_error
        super().__init__(message)

    @classmethod
    def from_provider_error(
        cls,
        *,
        provider: str,
        message: str,
        status_code: int | None = None,
        original_error: BaseException | None = None,
        **kwargs: Any,
    ) -> LLMError:
        """Create an instance from a provider-specific error.

        This is the preferred factory for provider retry modules.
        """
        return cls(
            message,
            provider=provider,
            status_code=status_code,
            original_error=original_error,
        )

    def __repr__(self) -> str:
        parts = [f"{type(self).__name__}({self!s})"]
        if self.provider != "unknown":
            parts.append(f"provider={self.provider!r}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        return " ".join(parts)


class AuthenticationError(LLMError):
    """Invalid API key or authentication credentials (HTTP 401)."""


class PermissionDeniedError(LLMError):
    """Access denied — valid key but insufficient permissions (HTTP 403)."""


class RateLimitError(LLMError):
    """Too many requests — rate limit exceeded (HTTP 429).

    Callers should implement backoff/retry logic or reduce request frequency.
    """


class InvalidRequestError(LLMError):
    """Bad request parameters — the API rejected the request (HTTP 400).

    Common causes: unsupported model, invalid parameter combinations,
    content too long, etc.
    """


class ModelNotFoundError(LLMError):
    """The requested model does not exist or is not accessible (HTTP 404)."""


class ContentFilterError(LLMError):
    """Content was blocked by the provider's safety/content filter.

    The request or response triggered a content policy violation.
    """


class ProviderServerError(LLMError):
    """Provider-side server error (HTTP 5xx).

    The provider's servers encountered an internal error. These are
    typically transient and can be retried.
    """


class ProviderConnectionError(LLMError):
    """Network or connection failure when reaching the provider.

    DNS resolution, TCP connection, TLS handshake, or timeout failures.
    """


class ContextLengthError(LLMError):
    """Input exceeds the model's context window limit.

    Common when accumulated conversation history or tool output
    pushes the prompt beyond the model's token capacity.
    Providers typically report this as a 400 with a specific message.
    """


class ProviderError(LLMError):
    """Catch-all for provider errors that don't fit a specific category.

    Used when the HTTP status code or error type doesn't match any
    of the specific exception classes above.
    """
