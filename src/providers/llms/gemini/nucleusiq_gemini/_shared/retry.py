"""Shared retry logic with exponential backoff for Gemini API calls.

Mirrors the OpenAI provider's retry pattern using ``google.genai.errors``
for consistent error handling across NucleusIQ providers.

All errors are mapped to framework-level exceptions from
``nucleusiq.llms.errors`` so callers never need to import SDK types.

Error classification:
- **Rate limit (429)**: retry with backoff → ``RateLimitError``
- **Server error (5xx)**: retry with backoff → ``ProviderServerError``
- **Auth error (401)**: no retry → ``AuthenticationError``
- **Permission (403)**: no retry → ``PermissionDeniedError``
- **Bad request (400)**: no retry → ``InvalidRequestError``
- **Not found (404)**: no retry → ``ModelNotFoundError``
- **Connection error**: retry with backoff → ``ProviderConnectionError``
- **Unexpected**: log and raise as ``ProviderError``
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from nucleusiq.llms.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    PermissionDeniedError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
)

_PROVIDER = "gemini"


async def call_with_retry(
    api_call: Callable[[], Any],
    *,
    max_retries: int = 3,
    logger: logging.Logger,
    on_bad_request: Callable[[], bool] | None = None,
) -> Any:
    """Execute *api_call* with retry and exponential backoff.

    Args:
        api_call: Zero-arg callable that performs the API request.
        max_retries: Maximum retry attempts for transient errors.
        logger: Logger instance for warning/error messages.
        on_bad_request: Optional callback invoked on 400 errors.
            Return ``True`` to retry, ``False`` to raise.

    Returns:
        The raw API response.

    Raises:
        AuthenticationError: Invalid API key (401).
        PermissionDeniedError: Access denied (403).
        InvalidRequestError: Bad request (400).
        ModelNotFoundError: Model not found (404).
        RateLimitError: Rate limit exceeded (429) after max retries.
        ProviderServerError: Server error (5xx) after max retries.
        ProviderConnectionError: Network failure after max retries.
        ProviderError: Any other unexpected error.
    """
    from google.genai import errors as genai_errors

    attempt = 0
    while True:
        try:
            return api_call()

        except genai_errors.ClientError as e:
            code = getattr(e, "code", None) or _extract_status_code(e)

            if code == 429:
                attempt += 1
                if attempt > max_retries:
                    logger.error(
                        "Rate limit exceeded after %d retries: %s",
                        max_retries,
                        e,
                    )
                    raise RateLimitError.from_provider_error(
                        provider=_PROVIDER,
                        message=f"Rate limit exceeded after {max_retries} retries: {e}",
                        status_code=429,
                        original_error=e,
                    ) from e
                backoff = 2**attempt
                logger.warning(
                    "Rate limit hit (429); retry %d/%d in %ds",
                    attempt,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            if code == 401:
                logger.error("Authentication failed: %s", e)
                raise AuthenticationError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Invalid Gemini API key or authentication failed: {e}",
                    status_code=401,
                    original_error=e,
                ) from e

            if code == 403:
                logger.error("Permission denied: %s", e)
                raise PermissionDeniedError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Permission denied: {e}",
                    status_code=403,
                    original_error=e,
                ) from e

            if code == 400:
                if on_bad_request is not None and on_bad_request():
                    continue
                logger.error("Invalid request: %s", e)
                raise InvalidRequestError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Invalid request parameters: {e}",
                    status_code=400,
                    original_error=e,
                ) from e

            if code == 404:
                logger.error("Model not found: %s", e)
                raise ModelNotFoundError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Model not found: {e}",
                    status_code=404,
                    original_error=e,
                ) from e

            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Client error (HTTP %s) after %d retries: %s",
                    code,
                    max_retries,
                    e,
                )
                raise ProviderError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Client error (HTTP {code}): {e}",
                    status_code=code,
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "Client error (HTTP %s); retry %d/%d in %ds",
                code,
                attempt,
                max_retries,
                backoff,
            )
            await asyncio.sleep(backoff)

        except genai_errors.ServerError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Server error after %d retries: %s",
                    max_retries,
                    e,
                )
                raise ProviderServerError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Server error after {max_retries} retries: {e}",
                    status_code=getattr(e, "code", 500),
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "Server error (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            await asyncio.sleep(backoff)

        except (ConnectionError, OSError) as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Connection error after %d retries: %s",
                    max_retries,
                    e,
                )
                raise ProviderConnectionError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Connection error after {max_retries} retries: {e}",
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "Connection error (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            await asyncio.sleep(backoff)

        except Exception as e:
            logger.error(
                "Unexpected error during Gemini call: %s",
                e,
                exc_info=True,
            )
            raise ProviderError.from_provider_error(
                provider=_PROVIDER,
                message=f"Unexpected Gemini error: {e}",
                original_error=e,
            ) from e


def _extract_status_code(error: Exception) -> int | None:
    """Try to extract HTTP status code from an error's response attribute."""
    response = getattr(error, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            return int(status)
    return None
