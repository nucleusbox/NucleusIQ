"""Shared retry logic with exponential backoff for OpenAI API calls.

Both the Chat Completions and Responses API backends delegate their
retry/error-handling to ``call_with_retry`` so the behaviour is
identical and maintained in one place.

All errors are mapped to framework-level exceptions from
``nucleusiq.llms.errors`` so callers never need to import SDK types.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
import openai
from nucleusiq.llms.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    PermissionDeniedError,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
)
from nucleusiq.llms.retry_policy import (
    compute_rate_limit_sleep,
    extract_retry_after_header,
)

_PROVIDER = "openai"


def _openai_request_error_text(exc: BaseException) -> str:
    """Collect searchable text from an OpenAI ``APIStatusError`` / request error."""
    parts: list[str] = [str(exc)]
    msg = getattr(exc, "message", None)
    if isinstance(msg, str):
        parts.append(msg)
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        parts.append(code)
    typ = getattr(exc, "type", None)
    if isinstance(typ, str):
        parts.append(typ)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            for key in ("message", "code", "type", "param"):
                v = err.get(key)
                if isinstance(v, str):
                    parts.append(v)
        else:
            for key in ("message", "code", "type"):
                v = body.get(key)
                if isinstance(v, str):
                    parts.append(v)
    return " ".join(parts).lower()


def _is_openai_content_filter(text: str) -> bool:
    markers = (
        "content_filter",
        "content_policy",
        "content_policy_violation",
        "safety system",
        "safety_system",
        "output filters",
        "responsible_ai",
        "responsibleaipolicy",
    )
    return any(m in text for m in markers)


def _is_openai_context_length(text: str) -> bool:
    if "maximum context length" in text or "context_length" in text:
        return True
    markers = (
        "context window",
        "reduce the length",
        "reduce your prompt",
        "prompt is too long",
        "input is too long",
        "max_tokens",
        "token limit",
        "too many tokens",
        "exceeds the context",
    )
    if any(m in text for m in markers):
        return True
    return "token" in text


async def call_with_retry(
    api_call: Callable[[], Any],
    *,
    max_retries: int,
    async_mode: bool,
    logger: logging.Logger,
    on_bad_request: Callable[[], bool] | None = None,
) -> Any:
    """Execute *api_call* with retry and exponential backoff.

    Args:
        api_call: Zero-arg callable (sync or async) that performs the API
            request and returns the raw response.
        max_retries: Maximum number of retry attempts for transient errors.
        async_mode: ``True`` when the underlying client is ``AsyncOpenAI``.
        logger: Logger instance for warning/error messages.
        on_bad_request: Optional callback invoked on ``BadRequestError`` /
            ``UnprocessableEntityError``.  Should return ``True`` if the
            request was patched and should be retried, ``False`` to raise.

    Returns:
        The raw API response from *api_call*.

    Raises:
        AuthenticationError: Invalid API key (maps from ``openai.AuthenticationError``).
        PermissionDeniedError: Access denied (maps from ``openai.PermissionDeniedError``).
        InvalidRequestError: Bad request (``BadRequestError`` / ``UnprocessableEntityError``)
            or conflict (``ConflictError`` → HTTP 409).
        ModelNotFoundError: Missing model or resource (maps from ``openai.NotFoundError``).
        ContentFilterError: Content blocked by provider safety / content policy.
        ContextLengthError: Prompt exceeds model context or token limits.
        RateLimitError: Rate limit exceeded after max retries.
        ProviderServerError: API server error after max retries.
        ProviderConnectionError: Connection failure after max retries.
        ProviderError: Any other unexpected error.
    """
    attempt = 0
    while True:
        try:
            if async_mode:
                return await api_call()
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, api_call)

        except openai.RateLimitError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("Rate limit exceeded after %d retries: %s", max_retries, e)
                raise RateLimitError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Rate limit exceeded after {max_retries} retries: {e}",
                    status_code=429,
                    original_error=e,
                ) from e
            resp = getattr(e, "response", None)
            ra_hdr = extract_retry_after_header(resp)
            sleep_s, policy_meta = compute_rate_limit_sleep(attempt, ra_hdr)
            logger.warning(
                "OpenAI rate limit (%s); retry %d/%d; sleep=%.2fs; policy=%s",
                e,
                attempt,
                max_retries,
                sleep_s,
                policy_meta,
            )
            if async_mode:
                await asyncio.sleep(sleep_s)
            else:
                time.sleep(sleep_s)

        except openai.APIConnectionError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("Connection error after %d retries: %s", max_retries, e)
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
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except openai.AuthenticationError as e:
            logger.error("Authentication failed: %s", e)
            raise AuthenticationError.from_provider_error(
                provider=_PROVIDER,
                message=f"Invalid API key or authentication failed: {e}",
                status_code=401,
                original_error=e,
            ) from e

        except openai.PermissionDeniedError as e:
            logger.error("Permission denied: %s", e)
            raise PermissionDeniedError.from_provider_error(
                provider=_PROVIDER,
                message=f"Permission denied: {e}",
                status_code=403,
                original_error=e,
            ) from e

        except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
            if on_bad_request is not None and on_bad_request():
                continue
            detail = _openai_request_error_text(e)
            status = getattr(e, "status_code", None)
            if _is_openai_content_filter(detail):
                logger.error("Content blocked by provider filter: %s", e)
                raise ContentFilterError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Content blocked by provider safety or content policy: {e}",
                    status_code=status if isinstance(status, int) else 400,
                    original_error=e,
                ) from e
            if _is_openai_context_length(detail):
                logger.error("Context length exceeded: %s", e)
                raise ContextLengthError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Input exceeds model context or token limits: {e}",
                    status_code=status if isinstance(status, int) else 400,
                    original_error=e,
                ) from e
            logger.error("Invalid request: %s", e)
            raise InvalidRequestError.from_provider_error(
                provider=_PROVIDER,
                message=f"Invalid request parameters: {e}",
                status_code=status if isinstance(status, int) else 400,
                original_error=e,
            ) from e

        except openai.NotFoundError as e:
            logger.error("Resource not found: %s", e)
            status = getattr(e, "status_code", None)
            raise ModelNotFoundError.from_provider_error(
                provider=_PROVIDER,
                message=f"Model or resource not found: {e}",
                status_code=status if isinstance(status, int) else 404,
                original_error=e,
            ) from e

        except openai.ConflictError as e:
            logger.error("Request conflict: %s", e)
            status = getattr(e, "status_code", None)
            raise InvalidRequestError.from_provider_error(
                provider=_PROVIDER,
                message=f"Request conflict: {e}",
                status_code=status if isinstance(status, int) else 409,
                original_error=e,
            ) from e

        except openai.APIError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("API error after %d retries: %s", max_retries, e)
                raise ProviderServerError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"API error after {max_retries} retries: {e}",
                    status_code=getattr(e, "status_code", None),
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "API error (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except httpx.HTTPError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("HTTP error after %d retries: %s", max_retries, e)
                raise ProviderConnectionError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"HTTP error after {max_retries} retries: {e}",
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "HTTP error (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except Exception as e:
            logger.error("Unexpected error during OpenAI call: %s", e, exc_info=True)
            raise ProviderError.from_provider_error(
                provider=_PROVIDER,
                message=f"Unexpected OpenAI error: {e}",
                original_error=e,
            ) from e
