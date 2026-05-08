"""Retry and error mapping for Groq using the official ``groq`` Python SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import groq
import httpx
from nucleusiq.llms.errors import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    ModelNotFoundError,
    PermissionDeniedError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
)

_PROVIDER = "groq"


def _groq_request_error_text(exc: BaseException) -> str:
    """Collect searchable text from a Groq ``APIStatusError`` / API error."""
    parts: list[str] = [str(exc)]
    msg = getattr(exc, "message", None)
    if isinstance(msg, str):
        parts.append(msg)
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


def _is_content_filter(text: str) -> bool:
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


def _is_context_length(text: str) -> bool:
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
    """Execute *api_call* with retry and exponential backoff."""
    attempt = 0
    while True:
        try:
            if async_mode:
                return await api_call()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, api_call)

        except groq.RateLimitError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("Rate limit exceeded after %d retries: %s", max_retries, e)
                raise RateLimitError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Rate limit exceeded after {max_retries} retries: {e}",
                    status_code=429,
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "Groq rate limit (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except groq.APIConnectionError as e:
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
                "Groq connection error (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except groq.AuthenticationError as e:
            logger.error("Groq authentication failed: %s", e)
            raise AuthenticationError.from_provider_error(
                provider=_PROVIDER,
                message=f"Invalid API key or authentication failed: {e}",
                status_code=401,
                original_error=e,
            ) from e

        except groq.PermissionDeniedError as e:
            logger.error("Groq permission denied: %s", e)
            raise PermissionDeniedError.from_provider_error(
                provider=_PROVIDER,
                message=f"Permission denied: {e}",
                status_code=403,
                original_error=e,
            ) from e

        except groq.NotFoundError as e:
            logger.error("Groq model not found: %s", e)
            raise ModelNotFoundError.from_provider_error(
                provider=_PROVIDER,
                message=f"Model not found: {e}",
                status_code=404,
                original_error=e,
            ) from e

        except (groq.BadRequestError, groq.UnprocessableEntityError) as e:
            if on_bad_request is not None and on_bad_request():
                continue
            detail = _groq_request_error_text(e)
            status = getattr(e, "status_code", None)
            if _is_content_filter(detail):
                logger.error("Groq content blocked: %s", e)
                raise ContentFilterError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Content blocked by provider safety or content policy: {e}",
                    status_code=status if isinstance(status, int) else 400,
                    original_error=e,
                ) from e
            if _is_context_length(detail):
                logger.error("Groq context length: %s", e)
                raise ContextLengthError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Input exceeds model context or token limits: {e}",
                    status_code=status if isinstance(status, int) else 400,
                    original_error=e,
                ) from e
            logger.error("Groq invalid request: %s", e)
            raise InvalidRequestError.from_provider_error(
                provider=_PROVIDER,
                message=f"Invalid request parameters: {e}",
                status_code=status if isinstance(status, int) else 400,
                original_error=e,
            ) from e

        except groq.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if isinstance(status, int) and status >= 500:
                attempt += 1
                if attempt > max_retries:
                    logger.error("Groq API error after %d retries: %s", max_retries, e)
                    raise ProviderServerError.from_provider_error(
                        provider=_PROVIDER,
                        message=f"API error after {max_retries} retries: {e}",
                        status_code=status,
                        original_error=e,
                    ) from e
                backoff = 2**attempt
                logger.warning(
                    "Groq API error (%s); retry %d/%d in %ds",
                    e,
                    attempt,
                    max_retries,
                    backoff,
                )
                if async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
                continue
            logger.error("Groq API status error: %s", e)
            raise InvalidRequestError.from_provider_error(
                provider=_PROVIDER,
                message=f"Groq API error: {e}",
                status_code=status if isinstance(status, int) else 400,
                original_error=e,
            ) from e

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
            logger.error("Unexpected error during Groq call: %s", e, exc_info=True)
            raise ProviderError.from_provider_error(
                provider=_PROVIDER,
                message=f"Unexpected Groq error: {e}",
                original_error=e,
            ) from e
