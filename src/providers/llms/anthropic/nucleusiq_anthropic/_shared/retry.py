"""Retry and error mapping for the official ``anthropic`` Python SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
from anthropic._exceptions import (
    APIConnectionError as SdkAPIConnectionError,
)
from anthropic._exceptions import (
    APIStatusError as SdkAPIStatusError,
)
from anthropic._exceptions import (
    APITimeoutError as SdkAPITimeoutError,
)
from anthropic._exceptions import (
    AuthenticationError as SdkAuthenticationError,
)
from anthropic._exceptions import (
    BadRequestError as SdkBadRequestError,
)
from anthropic._exceptions import (
    InternalServerError as SdkInternalServerError,
)
from anthropic._exceptions import (
    NotFoundError as SdkNotFoundError,
)
from anthropic._exceptions import (
    OverloadedError as SdkOverloadedError,
)
from anthropic._exceptions import (
    PermissionDeniedError as SdkPermissionDeniedError,
)
from anthropic._exceptions import (
    RateLimitError as SdkRateLimitError,
)
from anthropic._exceptions import (
    RequestTooLargeError as SdkRequestTooLargeError,
)
from anthropic._exceptions import (
    ServiceUnavailableError as SdkServiceUnavailableError,
)
from anthropic._exceptions import (
    UnprocessableEntityError as SdkUnprocessableEntityError,
)
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
from nucleusiq.llms.retry_policy import (
    compute_rate_limit_sleep,
    extract_retry_after_header,
)

_PROVIDER = "anthropic"


def _request_error_text(exc: BaseException) -> str:
    parts: list[str] = [str(exc)]
    msg = getattr(exc, "message", None)
    if isinstance(msg, str):
        parts.append(msg)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            for key in ("message", "type"):
                v = err.get(key)
                if isinstance(v, str):
                    parts.append(v)
        else:
            for key in ("message", "type"):
                v = body.get(key)
                if isinstance(v, str):
                    parts.append(v)

    return " ".join(parts).lower()


def _is_content_filter(text: str) -> bool:
    markers = (
        "content_filter",
        "content_policy",
        "harmful",
        "policy",
        "blocked",
        "safety",
        "refusal",
    )
    return any(m in text for m in markers)


def _is_context_length(text: str) -> bool:
    if "context_length" in text or "maximum context" in text:
        return True
    markers = (
        "prompt is too long",
        "too many tokens",
        "exceeds the context",
        "context window",
        "input is too long",
        "token limit",
    )
    return any(m in text for m in markers)


def _is_request_too_large(text: str) -> bool:
    return "413" in text or "too_large" in text or "request too large" in text


async def call_with_retry(
    api_call: Callable[[], Any],
    *,
    max_retries: int,
    async_mode: bool,
    logger: logging.Logger,
    on_bad_request: Callable[[], bool] | None = None,
) -> Any:
    """Execute *api_call* with backoff (mirrors Groq/OpenAI provider modules)."""

    attempt = 0
    while True:
        try:
            if async_mode:
                return await api_call()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, api_call)

        except SdkRateLimitError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Anthropic rate limit after %d retries: %s", max_retries, e
                )
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
                "Anthropic rate limit (%s); retry %d/%d; sleep=%.2fs; policy=%s",
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

        except (SdkAPIConnectionError, SdkAPITimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Anthropic connection error after %d retries: %s",
                    max_retries,
                    e,
                )
                raise ProviderConnectionError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Connection error after {max_retries} retries: {e}",
                    original_error=e,
                ) from e
            backoff = min(60, 2**attempt)
            logger.warning(
                "Anthropic connection error (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except SdkAuthenticationError as e:
            logger.error("Anthropic authentication failed: %s", e)
            raise AuthenticationError.from_provider_error(
                provider=_PROVIDER,
                message=f"Invalid API key or authentication failed: {e}",
                status_code=401,
                original_error=e,
            ) from e

        except SdkPermissionDeniedError as e:
            logger.error("Anthropic permission denied: %s", e)
            raise PermissionDeniedError.from_provider_error(
                provider=_PROVIDER,
                message=f"Permission denied: {e}",
                status_code=403,
                original_error=e,
            ) from e

        except SdkNotFoundError as e:
            logger.error("Anthropic model/resource not found: %s", e)
            raise ModelNotFoundError.from_provider_error(
                provider=_PROVIDER,
                message=f"Model or resource not found: {e}",
                status_code=404,
                original_error=e,
            ) from e

        except (SdkBadRequestError, SdkUnprocessableEntityError) as e:
            if on_bad_request is not None and on_bad_request():
                continue
            detail = _request_error_text(e)
            status = getattr(e, "status_code", None)
            if getattr(e, "type", "") == "request_too_large" or _is_request_too_large(
                detail
            ):
                raise InvalidRequestError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Request exceeds provider size limits: {e}",
                    status_code=413,
                    original_error=e,
                ) from e
            if _is_content_filter(detail):
                raise ContentFilterError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Content blocked by provider safety policy: {e}",
                    status_code=status if isinstance(status, int) else 400,
                    original_error=e,
                ) from e
            if _is_context_length(detail):
                raise ContextLengthError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Input exceeds context or token limits: {e}",
                    status_code=status if isinstance(status, int) else 400,
                    original_error=e,
                ) from e
            logger.error("Anthropic invalid request: %s", e)
            raise InvalidRequestError.from_provider_error(
                provider=_PROVIDER,
                message=f"Invalid request parameters: {e}",
                status_code=status if isinstance(status, int) else 400,
                original_error=e,
            ) from e

        except SdkRequestTooLargeError as e:
            logger.error("Anthropic request too large: %s", e)
            raise InvalidRequestError.from_provider_error(
                provider=_PROVIDER,
                message=f"Request exceeds Anthropic payload limits: {e}",
                status_code=413,
                original_error=e,
            ) from e

        except (
            SdkInternalServerError,
            SdkOverloadedError,
            SdkServiceUnavailableError,
        ) as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Anthropic server error after %d retries: %s", max_retries, e
                )
                status = getattr(e, "status_code", None)
                raise ProviderServerError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"Anthropic returned a server/overload error: {e}",
                    status_code=status if isinstance(status, int) else 503,
                    original_error=e,
                ) from e
            backoff = min(120, 2**attempt + 1)
            logger.warning(
                "Anthropic server/overload (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except SdkAPIStatusError as e:
            status = getattr(e, "status_code", None)
            if isinstance(status, int) and status >= 500:
                attempt += 1
                if attempt > max_retries:
                    raise ProviderServerError.from_provider_error(
                        provider=_PROVIDER,
                        message=f"Anthropic HTTP {status}: {e}",
                        status_code=status,
                        original_error=e,
                    ) from e
                backoff = 2 ** min(attempt, 6)
                if async_mode:
                    await asyncio.sleep(backoff)
                else:
                    time.sleep(backoff)
                continue
            raise InvalidRequestError.from_provider_error(
                provider=_PROVIDER,
                message=f"Anthropic API error: {e}",
                status_code=status if isinstance(status, int) else 400,
                original_error=e,
            ) from e

        except httpx.HTTPError as e:
            attempt += 1
            if attempt > max_retries:
                raise ProviderConnectionError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"HTTP transport error after {max_retries} retries: {e}",
                    original_error=e,
                ) from e
            backoff = 2**attempt
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except Exception as e:
            logger.error("Unexpected Anthropic error: %s", e, exc_info=True)
            raise ProviderError.from_provider_error(
                provider=_PROVIDER,
                message=f"Unexpected Anthropic error: {e}",
                original_error=e,
            ) from e
