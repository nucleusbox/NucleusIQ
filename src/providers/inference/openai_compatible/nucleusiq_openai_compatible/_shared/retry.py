"""Retry and error mapping for OpenAI-compatible endpoints.

Mirrors ``nucleusiq_groq._shared.retry`` and reuses the shared
``nucleusiq.llms.retry_policy``, with three differences that matter for
self-hosted servers:

* **Connection errors are expected.** GPU nodes restart and models take
  minutes to load, so the backoff path is a normal operating mode rather
  than an anomaly. vLLM's "model is still loading" 503 is retryable.
* **Context overflow is recognized from vLLM/TGI phrasing**, not just
  OpenAI's, so it surfaces as ``ContextLengthError`` and the context engine
  can compact and retry instead of failing the run.
* **Every message is scrubbed** by a :class:`~.redact.Redactor` before it
  reaches a log or an exception, because gateways in front of self-hosted
  servers routinely echo request headers into error bodies.
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

from .redact import Redactor

__all__ = ["PROVIDER", "call_with_retry", "is_context_length_error"]

PROVIDER = "openai_compatible"

_CONTENT_FILTER_MARKERS = (
    "content_filter",
    "content_policy",
    "content_policy_violation",
    "safety system",
    "safety_system",
    "responsible_ai",
    "jailbreak",
)

_CONTEXT_MARKERS = (
    # OpenAI / generic
    "maximum context length",
    "context_length",
    "context window",
    "reduce the length",
    "prompt is too long",
    "input is too long",
    "token limit",
    "too many tokens",
    "exceeds the context",
    # vLLM
    "longer than the maximum model length",
    "maximum model length",
    "max_model_len",
    "please reduce the length of the messages",
    # TGI / llama.cpp
    "input validation error",
    "exceeds the maximum",
    "n_ctx",
)

_MODEL_LOADING_MARKERS = (
    "model is still loading",
    "currently loading",
    "not yet ready",
    "still initializing",
    "model not loaded",
    "loading model",
)


def _error_text(exc: BaseException) -> str:
    """Collect searchable lowercase text from an SDK error."""
    parts: list[str] = [str(exc)]
    message = getattr(exc, "message", None)
    if isinstance(message, str):
        parts.append(message)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        source = err if isinstance(err, dict) else body
        for key in ("message", "code", "type", "param", "detail"):
            value = source.get(key)
            if isinstance(value, str):
                parts.append(value)
    return " ".join(parts).lower()


def is_context_length_error(text: str) -> bool:
    """Whether *text* describes a context/token-budget overflow."""
    return any(marker in text for marker in _CONTEXT_MARKERS)


def _is_content_filter(text: str) -> bool:
    return any(marker in text for marker in _CONTENT_FILTER_MARKERS)


def _is_model_loading(text: str) -> bool:
    return any(marker in text for marker in _MODEL_LOADING_MARKERS)


async def _sleep(seconds: float, *, async_mode: bool) -> None:
    if async_mode:
        await asyncio.sleep(seconds)
    else:  # pragma: no cover - sync path exercised via async_mode=False tests
        time.sleep(seconds)


async def call_with_retry(
    api_call: Callable[[], Any],
    *,
    max_retries: int,
    async_mode: bool,
    logger: logging.Logger,
    redactor: Redactor | None = None,
) -> Any:
    """Execute *api_call* with retry, backoff and error translation.

    Args:
        api_call: Zero-argument callable performing the request.
        max_retries: Attempts after the first for retryable failures.
        async_mode: Whether *api_call* is a coroutine function.
        logger: Logger for retry and failure messages.
        redactor: Scrubs credentials from every emitted message.

    Raises:
        nucleusiq.llms.errors.LLMError: A mapped, credential-free error.
    """
    scrub = (redactor or Redactor()).scrub
    attempt = 0

    while True:
        try:
            if async_mode:
                return await api_call()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, api_call)

        except openai.RateLimitError as e:
            attempt += 1
            detail = scrub(str(e))
            if attempt > max_retries:
                logger.error("Rate limit after %d retries: %s", max_retries, detail)
                raise RateLimitError.from_provider_error(
                    provider=PROVIDER,
                    message=f"Rate limit exceeded after {max_retries} retries: {detail}",
                    status_code=429,
                    original_error=e,
                ) from e
            sleep_s, policy_meta = compute_rate_limit_sleep(
                attempt, extract_retry_after_header(getattr(e, "response", None))
            )
            logger.warning(
                "Rate limited (%s); retry %d/%d; sleep=%.2fs; policy=%s",
                detail,
                attempt,
                max_retries,
                sleep_s,
                policy_meta,
            )
            await _sleep(sleep_s, async_mode=async_mode)

        except openai.APIConnectionError as e:
            attempt += 1
            detail = scrub(str(e))
            if attempt > max_retries:
                logger.error(
                    "Connection error after %d retries: %s", max_retries, detail
                )
                raise ProviderConnectionError.from_provider_error(
                    provider=PROVIDER,
                    message=(
                        f"Could not reach the inference server after "
                        f"{max_retries} retries: {detail}. Check that the "
                        "server is running and base_url is correct."
                    ),
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "Connection error (%s); retry %d/%d in %ds",
                detail,
                attempt,
                max_retries,
                backoff,
            )
            await _sleep(backoff, async_mode=async_mode)

        except openai.AuthenticationError as e:
            # A caller problem, never transient — do not retry.
            detail = scrub(str(e))
            logger.error("Authentication failed: %s", detail)
            raise AuthenticationError.from_provider_error(
                provider=PROVIDER,
                message=(
                    f"Authentication rejected by the server: {detail}. The "
                    "server was started with a key requirement; pass "
                    "api_key=... (or auth=HeaderAuth(...) if it expects a "
                    "non-standard header)."
                ),
                status_code=401,
                original_error=e,
            ) from e

        except openai.PermissionDeniedError as e:
            detail = scrub(str(e))
            logger.error("Permission denied: %s", detail)
            raise PermissionDeniedError.from_provider_error(
                provider=PROVIDER,
                message=f"Permission denied: {detail}",
                status_code=403,
                original_error=e,
            ) from e

        except openai.NotFoundError as e:
            detail = scrub(str(e))
            logger.error("Not found: %s", detail)
            raise ModelNotFoundError.from_provider_error(
                provider=PROVIDER,
                message=(
                    f"Model or route not found: {detail}. Confirm model= "
                    "matches a served model name and that base_url ends "
                    "with /v1."
                ),
                status_code=404,
                original_error=e,
            ) from e

        except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
            text = _error_text(e)
            detail = scrub(str(e))
            status = getattr(e, "status_code", None)
            status = status if isinstance(status, int) else 400

            if _is_content_filter(text):
                logger.error("Content blocked: %s", detail)
                raise ContentFilterError.from_provider_error(
                    provider=PROVIDER,
                    message=f"Content blocked by a safety or content policy: {detail}",
                    status_code=status,
                    original_error=e,
                ) from e
            if is_context_length_error(text):
                logger.error("Context length exceeded: %s", detail)
                raise ContextLengthError.from_provider_error(
                    provider=PROVIDER,
                    message=(
                        f"Input exceeds the model's context length: {detail}. "
                        "If this is unexpected, the configured context_window "
                        "may be larger than the server's --max-model-len."
                    ),
                    status_code=status,
                    original_error=e,
                ) from e

            logger.error("Invalid request: %s", detail)
            raise InvalidRequestError.from_provider_error(
                provider=PROVIDER,
                message=(
                    f"Server rejected the request: {detail}. Generic "
                    "OpenAI-compatible servers reject unknown parameters; "
                    "check llm_params and extra_body."
                ),
                status_code=status,
                original_error=e,
            ) from e

        except openai.APIStatusError as e:
            status = getattr(e, "status_code", None)
            detail = scrub(str(e))
            text = _error_text(e)

            if isinstance(status, int) and (status >= 500 or _is_model_loading(text)):
                attempt += 1
                if attempt > max_retries:
                    logger.error(
                        "Server error after %d retries: %s", max_retries, detail
                    )
                    raise ProviderServerError.from_provider_error(
                        provider=PROVIDER,
                        message=f"Server error after {max_retries} retries: {detail}",
                        status_code=status,
                        original_error=e,
                    ) from e
                backoff = 2**attempt
                logger.warning(
                    "Server error %s (%s); retry %d/%d in %ds",
                    status,
                    detail,
                    attempt,
                    max_retries,
                    backoff,
                )
                await _sleep(backoff, async_mode=async_mode)
                continue

            logger.error("API status error: %s", detail)
            raise InvalidRequestError.from_provider_error(
                provider=PROVIDER,
                message=f"Inference server error: {detail}",
                status_code=status if isinstance(status, int) else 400,
                original_error=e,
            ) from e

        except httpx.HTTPError as e:
            attempt += 1
            detail = scrub(str(e))
            if attempt > max_retries:
                logger.error("HTTP error after %d retries: %s", max_retries, detail)
                raise ProviderConnectionError.from_provider_error(
                    provider=PROVIDER,
                    message=f"HTTP error after {max_retries} retries: {detail}",
                    original_error=e,
                ) from e
            backoff = 2**attempt
            logger.warning(
                "HTTP error (%s); retry %d/%d in %ds",
                detail,
                attempt,
                max_retries,
                backoff,
            )
            await _sleep(backoff, async_mode=async_mode)

        except Exception as e:
            detail = scrub(str(e))
            logger.error("Unexpected error: %s", detail, exc_info=True)
            raise ProviderError.from_provider_error(
                provider=PROVIDER,
                message=f"Unexpected error calling the inference server: {detail}",
                original_error=e,
            ) from e
