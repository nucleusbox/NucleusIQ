"""Shared retry logic with exponential backoff for OpenAI API calls.

Both the Chat Completions and Responses API backends delegate their
retry/error-handling to ``call_with_retry`` so the behaviour is
identical and maintained in one place.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
import openai


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
                raise
            backoff = 2**attempt
            logger.warning(
                "Rate limit hit (%s); retry %d/%d in %ds",
                e,
                attempt,
                max_retries,
                backoff,
            )
            if async_mode:
                await asyncio.sleep(backoff)
            else:
                time.sleep(backoff)

        except openai.APIConnectionError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("Connection error after %d retries: %s", max_retries, e)
                raise
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
            raise ValueError(f"Invalid API key or authentication failed: {e}") from e

        except openai.PermissionDeniedError as e:
            logger.error("Permission denied: %s", e)
            raise ValueError(f"Permission denied: {e}") from e

        except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
            if on_bad_request is not None and on_bad_request():
                continue
            logger.error("Invalid request: %s", e)
            raise ValueError(f"Invalid request parameters: {e}") from e

        except openai.APIError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("API error after %d retries: %s", max_retries, e)
                raise
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
                raise
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
            raise
