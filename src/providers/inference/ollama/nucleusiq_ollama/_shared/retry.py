"""Retry helpers for the official ``ollama`` Python client."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
from nucleusiq.llms.errors import ProviderConnectionError, ProviderError
from ollama import ResponseError

from nucleusiq_ollama._shared.errors import map_ollama_response_error

_PROVIDER = "ollama"

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


async def call_with_retry(
    api_call: Callable[[], Any],
    *,
    max_retries: int,
    async_mode: bool,
    logger: logging.Logger,
) -> Any:
    """Run *api_call* with limited retries on rate limits and transient failures."""
    attempt = 0
    while True:
        try:
            if async_mode:
                return await api_call()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, api_call)

        except ResponseError as e:
            code = getattr(e, "status_code", None) or 0
            attempt += 1
            if code in _RETRYABLE_STATUS and attempt <= max_retries:
                backoff = min(2**attempt, 60.0)
                logger.warning(
                    "Ollama response error (%s); retry %d/%d in %.1fs",
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
            raise map_ollama_response_error(e) from e

        except ConnectionError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Ollama connection failed after %d retries: %s", max_retries, e
                )
                raise ProviderConnectionError.from_provider_error(
                    provider=_PROVIDER,
                    message=str(e) or "Could not connect to Ollama.",
                    original_error=e,
                ) from e
            backoff = min(2**attempt, 60.0)
            logger.warning(
                "Ollama connection error (%s); retry %d/%d in %.1fs",
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

        except httpx.HTTPError as e:
            attempt += 1
            if attempt > max_retries:
                logger.error("HTTP error after %d retries: %s", max_retries, e)
                raise ProviderConnectionError.from_provider_error(
                    provider=_PROVIDER,
                    message=f"HTTP error after {max_retries} retries: {e}",
                    original_error=e,
                ) from e
            backoff = min(2**attempt, 60.0)
            logger.warning(
                "HTTP error (%s); retry %d/%d in %.1fs",
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

        except Exception as e:
            logger.error("Unexpected error during Ollama call: %s", e, exc_info=True)
            raise ProviderError.from_provider_error(
                provider=_PROVIDER,
                message=f"Unexpected Ollama error: {e}",
                original_error=e,
            ) from e
