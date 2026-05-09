"""Streaming Chat Completions — open stream with the same retry policy as non-streaming.

Separates **transport** (SDK call + ``call_with_retry``) from **adaptation** (chunk →
``StreamEvent``) in ``nb_groq.stream_adapter`` — single responsibility / dependency
direction: adapter depends on shared retry, not the other way around.
"""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq_groq._shared.retry import call_with_retry

logger = logging.getLogger(__name__)


def apply_stream_options(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *payload* with ``stream`` and ``stream_options`` set."""
    out = dict(payload)
    out["stream"] = True
    out["stream_options"] = {"include_usage": True}
    return out


async def open_streaming_completion(
    client: Any,
    payload: dict[str, Any],
    *,
    async_mode: bool,
    max_retries: int,
) -> Any:
    """Call ``chat.completions.create`` in streaming mode with ``call_with_retry``.

    Returns the SDK stream object (async iterator or sync iterable, depending on
    *async_mode*).
    """
    stream_payload = apply_stream_options(payload)

    if async_mode:

        async def api_call() -> Any:
            return await client.chat.completions.create(**stream_payload)

    else:

        def api_call() -> Any:
            return client.chat.completions.create(**stream_payload)

    return await call_with_retry(
        api_call,
        max_retries=max_retries,
        async_mode=async_mode,
        logger=logger,
    )
