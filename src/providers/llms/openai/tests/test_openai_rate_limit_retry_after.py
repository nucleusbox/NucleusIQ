"""429 + Retry-After behavior for OpenAI ``call_with_retry`` (shared retry_policy)."""

from __future__ import annotations

import logging

import httpx
import openai
import pytest
from nucleusiq_openai._shared.retry import call_with_retry


def _resp_429(retry_after: str) -> httpx.Response:
    return httpx.Response(
        429,
        request=httpx.Request("GET", "https://api.openai.com/v1/chat/completions"),
        headers={"retry-after": retry_after},
    )


@pytest.mark.asyncio
async def test_rate_limit_uses_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def capture_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr(
        "nucleusiq_openai._shared.retry.asyncio.sleep",
        capture_sleep,
    )

    n = {"i": 0}

    async def api():
        n["i"] += 1
        if n["i"] < 2:
            raise openai.RateLimitError("rl", response=_resp_429("28"), body=None)
        return "ok"

    out = await call_with_retry(
        api,
        max_retries=3,
        async_mode=True,
        logger=logging.getLogger("openai_retry_test"),
    )
    assert out == "ok"
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(28.0)


@pytest.mark.asyncio
async def test_rate_limit_fallback_exponential_without_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def capture_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr(
        "nucleusiq_openai._shared.retry.asyncio.sleep",
        capture_sleep,
    )

    n = {"i": 0}

    async def api():
        n["i"] += 1
        if n["i"] < 2:
            raise openai.RateLimitError(
                "rl",
                response=httpx.Response(
                    429,
                    request=httpx.Request("GET", "https://api.openai.com/v1/x"),
                ),
                body=None,
            )
        return "done"

    await call_with_retry(
        api,
        max_retries=3,
        async_mode=True,
        logger=logging.getLogger("openai_retry_test"),
    )
    assert sleeps[0] == pytest.approx(2.0)
