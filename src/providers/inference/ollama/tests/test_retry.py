"""Retry mapping for Ollama."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from nucleusiq.llms.errors import ModelNotFoundError
from nucleusiq_ollama._shared.errors import map_ollama_response_error
from nucleusiq_ollama._shared.retry import call_with_retry
from ollama import ResponseError


def test_map_ollama_response_error_404() -> None:
    exc = ResponseError("missing", status_code=404)
    out = map_ollama_response_error(exc)
    assert isinstance(out, ModelNotFoundError)


@pytest.mark.asyncio
async def test_call_with_retry_raises_after_non_retryable() -> None:
    logger = MagicMock()
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise ResponseError("bad", status_code=400)

    with pytest.raises(Exception):
        await call_with_retry(boom, max_retries=2, async_mode=True, logger=logger)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_call_with_retry_connection_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    calls = {"n": 0}

    async def immediate_sleep(_):
        return None

    monkeypatch.setattr(
        "nucleusiq_ollama._shared.retry.asyncio.sleep",
        immediate_sleep,
    )

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("down")
        return "ok"

    out = await call_with_retry(flaky, max_retries=3, async_mode=True, logger=logger)
    assert out == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_call_with_retry_retries_429(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    n = {"c": 0}

    async def immediate_sleep(_):
        return None

    monkeypatch.setattr(
        "nucleusiq_ollama._shared.retry.asyncio.sleep",
        immediate_sleep,
    )

    async def flaky():
        n["c"] += 1
        if n["c"] < 2:
            raise ResponseError("rl", status_code=429)
        return "win"

    out = await call_with_retry(flaky, max_retries=3, async_mode=True, logger=logger)
    assert out == "win"
    assert n["c"] == 2


@pytest.mark.asyncio
async def test_call_with_retry_retries_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    logger = MagicMock()
    n = {"c": 0}

    async def immediate_sleep(_):
        return None

    monkeypatch.setattr(
        "nucleusiq_ollama._shared.retry.asyncio.sleep",
        immediate_sleep,
    )

    async def flaky():
        n["c"] += 1
        if n["c"] < 2:
            raise httpx.ReadTimeout("t")
        return "ok"

    out = await call_with_retry(flaky, max_retries=3, async_mode=True, logger=logger)
    assert out == "ok"


@pytest.mark.asyncio
async def test_call_with_retry_sync_callable() -> None:
    logger = MagicMock()

    def sync_ok():
        return 42

    out = await call_with_retry(sync_ok, max_retries=1, async_mode=False, logger=logger)
    assert out == 42
