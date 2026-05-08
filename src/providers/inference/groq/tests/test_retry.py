"""Tests for retry and Groq→Nucleus error mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

import groq
import httpx
import pytest
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
from nucleusiq_groq._shared import retry as retry_mod
from nucleusiq_groq._shared.retry import call_with_retry


def _resp(status: int = 400) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://api.groq.com"))


@pytest.mark.asyncio
async def test_call_with_retry_success_async() -> None:
    mock = MagicMock(return_value="ok")

    async def api():
        return mock()

    out = await call_with_retry(
        api, max_retries=2, async_mode=True, logger=retry_mod.logging.getLogger("t")
    )
    assert out == "ok"
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_call_with_retry_rate_limit_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("nucleusiq_groq._shared.retry.asyncio.sleep", fake_sleep)

    n = {"i": 0}

    async def api():
        n["i"] += 1
        if n["i"] < 2:
            raise groq.RateLimitError("rl", response=_resp(429), body=None)
        return "done"

    out = await call_with_retry(
        api, max_retries=3, async_mode=True, logger=retry_mod.logging.getLogger("t")
    )
    assert out == "done"
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_call_with_retry_rate_limit_exhausted() -> None:
    async def api():
        raise groq.RateLimitError("rl", response=_resp(429), body=None)

    with pytest.raises(RateLimitError):
        await call_with_retry(
            api, max_retries=1, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_authentication() -> None:
    async def api():
        raise groq.AuthenticationError("bad", response=_resp(401), body=None)

    with pytest.raises(AuthenticationError):
        await call_with_retry(
            api, max_retries=2, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_permission_denied() -> None:
    async def api():
        raise groq.PermissionDeniedError("no", response=_resp(403), body=None)

    with pytest.raises(PermissionDeniedError):
        await call_with_retry(
            api, max_retries=2, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_content_filter() -> None:
    async def api():
        body = {"error": {"message": "content_policy_violation"}}
        raise groq.BadRequestError("br", response=_resp(400), body=body)

    with pytest.raises(ContentFilterError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_context_length() -> None:
    async def api():
        raise groq.BadRequestError(
            "too long",
            response=_resp(400),
            body={"error": {"message": "prompt is too long"}},
        )

    with pytest.raises(ContextLengthError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_invalid_request() -> None:
    async def api():
        raise groq.BadRequestError(
            "bad", response=_resp(400), body={"error": {"message": "unknown"}}
        )

    with pytest.raises(InvalidRequestError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_bad_request_on_retry_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("nucleusiq_groq._shared.retry.asyncio.sleep", fake_sleep)
    n = {"i": 0}

    async def api():
        n["i"] += 1
        raise groq.BadRequestError(
            "br", response=_resp(400), body={"error": {"message": "x"}}
        )

    def on_bad():
        return n["i"] < 2

    with pytest.raises(InvalidRequestError):
        await call_with_retry(
            api,
            max_retries=2,
            async_mode=True,
            logger=retry_mod.logging.getLogger("t"),
            on_bad_request=on_bad,
        )


@pytest.mark.asyncio
async def test_call_with_retry_not_found() -> None:
    async def api():
        raise groq.NotFoundError("nf", response=_resp(404), body=None)

    with pytest.raises(ModelNotFoundError):
        await call_with_retry(
            api, max_retries=2, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_other_4xx_status_maps_invalid_request() -> None:
    async def api():
        raise groq.ConflictError("co", response=_resp(409), body=None)

    with pytest.raises(InvalidRequestError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_server_error_exhausted() -> None:
    async def api():
        raise groq.InternalServerError("srv", response=_resp(500), body=None)

    with pytest.raises(ProviderServerError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("nucleusiq_groq._shared.retry.asyncio.sleep", fake_sleep)

    async def api():
        raise groq.APIConnectionError(
            message="down", request=httpx.Request("GET", "http://x")
        )

    with pytest.raises(ProviderConnectionError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_httpx_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("nucleusiq_groq._shared.retry.asyncio.sleep", fake_sleep)

    async def api():
        raise httpx.ConnectError("nope", request=httpx.Request("GET", "http://x"))

    with pytest.raises(ProviderConnectionError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_unexpected_exception() -> None:
    async def api():
        raise ValueError("weird")

    with pytest.raises(ProviderError):
        await call_with_retry(
            api, max_retries=0, async_mode=True, logger=retry_mod.logging.getLogger("t")
        )


@pytest.mark.asyncio
async def test_call_with_retry_sync_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nucleusiq_groq._shared.retry.time.sleep", lambda s: None)

    def api():
        raise groq.RateLimitError("rl", response=_resp(429), body=None)

    with pytest.raises(RateLimitError):
        await call_with_retry(
            api,
            max_retries=0,
            async_mode=False,
            logger=retry_mod.logging.getLogger("t"),
        )
