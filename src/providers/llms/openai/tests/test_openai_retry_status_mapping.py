"""OpenAI ``call_with_retry`` maps SDK status errors to framework types (no spurious retry)."""

from __future__ import annotations

import logging

import httpx
import openai
import pytest
from nucleusiq.llms.errors import InvalidRequestError, ModelNotFoundError

from nucleusiq_openai._shared.retry import call_with_retry


def _req() -> httpx.Request:
    return httpx.Request("GET", "https://api.openai.com/v1/chat/completions")


@pytest.mark.asyncio
async def test_not_found_raises_model_not_found_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_: float) -> None:
        pytest.fail("should not sleep on 404")

    monkeypatch.setattr("nucleusiq_openai._shared.retry.asyncio.sleep", no_sleep)

    resp = httpx.Response(404, request=_req())
    err = openai.NotFoundError("missing", response=resp, body=None)

    async def api():
        raise err

    with pytest.raises(ModelNotFoundError) as ei:
        await call_with_retry(
            api,
            max_retries=3,
            async_mode=True,
            logger=logging.getLogger("t"),
        )
    assert ei.value.provider == "openai"
    assert ei.value.status_code == 404
    assert ei.value.original_error is err


@pytest.mark.asyncio
async def test_conflict_raises_invalid_request_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_: float) -> None:
        pytest.fail("should not sleep on 409")

    monkeypatch.setattr("nucleusiq_openai._shared.retry.asyncio.sleep", no_sleep)

    resp = httpx.Response(409, request=_req())
    err = openai.ConflictError("c", response=resp, body=None)

    async def api():
        raise err

    with pytest.raises(InvalidRequestError) as ei:
        await call_with_retry(
            api,
            max_retries=3,
            async_mode=True,
            logger=logging.getLogger("t"),
        )
    assert ei.value.provider == "openai"
    assert ei.value.status_code == 409
