"""Unit tests for :mod:`nucleusiq_anthropic._shared.retry`."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
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
    AuthenticationError as NucleusAuthError,
)
from nucleusiq.llms.errors import (
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
)
from nucleusiq.llms.errors import (
    PermissionDeniedError as NucleusPermError,
)
from nucleusiq.llms.errors import (
    RateLimitError as NucleusRateLimitError,
)
from nucleusiq_anthropic._shared import retry as retry_mod
from nucleusiq_anthropic._shared.retry import call_with_retry


def anthropic_dummy_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def anthropic_dummy_response(
    status: int, *, request: httpx.Request | None = None
) -> httpx.Response:
    req = request or anthropic_dummy_request()
    return httpx.Response(status, request=req)


def _exc(
    cls: type[SdkAPIStatusError],
    status: int,
    *,
    msg: str = "err",
    body: object | None = None,
) -> SdkAPIStatusError:
    req = anthropic_dummy_request()
    resp = anthropic_dummy_response(status, request=req)
    return cls(message=msg, response=resp, body=body)


@pytest.mark.asyncio
async def test_rate_limit_then_success_async() -> None:
    logger = __import__("logging").getLogger("t")

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _exc(
                SdkRateLimitError,
                429,
                body={"error": {"message": "slow down"}},
            )
        return "ok"

    with patch.object(retry_mod.asyncio, "sleep", new_callable=AsyncMock):
        out = await call_with_retry(
            flaky, max_retries=3, async_mode=True, logger=logger
        )
    assert out == "ok"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_rate_limit_exhaustion_async() -> None:
    logger = __import__("logging").getLogger("t")

    async def always_rl() -> None:
        raise _exc(SdkRateLimitError, 429)

    with (
        patch.object(retry_mod.asyncio, "sleep", new_callable=AsyncMock),
        pytest.raises(NucleusRateLimitError),
    ):
        await call_with_retry(always_rl, max_retries=2, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_rate_limit_then_success_sync() -> None:
    logger = __import__("logging").getLogger("t")

    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _exc(SdkRateLimitError, 429)
        return "done"

    with patch.object(retry_mod.time, "sleep"):
        out = await call_with_retry(
            flaky, max_retries=3, async_mode=False, logger=logger
        )
    assert out == "done"


@pytest.mark.asyncio
async def test_connection_retry_then_raise() -> None:
    logger = __import__("logging").getLogger("t")
    req = anthropic_dummy_request()

    attempts = {"n": 0}

    def flaky() -> None:
        attempts["n"] += 1
        raise SdkAPIConnectionError(request=req)

    with patch.object(retry_mod.time, "sleep"), pytest.raises(ProviderConnectionError):
        await call_with_retry(flaky, max_retries=1, async_mode=False, logger=logger)
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_timeout_retry_async() -> None:
    logger = __import__("logging").getLogger("t")
    req = anthropic_dummy_request()

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise SdkAPITimeoutError(request=req)
        return "alive"

    with patch.object(retry_mod.asyncio, "sleep", new_callable=AsyncMock):
        out = await call_with_retry(
            flaky, max_retries=3, async_mode=True, logger=logger
        )
    assert out == "alive"


@pytest.mark.asyncio
async def test_auth_raises_immediately() -> None:
    logger = __import__("logging").getLogger("t")

    async def boom() -> None:
        raise _exc(SdkAuthenticationError, 401)

    with pytest.raises(NucleusAuthError):
        await call_with_retry(boom, max_retries=2, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_permission_and_not_found() -> None:
    logger = __import__("logging").getLogger("t")

    async def perm() -> None:
        raise _exc(SdkPermissionDeniedError, 403)

    with pytest.raises(NucleusPermError):
        await call_with_retry(perm, max_retries=1, async_mode=True, logger=logger)

    async def nf() -> None:
        raise _exc(SdkNotFoundError, 404)

    with pytest.raises(ModelNotFoundError):
        await call_with_retry(nf, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.parametrize(
    ("markers", "expected"),
    [
        ("context window exceeded", ContextLengthError),
        ("blocked by safety content_policy", ContentFilterError),
        ("prompt is invalid", InvalidRequestError),
    ],
)
@pytest.mark.asyncio
async def test_bad_request_mappings(markers: str, expected: type[Exception]) -> None:
    logger = __import__("logging").getLogger("t")

    async def bad() -> None:
        raise SdkBadRequestError(
            message="nope",
            response=anthropic_dummy_response(400),
            body={"error": {"message": markers}},
        )

    with pytest.raises(expected):
        await call_with_retry(bad, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_bad_request_too_large_by_text() -> None:
    logger = __import__("logging").getLogger("t")

    async def bad() -> None:
        raise SdkBadRequestError(
            message="413",
            response=anthropic_dummy_response(400),
            body={"error": {"message": "request too large"}},
        )

    with pytest.raises(InvalidRequestError) as ei:
        await call_with_retry(bad, max_retries=1, async_mode=True, logger=logger)
    assert ei.value.status_code == 413


@pytest.mark.asyncio
async def test_bad_request_on_bad_request_continue() -> None:
    logger = __import__("logging").getLogger("t")
    called = {"n": 0}

    def on_bad() -> bool:
        called["n"] += 1
        return called["n"] == 1

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise SdkBadRequestError(
                message="retry me",
                response=anthropic_dummy_response(400),
                body=None,
            )
        return "fixed"

    out = await call_with_retry(
        flaky,
        max_retries=2,
        async_mode=True,
        logger=logger,
        on_bad_request=on_bad,
    )
    assert out == "fixed"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_request_too_large_exception() -> None:
    logger = __import__("logging").getLogger("t")

    async def big() -> None:
        raise SdkRequestTooLargeError(
            message="too big",
            response=anthropic_dummy_response(413),
            body=None,
        )

    with pytest.raises(InvalidRequestError) as ei:
        await call_with_retry(big, max_retries=1, async_mode=True, logger=logger)
    assert ei.value.status_code == 413


@pytest.mark.asyncio
async def test_server_error_retry_then_success() -> None:
    logger = __import__("logging").getLogger("t")

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _exc(SdkInternalServerError, 500)
        return "ok"

    with patch.object(retry_mod.asyncio, "sleep", new_callable=AsyncMock):
        out = await call_with_retry(
            flaky, max_retries=3, async_mode=True, logger=logger
        )
    assert out == "ok"


@pytest.mark.asyncio
async def test_overloaded_exhaustion() -> None:
    logger = __import__("logging").getLogger("t")

    async def down() -> None:
        raise _exc(SdkOverloadedError, 529)

    with (
        patch.object(retry_mod.asyncio, "sleep", new_callable=AsyncMock),
        pytest.raises(ProviderServerError),
    ):
        await call_with_retry(down, max_retries=2, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_service_unavailable_sync_retry() -> None:
    logger = __import__("logging").getLogger("t")

    attempts = {"n": 0}

    def flaky() -> int:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _exc(SdkServiceUnavailableError, 503)
        return 1

    with patch.object(retry_mod.time, "sleep"):
        out = await call_with_retry(
            flaky, max_retries=3, async_mode=False, logger=logger
        )
    assert out == 1


@pytest.mark.asyncio
async def test_api_status_500_retry_then_fail() -> None:
    logger = __import__("logging").getLogger("t")

    async def flaky() -> None:
        raise SdkAPIStatusError(
            message="bad gateway style",
            response=anthropic_dummy_response(502),
            body=None,
        )

    with (
        patch.object(retry_mod.asyncio, "sleep", new_callable=AsyncMock),
        pytest.raises(ProviderServerError),
    ):
        await call_with_retry(flaky, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_api_status_non_500_maps_invalid() -> None:
    logger = __import__("logging").getLogger("t")

    async def flaky() -> None:
        raise SdkAPIStatusError(
            message="teapot",
            response=anthropic_dummy_response(418),
            body=None,
        )

    with pytest.raises(InvalidRequestError):
        await call_with_retry(flaky, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_httpx_http_error_retry() -> None:
    logger = __import__("logging").getLogger("t")

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.HTTPError("transport")
        return "up"

    with patch.object(retry_mod.asyncio, "sleep", new_callable=AsyncMock):
        out = await call_with_retry(
            flaky, max_retries=2, async_mode=True, logger=logger
        )
    assert out == "up"


@pytest.mark.asyncio
async def test_unexpected_exception_maps_provider_error() -> None:
    logger = __import__("logging").getLogger("t")

    async def boom() -> None:
        raise RuntimeError("weird")

    with pytest.raises(ProviderError):
        await call_with_retry(boom, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_unprocessable_maps_context_via_body_shape() -> None:
    logger = __import__("logging").getLogger("t")

    async def bad() -> None:
        raise SdkUnprocessableEntityError(
            message="422",
            response=anthropic_dummy_response(422),
            body={"message": "exceeds the context"},
        )

    with pytest.raises(ContextLengthError):
        await call_with_retry(bad, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_sync_mode_plain_success() -> None:
    logger = __import__("logging").getLogger("t")

    def sync_ok() -> int:
        return 7

    assert (
        await call_with_retry(sync_ok, max_retries=1, async_mode=False, logger=logger)
        == 7
    )


@pytest.mark.asyncio
async def test_bad_request_uses_context_keyword_short_circuit() -> None:
    logger = __import__("logging").getLogger("t")

    async def bad() -> None:
        raise SdkBadRequestError(
            message="nope",
            response=anthropic_dummy_response(400),
            body={"error": {"message": "maximum context window hit"}},
        )

    with pytest.raises(ContextLengthError):
        await call_with_retry(bad, max_retries=1, async_mode=True, logger=logger)


@pytest.mark.asyncio
async def test_api_status_500_sync_mode_sleep_and_retry() -> None:
    logger = __import__("logging").getLogger("t")

    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise SdkAPIStatusError(
                message="500",
                response=anthropic_dummy_response(500),
                body=None,
            )
        return "recovered"

    with patch.object(retry_mod.time, "sleep"):
        out = await call_with_retry(
            flaky, max_retries=2, async_mode=False, logger=logger
        )

    assert out == "recovered"


@pytest.mark.asyncio
async def test_httpx_error_sync_mode_sleep() -> None:
    logger = __import__("logging").getLogger("t")

    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.HTTPError("sync transport")
        return "ok"

    with patch.object(retry_mod.time, "sleep"):
        out = await call_with_retry(
            flaky, max_retries=2, async_mode=False, logger=logger
        )

    assert out == "ok"
