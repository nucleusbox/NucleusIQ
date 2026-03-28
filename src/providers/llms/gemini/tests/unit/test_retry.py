"""Tests for nucleusiq_gemini._shared.retry — retry with exponential backoff.

Verifies that Gemini SDK errors are correctly mapped to framework-level
exceptions from ``nucleusiq.llms.errors``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.llms.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    PermissionDeniedError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
)
from nucleusiq_gemini._shared.retry import _extract_status_code, call_with_retry


def _make_client_error(code: int, message: str = "error"):
    """Create a ClientError with the given status code."""
    from google.genai import errors as genai_errors

    return genai_errors.ClientError(code, {"error": {"message": message}})


def _make_server_error(code: int = 500, message: str = "server error"):
    """Create a ServerError."""
    from google.genai import errors as genai_errors

    return genai_errors.ServerError(code, {"error": {"message": message}})


_NO_SLEEP = patch(
    "nucleusiq_gemini._shared.retry.asyncio.sleep",
    new_callable=AsyncMock,
)


class TestCallWithRetrySuccess:
    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        result = await call_with_retry(
            lambda: {"text": "hello"}, max_retries=3, logger=MagicMock()
        )
        assert result == {"text": "hello"}

    @pytest.mark.asyncio
    async def test_calls_api_once_on_success(self):
        mock_fn = MagicMock(return_value="ok")
        await call_with_retry(mock_fn, max_retries=3, logger=MagicMock())
        mock_fn.assert_called_once()


class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self):
        err = _make_client_error(429, "rate limited")
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise err
            return "success"

        with _NO_SLEEP:
            result = await call_with_retry(api_call, max_retries=3, logger=MagicMock())
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_after_max_retries(self):
        err = _make_client_error(429, "rate limited")

        with _NO_SLEEP:
            with pytest.raises(RateLimitError) as exc_info:
                await call_with_retry(
                    lambda: (_ for _ in ()).throw(err),
                    max_retries=2,
                    logger=MagicMock(),
                )
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.status_code == 429
        assert exc_info.value.original_error is err

    @pytest.mark.asyncio
    async def test_backoff_sleep_called(self):
        err = _make_client_error(429, "rate limited")
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise err
            return "ok"

        with patch(
            "nucleusiq_gemini._shared.retry.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep:
            await call_with_retry(api_call, max_retries=3, logger=MagicMock())
            mock_sleep.assert_awaited_once_with(2)


class TestAuthErrors:
    @pytest.mark.asyncio
    async def test_401_raises_authentication_error(self):
        err = _make_client_error(401, "unauthorized")

        with pytest.raises(AuthenticationError) as exc_info:
            await call_with_retry(
                lambda: (_ for _ in ()).throw(err),
                max_retries=3,
                logger=MagicMock(),
            )
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_403_raises_permission_denied_error(self):
        err = _make_client_error(403, "forbidden")

        with pytest.raises(PermissionDeniedError) as exc_info:
            await call_with_retry(
                lambda: (_ for _ in ()).throw(err),
                max_retries=3,
                logger=MagicMock(),
            )
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_auth_no_retry(self):
        err = _make_client_error(401, "bad key")
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            raise err

        with pytest.raises(AuthenticationError):
            await call_with_retry(api_call, max_retries=5, logger=MagicMock())
        assert call_count == 1


class TestBadRequestErrors:
    @pytest.mark.asyncio
    async def test_400_raises_invalid_request_error(self):
        err = _make_client_error(400, "bad request")

        with pytest.raises(InvalidRequestError) as exc_info:
            await call_with_retry(
                lambda: (_ for _ in ()).throw(err),
                max_retries=3,
                logger=MagicMock(),
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_400_with_on_bad_request_callback(self):
        err = _make_client_error(400, "bad request")
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise err
            return "fixed"

        result = await call_with_retry(
            api_call,
            max_retries=3,
            logger=MagicMock(),
            on_bad_request=lambda: True,
        )
        assert result == "fixed"

    @pytest.mark.asyncio
    async def test_400_callback_returns_false(self):
        err = _make_client_error(400, "bad")

        with pytest.raises(InvalidRequestError):
            await call_with_retry(
                lambda: (_ for _ in ()).throw(err),
                max_retries=3,
                logger=MagicMock(),
                on_bad_request=lambda: False,
            )


class TestModelNotFoundError:
    @pytest.mark.asyncio
    async def test_404_raises_model_not_found(self):
        err = _make_client_error(404, "model not found")

        with pytest.raises(ModelNotFoundError) as exc_info:
            await call_with_retry(
                lambda: (_ for _ in ()).throw(err),
                max_retries=3,
                logger=MagicMock(),
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_no_retry(self):
        err = _make_client_error(404, "not found")
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            raise err

        with pytest.raises(ModelNotFoundError):
            await call_with_retry(api_call, max_retries=5, logger=MagicMock())
        assert call_count == 1


class TestServerErrorRetry:
    @pytest.mark.asyncio
    async def test_retries_on_server_error(self):
        err = _make_server_error(500, "internal error")
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise err
            return "recovered"

        with _NO_SLEEP:
            result = await call_with_retry(api_call, max_retries=3, logger=MagicMock())
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_provider_server_error_after_retries(self):
        err = _make_server_error(500, "down")

        with _NO_SLEEP:
            with pytest.raises(ProviderServerError) as exc_info:
                await call_with_retry(
                    lambda: (_ for _ in ()).throw(err),
                    max_retries=2,
                    logger=MagicMock(),
                )
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.original_error is err


class TestConnectionErrorRetry:
    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("network down")
            return "reconnected"

        with _NO_SLEEP:
            result = await call_with_retry(api_call, max_retries=3, logger=MagicMock())
        assert result == "reconnected"

    @pytest.mark.asyncio
    async def test_retries_on_os_error(self):
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("socket error")
            return "ok"

        with _NO_SLEEP:
            result = await call_with_retry(api_call, max_retries=3, logger=MagicMock())
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_provider_connection_error_after_retries(self):
        with _NO_SLEEP:
            with pytest.raises(ProviderConnectionError) as exc_info:
                await call_with_retry(
                    lambda: (_ for _ in ()).throw(ConnectionError("down")),
                    max_retries=1,
                    logger=MagicMock(),
                )
        assert exc_info.value.provider == "gemini"


class TestUnexpectedErrors:
    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped_as_provider_error(self):
        with pytest.raises(ProviderError) as exc_info:
            await call_with_retry(
                lambda: (_ for _ in ()).throw(RuntimeError("unexpected")),
                max_retries=3,
                logger=MagicMock(),
            )
        assert exc_info.value.provider == "gemini"
        assert isinstance(exc_info.value.original_error, RuntimeError)

    @pytest.mark.asyncio
    async def test_no_retry_on_unexpected(self):
        call_count = 0

        def api_call():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(ProviderError):
            await call_with_retry(api_call, max_retries=5, logger=MagicMock())
        assert call_count == 1


class TestOtherClientErrors:
    @pytest.mark.asyncio
    async def test_other_4xx_retried_then_provider_error(self):
        err = _make_client_error(408, "timeout")

        with _NO_SLEEP:
            with pytest.raises(ProviderError) as exc_info:
                await call_with_retry(
                    lambda: (_ for _ in ()).throw(err),
                    max_retries=2,
                    logger=MagicMock(),
                )
        assert exc_info.value.status_code == 408


class TestErrorAttributes:
    @pytest.mark.asyncio
    async def test_original_error_preserved(self):
        err = _make_client_error(401, "bad key")

        with pytest.raises(AuthenticationError) as exc_info:
            await call_with_retry(
                lambda: (_ for _ in ()).throw(err),
                max_retries=1,
                logger=MagicMock(),
            )
        assert exc_info.value.original_error is err

    @pytest.mark.asyncio
    async def test_error_chain_preserved(self):
        err = _make_client_error(401, "bad key")

        with pytest.raises(AuthenticationError) as exc_info:
            await call_with_retry(
                lambda: (_ for _ in ()).throw(err),
                max_retries=1,
                logger=MagicMock(),
            )
        assert exc_info.value.__cause__ is err


class TestExtractStatusCode:
    def test_extracts_from_response(self):
        err = Exception("test")
        err.response = MagicMock(status_code=429)
        assert _extract_status_code(err) == 429

    def test_returns_none_without_response(self):
        assert _extract_status_code(Exception("test")) is None

    def test_returns_none_without_status_code(self):
        err = Exception("test")
        err.response = MagicMock(spec=[])
        assert _extract_status_code(err) is None
