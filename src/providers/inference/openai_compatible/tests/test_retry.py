"""Error translation, retry behaviour and credential scrubbing on the way out."""

from __future__ import annotations

import logging

import httpx
import openai
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
from nucleusiq_openai_compatible._shared.redact import Redactor
from nucleusiq_openai_compatible._shared.retry import (
    PROVIDER,
    call_with_retry,
    is_context_length_error,
)

LOGGER = logging.getLogger("test")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff is real time; skip it so retry tests stay fast."""

    async def instant(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "nucleusiq_openai_compatible._shared.retry.asyncio.sleep", instant
    )


def make_response(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "http://gpu:8000/v1/chat/completions"),
        json=body if body is not None else {"error": {"message": "err"}},
    )


def api_error(cls, status: int, message: str, body: object = None):
    return cls(message=message, response=make_response(status, body), body=body)


async def run(side_effect, *, max_retries: int = 0, redactor=None):
    calls = {"n": 0}

    async def api_call():
        calls["n"] += 1
        if isinstance(side_effect, list):
            outcome = side_effect[min(calls["n"] - 1, len(side_effect) - 1)]
        else:
            outcome = side_effect
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    result = await call_with_retry(
        api_call,
        max_retries=max_retries,
        async_mode=True,
        logger=LOGGER,
        redactor=redactor,
    )
    return result, calls["n"]


class TestSuccess:
    async def test_passes_result_through(self) -> None:
        result, attempts = await run("ok")
        assert result == "ok"
        assert attempts == 1

    async def test_provider_name(self) -> None:
        assert PROVIDER == "openai_compatible"


class TestNonRetryable:
    async def test_authentication(self) -> None:
        exc = api_error(openai.AuthenticationError, 401, "invalid api key")
        with pytest.raises(AuthenticationError) as caught:
            await run(exc, max_retries=3)
        assert "api_key=" in str(caught.value), "must say how to fix it"

    async def test_authentication_is_not_retried(self) -> None:
        exc = api_error(openai.AuthenticationError, 401, "invalid api key")
        calls = {"n": 0}

        async def api_call():
            calls["n"] += 1
            raise exc

        with pytest.raises(AuthenticationError):
            await call_with_retry(
                api_call, max_retries=5, async_mode=True, logger=LOGGER
            )
        assert calls["n"] == 1, "a bad key never becomes a good key by waiting"

    async def test_permission_denied(self) -> None:
        exc = api_error(openai.PermissionDeniedError, 403, "forbidden")
        with pytest.raises(PermissionDeniedError):
            await run(exc)

    async def test_not_found_mentions_v1(self) -> None:
        exc = api_error(openai.NotFoundError, 404, "model not found")
        with pytest.raises(ModelNotFoundError) as caught:
            await run(exc)
        assert "/v1" in str(caught.value)


class TestBadRequestRouting:
    @pytest.mark.parametrize(
        "message",
        [
            "This model's maximum context length is 8192 tokens",
            "the input is longer than the maximum model length",
            "Please reduce the length of the messages",
            "max_model_len exceeded",
            "Input validation error: inputs tokens + max_new_tokens exceeds the maximum",
            "requested tokens exceed n_ctx",
        ],
    )
    async def test_context_overflow_across_engines(self, message: str) -> None:
        exc = api_error(openai.BadRequestError, 400, message)
        with pytest.raises(ContextLengthError) as caught:
            await run(exc)
        assert "context_window" in str(caught.value), (
            "the operator needs to know their configured window may be wrong"
        )

    async def test_content_filter(self) -> None:
        exc = api_error(openai.BadRequestError, 400, "content_policy_violation")
        with pytest.raises(ContentFilterError):
            await run(exc)

    async def test_unknown_parameter_is_invalid_request(self) -> None:
        exc = api_error(openai.BadRequestError, 400, "unknown parameter: store")
        with pytest.raises(InvalidRequestError) as caught:
            await run(exc)
        assert "extra_body" in str(caught.value)

    async def test_unprocessable_entity(self) -> None:
        exc = api_error(openai.UnprocessableEntityError, 422, "bad schema")
        with pytest.raises(InvalidRequestError):
            await run(exc)

    async def test_body_dict_is_searched_for_markers(self) -> None:
        exc = api_error(
            openai.BadRequestError,
            400,
            "request failed",
            body={"error": {"message": "maximum context length reached"}},
        )
        with pytest.raises(ContextLengthError):
            await run(exc)

    def test_marker_helper(self) -> None:
        assert is_context_length_error("maximum context length is 4096")
        assert not is_context_length_error("temperature must be between 0 and 2")


class TestRetryable:
    async def test_rate_limit_recovers(self) -> None:
        exc = api_error(openai.RateLimitError, 429, "slow down")
        result, attempts = await run([exc, "ok"], max_retries=2)
        assert result == "ok"
        assert attempts == 2

    async def test_rate_limit_exhausted(self) -> None:
        exc = api_error(openai.RateLimitError, 429, "slow down")
        with pytest.raises(RateLimitError):
            await run(exc, max_retries=1)

    async def test_connection_error_recovers(self) -> None:
        exc = openai.APIConnectionError(
            request=httpx.Request("POST", "http://gpu:8000/v1")
        )
        result, attempts = await run([exc, "ok"], max_retries=3)
        assert result == "ok"
        assert attempts == 2

    async def test_connection_error_exhausted_is_actionable(self) -> None:
        exc = openai.APIConnectionError(
            request=httpx.Request("POST", "http://gpu:8000/v1")
        )
        with pytest.raises(ProviderConnectionError) as caught:
            await run(exc, max_retries=1)
        assert "base_url is correct" in str(caught.value)

    async def test_5xx_retried(self) -> None:
        exc = api_error(openai.InternalServerError, 500, "boom")
        result, attempts = await run([exc, "ok"], max_retries=2)
        assert result == "ok"
        assert attempts == 2

    async def test_5xx_exhausted(self) -> None:
        exc = api_error(openai.InternalServerError, 503, "unavailable")
        with pytest.raises(ProviderServerError):
            await run(exc, max_retries=1)

    @pytest.mark.parametrize(
        "message",
        [
            "model is still loading",
            "the model is currently loading, please retry",
            "server not yet ready",
        ],
    )
    async def test_model_loading_is_retried(self, message: str) -> None:
        # A cold GPU node needs minutes; this must not fail the run.
        exc = api_error(openai.APIStatusError, 503, message)
        result, attempts = await run([exc, "ok"], max_retries=2)
        assert result == "ok"
        assert attempts == 2

    async def test_4xx_status_error_not_retried(self) -> None:
        exc = api_error(openai.APIStatusError, 418, "teapot")
        with pytest.raises(InvalidRequestError):
            await run(exc, max_retries=3)

    async def test_httpx_error_retried(self) -> None:
        result, attempts = await run(
            [httpx.ReadTimeout("timed out"), "ok"], max_retries=2
        )
        assert result == "ok"
        assert attempts == 2

    async def test_httpx_error_exhausted(self) -> None:
        with pytest.raises(ProviderConnectionError):
            await run(httpx.ReadTimeout("timed out"), max_retries=1)

    async def test_unexpected_error_wrapped(self) -> None:
        with pytest.raises(ProviderError) as caught:
            await run(ValueError("something odd"))
        assert "something odd" in str(caught.value)


class TestCredentialScrubbing:
    async def test_secret_absent_from_raised_error(self) -> None:
        exc = api_error(
            openai.BadRequestError,
            400,
            "rejected header Authorization: Bearer token-abc123",
        )
        redactor = Redactor(secrets=["token-abc123"])
        with pytest.raises(InvalidRequestError) as caught:
            await run(exc, redactor=redactor)
        assert "token-abc123" not in str(caught.value), (
            "gateways echo request headers into error bodies; the secret must "
            "not reach a traceback or a log aggregator"
        )

    async def test_secret_absent_from_logs(self, caplog) -> None:
        exc = api_error(openai.NotFoundError, 404, "no route for sk-abcdefghijklmnop")
        with caplog.at_level("ERROR"), pytest.raises(ModelNotFoundError):
            await run(exc)
        assert "sk-abcdefghijklmnop" not in caplog.text

    async def test_scrubbed_on_retry_warnings(self, caplog) -> None:
        exc = api_error(openai.RateLimitError, 429, "limit for token-abc123")
        redactor = Redactor(secrets=["token-abc123"])
        with caplog.at_level("WARNING"):
            await run([exc, "ok"], max_retries=2, redactor=redactor)
        assert "token-abc123" not in caplog.text

    async def test_default_redactor_when_none_supplied(self) -> None:
        exc = api_error(openai.NotFoundError, 404, "bad sk-abcdefghijklmnopqrs")
        with pytest.raises(ModelNotFoundError) as caught:
            await run(exc, redactor=None)
        assert "sk-abcdefghijklmnopqrs" not in str(caught.value)
