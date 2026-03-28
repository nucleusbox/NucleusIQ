"""Tests for nucleusiq.llms.errors — framework-level LLM error taxonomy."""

import pytest
from nucleusiq.llms.errors import (
    AuthenticationError,
    ContentFilterError,
    InvalidRequestError,
    LLMError,
    ModelNotFoundError,
    NucleusIQError,
    PermissionDeniedError,
    ProviderConnectionError,
    ProviderError,
    ProviderServerError,
    RateLimitError,
)


class TestErrorHierarchy:
    """Every error is a subclass of LLMError → NucleusIQError → Exception."""

    def test_llm_error_is_nucleusiq_error(self):
        assert issubclass(LLMError, NucleusIQError)

    def test_nucleusiq_error_is_exception(self):
        assert issubclass(NucleusIQError, Exception)

    @pytest.mark.parametrize(
        "cls",
        [
            AuthenticationError,
            PermissionDeniedError,
            RateLimitError,
            InvalidRequestError,
            ModelNotFoundError,
            ContentFilterError,
            ProviderServerError,
            ProviderConnectionError,
            ProviderError,
        ],
    )
    def test_subclass_is_llm_error(self, cls):
        assert issubclass(cls, LLMError)

    @pytest.mark.parametrize(
        "cls",
        [
            AuthenticationError,
            PermissionDeniedError,
            RateLimitError,
            InvalidRequestError,
            ModelNotFoundError,
            ContentFilterError,
            ProviderServerError,
            ProviderConnectionError,
            ProviderError,
        ],
    )
    def test_subclass_is_nucleusiq_error(self, cls):
        assert issubclass(cls, NucleusIQError)


class TestLLMErrorAttributes:
    def test_default_attributes(self):
        err = LLMError("something broke")
        assert err.provider == "unknown"
        assert err.status_code is None
        assert err.original_error is None
        assert str(err) == "something broke"

    def test_custom_attributes(self):
        orig = ValueError("original")
        err = LLMError(
            "test",
            provider="openai",
            status_code=429,
            original_error=orig,
        )
        assert err.provider == "openai"
        assert err.status_code == 429
        assert err.original_error is orig

    def test_repr(self):
        err = LLMError("msg", provider="gemini", status_code=500)
        r = repr(err)
        assert "LLMError" in r
        assert "gemini" in r
        assert "500" in r


class TestFromProviderError:
    def test_creates_instance(self):
        orig = RuntimeError("sdk error")
        err = RateLimitError.from_provider_error(
            provider="gemini",
            message="Rate limited",
            status_code=429,
            original_error=orig,
        )
        assert isinstance(err, RateLimitError)
        assert err.provider == "gemini"
        assert err.status_code == 429
        assert err.original_error is orig
        assert str(err) == "Rate limited"

    def test_works_for_all_subclasses(self):
        for cls in [
            AuthenticationError,
            PermissionDeniedError,
            RateLimitError,
            InvalidRequestError,
            ModelNotFoundError,
            ContentFilterError,
            ProviderServerError,
            ProviderConnectionError,
            ProviderError,
        ]:
            err = cls.from_provider_error(
                provider="test",
                message=f"Test {cls.__name__}",
            )
            assert isinstance(err, cls)
            assert isinstance(err, LLMError)
            assert err.provider == "test"


class TestCatchability:
    """Users should be able to catch errors at different levels."""

    def test_catch_specific(self):
        err = RateLimitError("too fast", provider="openai", status_code=429)
        with pytest.raises(RateLimitError):
            raise err

    def test_catch_as_llm_error(self):
        err = RateLimitError("too fast", provider="openai")
        with pytest.raises(LLMError):
            raise err

    def test_catch_as_nucleusiq_error(self):
        err = RateLimitError("too fast", provider="openai")
        with pytest.raises(NucleusIQError):
            raise err

    def test_catch_as_exception(self):
        err = RateLimitError("too fast", provider="openai")
        with pytest.raises(Exception):
            raise err

    def test_distinct_types_not_confused(self):
        err = AuthenticationError("bad key", provider="gemini")
        with pytest.raises(AuthenticationError):
            raise err
        assert not isinstance(err, RateLimitError)
        assert not isinstance(err, ProviderServerError)


class TestProviderAgnosticCatching:
    """Same exception type regardless of provider."""

    def test_openai_and_gemini_same_type(self):
        openai_err = RateLimitError("too fast", provider="openai", status_code=429)
        gemini_err = RateLimitError("too fast", provider="gemini", status_code=429)

        assert type(openai_err) is type(gemini_err)
        assert openai_err.provider != gemini_err.provider

    def test_catch_both_providers_with_one_except(self):
        errors = [
            AuthenticationError("bad key", provider="openai"),
            AuthenticationError("bad key", provider="gemini"),
        ]
        for err in errors:
            with pytest.raises(AuthenticationError):
                raise err


class TestImports:
    """Errors importable from the public API."""

    def test_import_from_llms_package(self):
        from nucleusiq.llms import (
            AuthenticationError,
            LLMError,
        )

        assert issubclass(AuthenticationError, LLMError)

    def test_import_from_errors_module(self):
        from nucleusiq.llms.errors import (
            LLMError,
            NucleusIQError,
        )

        assert issubclass(LLMError, NucleusIQError)
