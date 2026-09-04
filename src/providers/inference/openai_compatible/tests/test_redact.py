"""Credential scrubbing — the guarantee that secrets never leave the package."""

from __future__ import annotations

import pytest
from nucleusiq_openai_compatible._shared.redact import PLACEHOLDER, Redactor


class TestKnownSecrets:
    def test_removes_exact_secret(self) -> None:
        scrubbed = Redactor(secrets=["token-abc123"]).scrub("key=token-abc123 sent")
        assert "token-abc123" not in scrubbed
        assert PLACEHOLDER in scrubbed

    def test_removes_every_occurrence(self) -> None:
        out = Redactor(secrets=["s3cret-value"]).scrub("s3cret-value s3cret-value")
        assert "s3cret" not in out

    def test_longest_secret_first(self) -> None:
        out = Redactor(secrets=["abc", "abcdef123456"]).scrub("abcdef123456")
        assert out == PLACEHOLDER

    def test_ignores_trivially_short_secrets(self) -> None:
        # A 3-char "secret" would redact half of every message.
        assert Redactor(secrets=["ab"]).scrub("about") == "about"

    def test_empty_text_passthrough(self) -> None:
        assert Redactor(secrets=["x-secret"]).scrub("") == ""


class TestHeaderScrubbing:
    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer token-abc123",
            "authorization: token-abc123",
            'Authorization="Bearer token-abc123"',
            "api-key: token-abc123",
            "x-api-key: token-abc123",
            "proxy-authorization: Basic token-abc123",
        ],
    )
    def test_known_header_values_removed(self, text: str) -> None:
        assert "token-abc123" not in Redactor().scrub(text)

    def test_scheme_word_consumed(self) -> None:
        assert (
            Redactor(secrets=["token-abc123"]).scrub(
                "Authorization: Bearer token-abc123"
            )
            == f"Authorization: {PLACEHOLDER}"
        ), "the redacted value should read cleanly, not leave a bare 'Bearer'"

    def test_custom_header_name(self) -> None:
        out = Redactor(header_names=["x-tenant-token"]).scrub(
            "x-tenant-token: abc123xyz789"
        )
        assert "abc123xyz789" not in out

    def test_unknown_header_name_not_scrubbed_by_name(self) -> None:
        assert "hello" in Redactor().scrub("x-trace-id: hello")


class TestPatternScrubbing:
    @pytest.mark.parametrize(
        "token",
        [
            "sk-abcdefghijklmnopqrs",
            "sk_abcdefghijklmnopqrs",
            "xoxb-1234567890abcdef",
            "hf-abcdefghijklmnopqrs",
            "nvapi-abcdefghijklmnop",
            "gsk_abcdefghijklmnopqr",
            "dapi-abcdefghijklmnopq",
        ],
    )
    def test_vendor_prefixed_tokens(self, token: str) -> None:
        assert token not in Redactor().scrub(f"leaked {token} here")

    def test_jwt(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefgh"
        assert jwt not in Redactor().scrub(f"token {jwt}")

    def test_bearer_without_header_name(self) -> None:
        assert "abcd1234efgh" not in Redactor().scrub("sent Bearer abcd1234efgh")

    def test_leaves_ordinary_text_alone(self) -> None:
        text = "Model gemma-4-27b-it exceeds max_model_len of 32768 tokens"
        assert Redactor().scrub(text) == text, (
            "over-eager redaction would destroy the diagnostic value of error messages"
        )


class TestExceptionScrubbing:
    def test_scrub_exception(self) -> None:
        exc = RuntimeError("401 for Authorization: Bearer token-abc123")
        out = Redactor(secrets=["token-abc123"]).scrub_exception(exc)
        assert "token-abc123" not in out

    def test_repr_reports_no_secrets(self) -> None:
        text = repr(Redactor(secrets=["token-abc123"], header_names=["api-key"]))
        assert "token-abc123" not in text
        assert text.startswith("Redactor(")
