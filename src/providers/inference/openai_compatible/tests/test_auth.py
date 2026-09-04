"""Credential strategies, lazy resolution and secret hygiene (BYOK)."""

from __future__ import annotations

import pytest
from nucleusiq_openai_compatible.auth import (
    ENV_API_KEY,
    AuthStrategy,
    BearerAuth,
    HeaderAuth,
    NoAuth,
    build_auth,
)


class TestNoAuth:
    async def test_sends_nothing(self) -> None:
        auth = NoAuth()
        assert await auth.headers() == {}
        assert await auth.api_key() is None
        assert auth.header_names == ()

    def test_is_a_value(self) -> None:
        assert NoAuth() == NoAuth()
        assert NoAuth() != BearerAuth("x-token-value")
        assert len({NoAuth(), NoAuth()}) == 1

    def test_repr(self) -> None:
        assert repr(NoAuth()) == "NoAuth()"

    def test_satisfies_protocol(self) -> None:
        assert isinstance(NoAuth(), AuthStrategy)


class TestBearerAuth:
    async def test_static_token(self) -> None:
        auth = BearerAuth("token-abc123")
        assert await auth.headers() == {"Authorization": "Bearer token-abc123"}
        assert await auth.api_key() == "token-abc123"
        assert auth.header_names == ("Authorization",)

    async def test_sync_callable_resolves_per_request(self) -> None:
        calls: list[int] = []

        def provider() -> str:
            calls.append(1)
            return f"key-{len(calls)}"

        auth = BearerAuth(provider)
        first = await auth.headers()
        second = await auth.headers()

        assert first == {"Authorization": "Bearer key-1"}
        assert second == {"Authorization": "Bearer key-2"}, (
            "a callable credential must be re-resolved every request so key "
            "rotation takes effect without rebuilding the agent"
        )

    async def test_async_callable(self) -> None:
        async def provider() -> str:
            return "async-key"

        assert await BearerAuth(provider).api_key() == "async-key"

    def test_rejects_empty_token(self) -> None:
        with pytest.raises(ValueError, match="non-empty token"):
            BearerAuth("   ")

    async def test_rejects_non_callable_non_string(self) -> None:
        auth = BearerAuth("placeholder")
        auth._token = 12345  # type: ignore[assignment]
        with pytest.raises(TypeError, match="string or a callable"):
            await auth.headers()

    async def test_rejects_provider_returning_blank(self) -> None:
        with pytest.raises(ValueError, match="no usable value"):
            await BearerAuth(lambda: "").headers()

    async def test_rejects_provider_returning_non_string(self) -> None:
        with pytest.raises(ValueError, match="no usable value"):
            await BearerAuth(lambda: None).headers()  # type: ignore[return-value]

    def test_repr_hides_token(self) -> None:
        assert repr(BearerAuth("token-abc123")) == "BearerAuth(token=<redacted>)"
        assert "token-abc123" not in repr(BearerAuth("token-abc123"))


class TestHeaderAuth:
    async def test_custom_header(self) -> None:
        auth = HeaderAuth("api-key", "azure-secret")
        assert await auth.headers() == {"api-key": "azure-secret"}
        assert auth.header_names == ("api-key",)

    async def test_leaves_sdk_key_slot_empty(self) -> None:
        assert await HeaderAuth("api-key", "v-secret").api_key() is None

    async def test_callable_value(self) -> None:
        assert await HeaderAuth("x-api-key", lambda: "dyn").headers() == {
            "x-api-key": "dyn"
        }

    @pytest.mark.parametrize(
        ("name", "value", "match"),
        [
            ("", "v", "header name"),
            ("   ", "v", "header name"),
            ("api-key", "  ", "non-empty value"),
        ],
    )
    def test_rejects_blanks(self, name: str, value: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            HeaderAuth(name, value)

    def test_repr_hides_value(self) -> None:
        text = repr(HeaderAuth("api-key", "azure-secret"))
        assert text == "HeaderAuth(name='api-key', value=<redacted>)"
        assert "azure-secret" not in text

    def test_strips_header_name(self) -> None:
        assert HeaderAuth("  api-key  ", "v").header_names == ("api-key",)


class TestBuildAuth:
    def test_defaults_to_no_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        assert isinstance(build_auth(), NoAuth), (
            "an unauthenticated local server is the normal case and must never raise"
        )

    def test_api_key_becomes_bearer(self) -> None:
        assert isinstance(build_auth(api_key="k-value"), BearerAuth)

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_API_KEY, "env-token")
        auth = build_auth()
        assert isinstance(auth, BearerAuth)

    def test_blank_env_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_API_KEY, "   ")
        assert isinstance(build_auth(), NoAuth)

    def test_explicit_auth_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_API_KEY, "env-token")
        strategy = HeaderAuth("api-key", "explicit")
        assert build_auth(auth=strategy) is strategy

    def test_rejects_both(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            build_auth(auth=NoAuth(), api_key="k-value")

    def test_rejects_non_strategy(self) -> None:
        with pytest.raises(TypeError, match="AuthStrategy protocol"):
            build_auth(auth="Bearer token")  # type: ignore[arg-type]

    def test_custom_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TENANT_A_KEY", "tenant-a")
        auth = build_auth(env_var="TENANT_A_KEY")
        assert isinstance(auth, BearerAuth)
