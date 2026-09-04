"""Bring your own key: every credential shape these servers use.

Absence of a credential is not an error — a local vLLM server started
without ``--api-key`` is the common case, so :class:`NoAuth` is the default.

Callable credentials are resolved **once per request**, so rotation and
per-tenant keys work without rebuilding the agent. Anything more exotic
(mTLS, SigV4, cookie auth) is handled by passing a pre-configured
``httpx.AsyncClient`` as ``http_client=`` rather than a new strategy class.

Run:
    python examples/07_bring_your_own_key.py
"""

from __future__ import annotations

import asyncio
import os
import time

from nucleusiq_openai_compatible import (
    BearerAuth,
    HeaderAuth,
    NoAuth,
    OpenAICompatibleLLM,
)

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gemma-4-27b-it")

COMMON = {"base_url": BASE_URL, "model": MODEL, "context_window": 32_768}


def unauthenticated() -> OpenAICompatibleLLM:
    """A local server started without --api-key. The default."""
    return OpenAICompatibleLLM(**COMMON, engine="vllm")


def static_key() -> OpenAICompatibleLLM:
    """vllm serve --api-key secret-123, or any hosted OpenAI-compatible cloud.

    Sends ``Authorization: Bearer secret-123``.
    """
    return OpenAICompatibleLLM(
        **COMMON, engine="vllm", api_key=os.getenv("MY_SERVER_KEY", "secret-123")
    )


def key_from_environment() -> OpenAICompatibleLLM:
    """Nothing passed: $OPENAI_COMPATIBLE_API_KEY is read if present."""
    return OpenAICompatibleLLM(**COMMON, engine="vllm")


def rotating_key() -> OpenAICompatibleLLM:
    """A short-lived token minted per request.

    The callable runs once per call, so an expiring token refreshes itself.
    """
    cache: dict[str, tuple[str, float]] = {}

    def mint_token() -> str:
        token, expiry = cache.get("t", ("", 0.0))
        if time.monotonic() >= expiry:
            # In production: call your STS / vault here.
            token = f"minted-{int(time.monotonic())}"
            cache["t"] = (token, time.monotonic() + 300)
        return token

    return OpenAICompatibleLLM(**COMMON, engine="vllm", api_key=mint_token)


def async_key() -> OpenAICompatibleLLM:
    """An async credential provider, e.g. an HTTP call to a vault."""

    async def fetch_token() -> str:
        await asyncio.sleep(0)  # stand-in for the real await
        return os.getenv("VAULT_TOKEN", "vault-token")

    return OpenAICompatibleLLM(**COMMON, engine="vllm", api_key=fetch_token)


def azure_style() -> OpenAICompatibleLLM:
    """Azure OpenAI wants ``api-key: <key>``, not ``Authorization: Bearer``.

    Note the base_url must include the /openai/v1 prefix, and the model name
    is the *deployment* name.
    """
    return OpenAICompatibleLLM(
        base_url=os.getenv(
            "AZURE_BASE_URL", "https://my-resource.openai.azure.com/openai/v1"
        ),
        model=os.getenv("AZURE_DEPLOYMENT", "my-deployment"),
        engine="azure",
        context_window=128_000,
        auth=HeaderAuth("api-key", os.getenv("AZURE_API_KEY", "azure-key")),
    )


def gateway_style() -> OpenAICompatibleLLM:
    """A corporate gateway wanting X-API-Key plus routing headers."""
    return OpenAICompatibleLLM(
        **COMMON,
        engine="litellm",
        auth=HeaderAuth("X-API-Key", os.getenv("GATEWAY_KEY", "gw-key")),
        default_headers={"X-Team": "platform", "X-Cost-Center": "ml-infra"},
    )


def explicit_no_auth() -> OpenAICompatibleLLM:
    """Ignore $OPENAI_COMPATIBLE_API_KEY even if it is set."""
    return OpenAICompatibleLLM(**COMMON, engine="vllm", auth=NoAuth())


def per_tenant_calls() -> OpenAICompatibleLLM:
    """One provider instance, a different key per call.

    ``call(api_key=...)`` overrides the configured credential for that
    request only, sharing the connection pool across tenants.
    """
    return OpenAICompatibleLLM(**COMMON, engine="vllm", api_key="default-key")


async def main() -> None:
    builders = {
        "unauthenticated (default)": unauthenticated,
        "static key": static_key,
        "key from $OPENAI_COMPATIBLE_API_KEY": key_from_environment,
        "rotating callable": rotating_key,
        "async callable": async_key,
        "Azure api-key header": azure_style,
        "gateway X-API-Key": gateway_style,
        "explicit NoAuth": explicit_no_auth,
        "per-tenant": per_tenant_calls,
    }

    for label, build in builders.items():
        llm = build()
        # repr never contains a credential.
        print(f"{label:38} {llm!r}")

    print("\nBearerAuth / HeaderAuth also hide their values:")
    print(f"  {BearerAuth('super-secret-token')!r}")
    print(f"  {HeaderAuth('api-key', 'super-secret-value')!r}")

    print("\nPer-call override (not sent — illustration only):")
    print("  await llm.call(messages=..., api_key=tenant_key)")


if __name__ == "__main__":
    asyncio.run(main())
