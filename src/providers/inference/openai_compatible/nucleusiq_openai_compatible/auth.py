"""Credential strategies for OpenAI-compatible endpoints.

Three concrete strategies cover every OpenAI-compatible server on the market:

* :class:`NoAuth` — local vLLM / SGLang / llama.cpp / LM Studio started
  without a key.  A real strategy rather than a ``None`` special case, so
  calling code never branches on absence.
* :class:`BearerAuth` — ``Authorization: Bearer <token>``.  vLLM's
  ``--api-key`` / ``VLLM_API_KEY`` and every hosted OpenAI-compatible cloud.
* :class:`HeaderAuth` — credential in a non-standard header.  Azure OpenAI
  (``api-key``) and gateways using ``X-API-Key``.

Anything more exotic (mTLS, SigV4, cookie auth) is served by passing a
pre-configured ``httpx.AsyncClient`` as ``http_client=`` rather than by adding
a strategy class.

Every strategy accepts either a literal value or a **callable**, sync or
async.  Callables are resolved per request, so credential rotation and
per-tenant keys work without rebuilding the agent — the same lazy-resolution
contract as ``nucleusiq_mcp.auth.EnvAuth``.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AuthMaterial",
    "AuthStrategy",
    "BearerAuth",
    "HeaderAuth",
    "NoAuth",
    "CredentialSource",
    "build_auth",
    "ENV_API_KEY",
]

ENV_API_KEY = "OPENAI_COMPATIBLE_API_KEY"

CredentialSource = str | Callable[[], str] | Callable[[], Awaitable[str]]
"""A literal credential, or a sync/async callable returning one."""


async def _resolve(source: CredentialSource) -> str:
    """Resolve a credential source to a string, awaiting async providers."""
    if isinstance(source, str):
        return source
    if not callable(source):
        raise TypeError(
            "Credential must be a string or a callable returning a string, "
            f"got {type(source).__name__}"
        )
    value = source()
    if inspect.isawaitable(value):
        value = await value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Credential provider returned no usable value; expected a non-empty string"
        )
    return value


@dataclass(frozen=True, slots=True)
class AuthMaterial:
    """Everything one request needs, minted from a single credential read.

    Args:
        headers: Headers to merge into the request.
        api_key: Value for the ``openai`` SDK's ``api_key`` slot, or ``None``
            for header-only strategies.

    Resolving the credential once per request — rather than once for the
    header and again for the SDK slot — matters for two reasons: a callable
    that mints a short-lived token would otherwise be charged twice per
    request, and a rotating provider could return *different* values for the
    two, sending a header and an SDK key that disagree.
    """

    headers: dict[str, str]
    api_key: str | None

    @property
    def secrets(self) -> list[str]:
        """Every literal secret in this material, for the redactor."""
        values = [v for v in self.headers.values() if isinstance(v, str)]
        if self.api_key:
            values.append(self.api_key)
        return values


# ====================================================================== #
# Protocol — the Strategy interface                                        #
# ====================================================================== #


@runtime_checkable
class AuthStrategy(Protocol):
    """Strategy interface for OpenAI-compatible HTTP authentication.

    ``headers``
        Await and return the headers to merge into a request.  Resolved per
        call so callable credentials pick up rotation.

    ``api_key``
        Await and return a value for the ``openai`` SDK's ``api_key``
        argument, or ``None`` when the strategy works purely via headers.
        The SDK requires *some* value, so header-based strategies rely on
        the caller substituting a placeholder.

    ``resolve``
        Await and return an :class:`AuthMaterial` for one request, reading
        the underlying credential exactly once.  This is what callers should
        use; ``headers`` and ``api_key`` are conveniences over it.

    ``header_names``
        Header names this strategy populates, so :class:`Redactor` can scrub
        them even when the name is caller-defined.
    """

    async def resolve(self) -> AuthMaterial:
        """Return headers and SDK key from a single credential read."""
        ...

    async def headers(self) -> dict[str, str]:
        """Return additional HTTP headers (may be empty)."""
        ...

    async def api_key(self) -> str | None:
        """Return an SDK ``api_key`` value, or ``None`` for header-only auth."""
        ...

    @property
    def header_names(self) -> tuple[str, ...]:
        """Header names populated by this strategy."""
        ...


# ====================================================================== #
# Concrete strategies                                                      #
# ====================================================================== #


class NoAuth:
    """No credential at all — the default for local inference servers.

    vLLM, SGLang, llama.cpp and LM Studio accept unauthenticated requests
    unless started with an explicit key, so this must be a first-class,
    non-exceptional state.

    Example:
        >>> import asyncio
        >>> asyncio.run(NoAuth().headers())
        {}
    """

    __slots__ = ()

    async def resolve(self) -> AuthMaterial:
        return AuthMaterial(headers={}, api_key=None)

    async def headers(self) -> dict[str, str]:
        return {}

    async def api_key(self) -> str | None:
        return None

    @property
    def header_names(self) -> tuple[str, ...]:
        return ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NoAuth)

    def __hash__(self) -> int:
        return hash(NoAuth)

    def __repr__(self) -> str:
        return "NoAuth()"


class BearerAuth:
    """``Authorization: Bearer <token>`` — the overwhelmingly common case.

    Covers vLLM's ``--api-key`` / ``VLLM_API_KEY``, SGLang, llama.cpp's
    ``--api-key``, and every hosted OpenAI-compatible cloud (OpenRouter,
    Together, Fireworks, DeepInfra, Databricks, LiteLLM, NIM).

    Args:
        token: A literal token, or a sync/async callable returning one.
            Callables are resolved per request, enabling rotation and
            per-tenant credentials.

    Example:
        >>> BearerAuth("token-abc123")
        BearerAuth(token=<redacted>)
    """

    __slots__ = ("_token",)

    def __init__(self, token: CredentialSource) -> None:
        if isinstance(token, str) and not token.strip():
            raise ValueError(
                "BearerAuth requires a non-empty token; use NoAuth() for "
                "servers that do not require a credential"
            )
        self._token = token

    async def resolve(self) -> AuthMaterial:
        token = await _resolve(self._token)
        return AuthMaterial(headers={"Authorization": f"Bearer {token}"}, api_key=token)

    async def headers(self) -> dict[str, str]:
        return (await self.resolve()).headers

    async def api_key(self) -> str | None:
        return (await self.resolve()).api_key

    @property
    def header_names(self) -> tuple[str, ...]:
        return ("Authorization",)

    def __repr__(self) -> str:
        # NEVER include the token — secrets must not leak via repr / logs.
        return "BearerAuth(token=<redacted>)"


class HeaderAuth:
    """Credential in a caller-named header.

    The one case ``api_key=`` cannot express.  Azure OpenAI expects
    ``api-key: <key>``; corporate gateways commonly want ``X-API-Key``.

    Args:
        name: Header name, e.g. ``"api-key"``.
        value: A literal value, or a sync/async callable returning one.

    Example:
        >>> HeaderAuth("api-key", "abc123")
        HeaderAuth(name='api-key', value=<redacted>)
    """

    __slots__ = ("_name", "_value")

    def __init__(self, name: str, value: CredentialSource) -> None:
        if not name or not name.strip():
            raise ValueError("HeaderAuth requires a non-empty header name")
        if isinstance(value, str) and not value.strip():
            raise ValueError("HeaderAuth requires a non-empty value")
        self._name = name.strip()
        self._value = value

    async def resolve(self) -> AuthMaterial:
        # The credential travels in a custom header; the SDK's own api_key
        # slot is unused and the caller supplies a placeholder.
        return AuthMaterial(
            headers={self._name: await _resolve(self._value)}, api_key=None
        )

    async def headers(self) -> dict[str, str]:
        return (await self.resolve()).headers

    async def api_key(self) -> str | None:
        return None

    @property
    def header_names(self) -> tuple[str, ...]:
        return (self._name,)

    def __repr__(self) -> str:
        return f"HeaderAuth(name={self._name!r}, value=<redacted>)"


# ====================================================================== #
# Coercion                                                                 #
# ====================================================================== #


def build_auth(
    *,
    auth: Any | None = None,
    api_key: CredentialSource | None = None,
    env_var: str = ENV_API_KEY,
) -> AuthStrategy:
    """Coerce user input into an :class:`AuthStrategy`.

    Resolution order: explicit *auth* → *api_key* → ``$OPENAI_COMPATIBLE_API_KEY``
    → :class:`NoAuth`.  Absence of any credential is never an error, because
    unauthenticated local servers are the normal development case.

    Raises:
        ValueError: Both *auth* and *api_key* were supplied, which is
            ambiguous about which credential wins.
        TypeError: *auth* does not implement :class:`AuthStrategy`.
    """
    if auth is not None and api_key is not None:
        raise ValueError(
            "Pass either auth= or api_key=, not both. api_key= is shorthand "
            "for BearerAuth(...); use auth=HeaderAuth(...) only when the "
            "server expects the credential in a non-standard header."
        )

    if auth is not None:
        if not isinstance(auth, AuthStrategy):
            raise TypeError(
                "auth= must implement the AuthStrategy protocol "
                "(headers/api_key/header_names); got "
                f"{type(auth).__name__}"
            )
        return auth

    if api_key is not None:
        return BearerAuth(api_key)

    from_env = os.getenv(env_var)
    if from_env and from_env.strip():
        return BearerAuth(from_env.strip())

    return NoAuth()
