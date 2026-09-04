"""Credential scrubbing for logs, ``repr`` output and provider error text.

Self-hosted deployments are frequently fronted by gateways that echo request
headers back in error bodies, so any string that leaves this package — log
line, exception message, telemetry field — passes through :class:`Redactor`
first.

The rule mirrors ``nucleusiq_mcp.auth.BearerAuth.__repr__``: a credential must
never be reconstructable from framework output.
"""

from __future__ import annotations

import re

__all__ = ["PLACEHOLDER", "Redactor"]

PLACEHOLDER = "<redacted>"

# Header names whose *values* are always credentials.  Matched
# case-insensitively against ``Name: value`` and ``"Name": "value"`` shapes.
_SENSITIVE_HEADERS = (
    "authorization",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "proxy-authorization",
)

# Credential-shaped bare tokens: common vendor prefixes followed by a long
# opaque body.  Deliberately conservative — we would rather leave an
# unrecognized token than mangle an unrelated identifier.
_TOKEN_PATTERNS = (
    re.compile(r"\b(?:sk|pk|rk|xoxb|xoxp|hf|nvapi|gsk|dapi|or)[-_][A-Za-z0-9_\-]{12,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+"),
)


class Redactor:
    """Scrubs credentials out of arbitrary text.

    A redactor is constructed with the *literal* secrets it knows about (the
    resolved API key, plus any custom auth header name in use) and combines
    exact-match removal with pattern-based removal for values it was never
    told about.

    Example:
        >>> Redactor(secrets=["token-abc123"]).scrub("key=token-abc123")
        'key=<redacted>'
    """

    __slots__ = ("_header_names", "_secrets")

    def __init__(
        self,
        *,
        secrets: list[str] | None = None,
        header_names: list[str] | None = None,
    ) -> None:
        # Longest first, so overlapping secrets redact the wider match.
        self._secrets = sorted(
            {s for s in (secrets or []) if isinstance(s, str) and len(s.strip()) >= 4},
            key=len,
            reverse=True,
        )
        extra = [h.lower() for h in (header_names or []) if h]
        self._header_names = tuple(dict.fromkeys((*_SENSITIVE_HEADERS, *extra)))

    def scrub(self, text: str) -> str:
        """Return *text* with every known or credential-shaped value removed."""
        if not text:
            return text

        out = text
        for secret in self._secrets:
            out = out.replace(secret, PLACEHOLDER)

        for name in self._header_names:
            # Consume an optional scheme word ("Bearer", "Basic", ...) along
            # with the value, so the result reads `Authorization: <redacted>`
            # rather than leaving a bare scheme behind.
            out = re.sub(
                rf"({re.escape(name)}\"?\s*[:=]\s*\"?)"
                rf"(?:(?:Bearer|Basic|Token)\s+)?[^\s,;\"'}}\])]*",
                rf"\1{PLACEHOLDER}",
                out,
                flags=re.IGNORECASE,
            )

        for pattern in _TOKEN_PATTERNS:
            out = pattern.sub(PLACEHOLDER, out)

        return out

    def scrub_exception(self, exc: BaseException) -> str:
        """Return a scrubbed string form of *exc*."""
        return self.scrub(str(exc))

    def __repr__(self) -> str:
        return (
            f"Redactor(secrets={len(self._secrets)}, headers={len(self._header_names)})"
        )
