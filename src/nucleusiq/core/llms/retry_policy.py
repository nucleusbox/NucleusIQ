"""Shared rate-limit retry policy (Retry-After + exponential backoff, capped).

Provider SDKs surface 429s with optional ``Retry-After`` response headers. This
module centralizes parsing and sleep computation so Groq, OpenAI, and future
adapters behave consistently without duplicating header logic.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypedDict

__all__ = [
    "DEFAULT_RATE_LIMIT_MAX_SLEEP_SECONDS",
    "RateLimitRetryMeta",
    "compute_rate_limit_sleep",
    "extract_retry_after_header",
    "parse_retry_after_seconds",
]

# Default ceiling for a single sleep between 429 retries (seconds).
DEFAULT_RATE_LIMIT_MAX_SLEEP_SECONDS = 120.0

_DIGITS_ONLY = re.compile(r"^\s*(\d+)\s*$")


class RateLimitRetryMeta(TypedDict, total=False):
    """Structured fields for logs / telemetry on a rate-limit retry sleep."""

    attempt: int
    sleep_seconds: float
    retry_after_header_raw: str | None
    retry_after_seconds: int | None
    exponential_seconds: float
    max_sleep_cap: float
    used_retry_after: bool


def extract_retry_after_header(response: Any) -> str | None:
    """Return the raw ``Retry-After`` header value, if present."""
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except (TypeError, AttributeError):
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("ascii", errors="replace")
        except Exception:
            return None
    s = str(raw).strip()
    return s or None


def parse_retry_after_seconds(
    value: str | None,
    *,
    now: datetime | None = None,
) -> int | None:
    """Parse ``Retry-After`` as delay seconds (integer) or HTTP-date.

    Returns ``None`` if *value* is missing or cannot be parsed. For HTTP-dates
    in the past (clock skew), returns ``0``.
    """
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    m = _DIGITS_ONLY.match(v)
    if m:
        return max(0, int(m.group(1)))
    try:
        when = parsedate_to_datetime(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if when is None:
        return None
    if now is None:
        now_utc = datetime.now(timezone.utc)
    else:
        now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delta = (when - now_utc).total_seconds()
    return max(0, int(delta)) if delta >= 0 else 0


def compute_rate_limit_sleep(
    attempt: int,
    retry_after_header: str | None,
    *,
    exponent_base: float = 2.0,
    max_sleep_seconds: float = DEFAULT_RATE_LIMIT_MAX_SLEEP_SECONDS,
    now: datetime | None = None,
) -> tuple[float, RateLimitRetryMeta]:
    """Compute sleep after a 429 using server hint and exponential backoff.

    *attempt* is 1-based (first retry after the initial failure uses ``1``).

    Policy: take the greater of (parsed ``Retry-After`` delay if any,
    capped exponential ``min(exponent_base**attempt, max_sleep_seconds)``),
    then apply ``max_sleep_seconds`` as a final ceiling.

    Returns ``(sleep_seconds, metadata)`` suitable for structured logging.
    """
    ra_sec = parse_retry_after_seconds(retry_after_header, now=now)
    exp = min(exponent_base**attempt, max_sleep_seconds)
    used_ra = ra_sec is not None
    if ra_sec is None:
        sleep = exp
    else:
        combined = max(ra_sec, exp)
        sleep = min(combined, max_sleep_seconds)
    meta: RateLimitRetryMeta = {
        "attempt": attempt,
        "sleep_seconds": sleep,
        "retry_after_header_raw": retry_after_header,
        "retry_after_seconds": ra_sec,
        "exponential_seconds": exp,
        "max_sleep_cap": max_sleep_seconds,
        "used_retry_after": used_ra,
    }
    return sleep, meta
