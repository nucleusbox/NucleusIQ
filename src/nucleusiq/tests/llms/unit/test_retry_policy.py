"""Unit tests for :mod:`nucleusiq.llms.retry_policy`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from nucleusiq.llms.retry_policy import (
    compute_rate_limit_sleep,
    extract_retry_after_header,
    parse_retry_after_seconds,
)


def test_extract_retry_after_header_missing() -> None:
    assert extract_retry_after_header(None) is None
    assert extract_retry_after_header(object()) is None


def test_extract_retry_after_from_httpx_response() -> None:
    req = httpx.Request("GET", "https://api.example.com")
    resp = httpx.Response(429, headers={"retry-after": "7"}, request=req)
    assert extract_retry_after_header(resp) == "7"


def test_parse_retry_after_integer() -> None:
    assert parse_retry_after_seconds("15") == 15
    assert parse_retry_after_seconds("  0  ") == 0


def test_parse_retry_after_bad() -> None:
    assert parse_retry_after_seconds(None) is None
    assert parse_retry_after_seconds("") is None
    assert parse_retry_after_seconds("not-a-date") is None


def test_parse_retry_after_http_date() -> None:
    now = datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)
    future = now + timedelta(seconds=42)
    # RFC 7231 IMF-fixdate
    hdr = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after_seconds(hdr, now=now) == 42


def test_compute_no_header_uses_exponential_capped() -> None:
    sleep, meta = compute_rate_limit_sleep(3, None, max_sleep_seconds=100.0)
    assert sleep == pytest.approx(8.0)
    assert meta["exponential_seconds"] == pytest.approx(8.0)
    assert meta["used_retry_after"] is False
    assert meta["retry_after_seconds"] is None


def test_compute_retry_after_greater_than_exp() -> None:
    sleep, meta = compute_rate_limit_sleep(1, "25", max_sleep_seconds=100.0)
    assert sleep == pytest.approx(25.0)
    assert meta["used_retry_after"] is True
    assert meta["retry_after_seconds"] == 25


def test_compute_exp_greater_than_retry_after() -> None:
    sleep, meta = compute_rate_limit_sleep(3, "3", max_sleep_seconds=100.0)
    assert sleep == pytest.approx(8.0)


def test_compute_final_cap() -> None:
    sleep, meta = compute_rate_limit_sleep(10, "500", max_sleep_seconds=60.0)
    assert sleep == pytest.approx(60.0)
    assert meta["retry_after_seconds"] == 500
