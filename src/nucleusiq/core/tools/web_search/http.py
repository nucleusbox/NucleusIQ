"""Stdlib JSON HTTP helper for paid search APIs.

Core does not depend on httpx/requests. Adapters that need an HTTP client
use this module so a missing extra never blocks ``WebSearchTool``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode


class WebSearchHttpError(Exception):
    """HTTP failure from a search API, with status when available."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> Any:
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urlencode(params)}"

    data = None
    req_headers = {
        "Accept": "application/json",
        "User-Agent": "NucleusIQ-WebSearch/0.7.14 (+https://github.com/nucleusbox/NucleusIQ)",
        **(headers or {}),
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(
        url, data=data, headers=req_headers, method=method.upper()
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code == 429 or "ratelimit" in detail.lower():
            raise WebSearchHttpError(
                f"HTTP 429 rate-limited: {detail}",
                status=exc.code,
            ) from exc
        raise WebSearchHttpError(
            f"HTTP {exc.code}: {detail or exc.reason}",
            status=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise WebSearchHttpError(f"connection failed: {exc.reason}") from exc

    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WebSearchHttpError("provider returned non-JSON") from exc
