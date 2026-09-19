"""Shared DuckDuckGo / Bing adapter via the ``ddgs`` metasearch library."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nucleusiq.tools.web_search.protocol import WebSearchBackend
from nucleusiq.tools.web_search.types import SearchHit

_MISSING_DDGS = (
    "DuckDuckGo / Bing search requires the ddgs package, which nucleusiq "
    "installs by default. Reinstall nucleusiq if this import fails. "
    "Do not install the frozen duckduckgo-search package."
)


def _hit_from_row(row: dict[str, Any]) -> SearchHit:
    return SearchHit(
        title=str(row.get("title") or row.get("name") or "(untitled)").strip(),
        url=str(row.get("href") or row.get("url") or "").strip(),
        snippet=str(
            row.get("body") or row.get("snippet") or row.get("description") or ""
        ).strip(),
    )


def _search_sync(
    query: str,
    *,
    max_results: int,
    region: str,
    safesearch: str,
    engine: str,
    timeout: float,
) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError(_MISSING_DDGS) from exc

    # Text search only — never ``extract()`` (would fetch arbitrary URLs).
    try:
        raw = DDGS(timeout=int(timeout)).text(
            query,
            region=region,
            safesearch=safesearch,
            max_results=max_results,
            backend=engine,
        )
    except Exception as exc:
        if "no results" in str(exc).lower():
            return []
        raise
    return list(raw or [])


class DdgsSearchBackend(WebSearchBackend):
    """Free metasearch via ``ddgs``. No API key.

    *engine* is a ddgs backend name (``duckduckgo``, ``bing``, ``auto``, …).
    """

    name: ClassVar[str] = "duckduckgo"

    def __init__(
        self,
        *,
        engine: str | None = None,
        region: str = "us-en",
        safesearch: str = "moderate",
        timeout: float = 15.0,
        **_: Any,
    ) -> None:
        self.engine = engine or self.name
        self.region = region
        self.safesearch = safesearch
        self.timeout = timeout

    async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        rows = await asyncio.to_thread(
            _search_sync,
            query,
            max_results=max_results,
            region=self.region,
            safesearch=self.safesearch,
            engine=self.engine,
            timeout=self.timeout,
        )
        return [_hit_from_row(row) for row in rows]


class DuckDuckGoSearch(DdgsSearchBackend):
    """Default search adapter. Free, no configuration.

    DuckDuckGo's documented Instant Answer API is **not** a web-search
    API (no ranked links). This adapter uses ``ddgs`` for title / URL /
    snippet only and never calls ``extract()`` (no page fetch).
    """

    name = "duckduckgo"


class BingSearchBackend(DdgsSearchBackend):
    """Bing index via ``ddgs`` — Microsoft retired the Bing Web Search API
    in August 2025. There is no official raw-results key to collect.
    """

    name = "bing"
