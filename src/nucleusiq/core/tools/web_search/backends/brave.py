"""Brave Search API."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nucleusiq.tools.web_search.config import env_or, require_config
from nucleusiq.tools.web_search.http import request_json
from nucleusiq.tools.web_search.protocol import WebSearchBackend
from nucleusiq.tools.web_search.types import SearchHit

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchBackend(WebSearchBackend):
    """Brave Search API. Requires ``api_key`` or ``BRAVE_API_KEY``."""

    name: ClassVar[str] = "brave"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 15.0,
        **_: Any,
    ) -> None:
        resolved = require_config(
            provider="Brave",
            fields={"api_key": env_or(api_key, "BRAVE_API_KEY")},
            how=(
                "Pass WebSearchTool(provider='brave', api_key='...') "
                "or set BRAVE_API_KEY."
            ),
        )
        self.api_key = resolved["api_key"]
        self.timeout = timeout

    async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        payload = await asyncio.to_thread(
            request_json,
            "GET",
            _ENDPOINT,
            headers={"X-Subscription-Token": self.api_key},
            params={"q": query, "count": max_results},
            timeout=self.timeout,
        )
        results = (payload.get("web") or {}).get("results") or []
        return [
            SearchHit(
                title=str(row.get("title") or "(untitled)"),
                url=str(row.get("url") or ""),
                snippet=str(row.get("description") or ""),
            )
            for row in results[:max_results]
        ]
