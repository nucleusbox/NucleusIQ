"""Serper.dev — Google SERP results without a CSE id (CrewAI's Google path)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nucleusiq.tools.web_search.config import env_or, require_config
from nucleusiq.tools.web_search.http import request_json
from nucleusiq.tools.web_search.protocol import WebSearchBackend
from nucleusiq.tools.web_search.types import SearchHit

_ENDPOINT = "https://google.serper.dev/search"


class SerperSearchBackend(WebSearchBackend):
    """Serper Google search. Requires ``api_key`` or ``SERPER_API_KEY``."""

    name: ClassVar[str] = "serper"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 15.0,
        **_: Any,
    ) -> None:
        resolved = require_config(
            provider="Serper",
            fields={"api_key": env_or(api_key, "SERPER_API_KEY")},
            how=(
                "Pass WebSearchTool(provider='serper', api_key='...') "
                "or set SERPER_API_KEY."
            ),
        )
        self.api_key = resolved["api_key"]
        self.timeout = timeout

    async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        payload = await asyncio.to_thread(
            request_json,
            "POST",
            _ENDPOINT,
            headers={"X-API-KEY": self.api_key},
            body={"q": query, "num": max_results},
            timeout=self.timeout,
        )
        return [
            SearchHit(
                title=str(row.get("title") or "(untitled)"),
                url=str(row.get("link") or row.get("url") or ""),
                snippet=str(row.get("snippet") or ""),
            )
            for row in (payload.get("organic") or [])[:max_results]
        ]
