"""Tavily Search API (agent-oriented, used by LangChain / CrewAI)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nucleusiq.tools.web_search.config import env_or, require_config
from nucleusiq.tools.web_search.http import request_json
from nucleusiq.tools.web_search.protocol import WebSearchBackend
from nucleusiq.tools.web_search.types import SearchHit

_ENDPOINT = "https://api.tavily.com/search"


class TavilySearchBackend(WebSearchBackend):
    """Tavily Search API. Requires ``api_key`` or ``TAVILY_API_KEY``."""

    name: ClassVar[str] = "tavily"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 15.0,
        include_domains: list[str] | tuple[str, ...] | None = None,
        **_: Any,
    ) -> None:
        resolved = require_config(
            provider="Tavily",
            fields={"api_key": env_or(api_key, "TAVILY_API_KEY")},
            how=(
                "Pass WebSearchTool(provider='tavily', api_key='...') "
                "or set TAVILY_API_KEY."
            ),
        )
        self.api_key = resolved["api_key"]
        self.timeout = timeout
        self.include_domains = list(include_domains or [])

    async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        payload = await asyncio.to_thread(
            request_json,
            "POST",
            _ENDPOINT,
            body={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                **(
                    {"include_domains": self.include_domains}
                    if self.include_domains
                    else {}
                ),
            },
            timeout=self.timeout,
        )
        return [
            SearchHit(
                title=str(row.get("title") or "(untitled)"),
                url=str(row.get("url") or ""),
                snippet=str(row.get("content") or row.get("snippet") or ""),
            )
            for row in (payload.get("results") or [])[:max_results]
        ]
