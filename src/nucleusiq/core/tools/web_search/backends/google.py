"""Google Programmable Search (Custom Search JSON API)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from nucleusiq.tools.web_search.config import env_or, require_config
from nucleusiq.tools.web_search.http import request_json
from nucleusiq.tools.web_search.protocol import WebSearchBackend
from nucleusiq.tools.web_search.types import SearchHit

_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchBackend(WebSearchBackend):
    """Official Google Custom Search JSON API.

    Requires a Cloud API key **and** a Programmable Search Engine id
    (https://programmablesearchengine.google.com/).
    """

    name: ClassVar[str] = "google"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        search_engine_id: str | None = None,
        cx: str | None = None,
        timeout: float = 15.0,
        **_: Any,
    ) -> None:
        resolved = require_config(
            provider="Google",
            fields={
                "api_key": env_or(api_key, "GOOGLE_API_KEY", "GOOGLE_CSE_API_KEY"),
                "search_engine_id": env_or(
                    search_engine_id or cx,
                    "GOOGLE_CSE_ID",
                    "GOOGLE_SEARCH_ENGINE_ID",
                ),
            },
            how=(
                "Pass WebSearchTool(provider='google', api_key='...', "
                "search_engine_id='...') or set GOOGLE_API_KEY and GOOGLE_CSE_ID. "
                "Create the engine at https://programmablesearchengine.google.com/"
            ),
        )
        self.api_key = resolved["api_key"]
        self.search_engine_id = resolved["search_engine_id"]
        self.timeout = timeout

    async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        payload = await asyncio.to_thread(
            request_json,
            "GET",
            _ENDPOINT,
            params={
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": max_results,
            },
            timeout=self.timeout,
        )
        hits: list[SearchHit] = []
        for item in payload.get("items") or []:
            hits.append(
                SearchHit(
                    title=str(item.get("title") or "(untitled)"),
                    url=str(item.get("link") or ""),
                    snippet=str(item.get("snippet") or ""),
                )
            )
        return hits
