"""WebSearchTool — first-class web search with a swappable backend."""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.web_search.factory import (
    WebSearchBackendFactory,
    WebSearchProvider,
)
from nucleusiq.tools.web_search.format import (
    DEFAULT_MAX_RESULTS,
    HARD_MAX_RESULTS,
    clamp_max_results,
    format_results,
)
from nucleusiq.tools.web_search.http import WebSearchHttpError
from nucleusiq.tools.web_search.protocol import WebSearchBackend
from nucleusiq.tools.web_search.safety import (
    normalize_domains,
    sanitize_hits,
)

logger = logging.getLogger(__name__)

_DEFAULT_DESCRIPTION = (
    "Search the public web for current information, news, and facts "
    "not in your training data. Input is a search query string. "
    "Returns numbered results with title, URL, and snippet. "
    "Use this instead of guessing about recent events."
)


class WebSearchTool(BaseTool):
    """First-class web search. Default backend is free DuckDuckGo.

    Switch providers with ``provider=`` and the credentials that adapter
    asks for. Or pass a custom ``backend=`` instance.

    Examples::

        WebSearchTool()
        WebSearchTool(allowed_domains=["docs.python.org", "peps.python.org"])
        WebSearchTool(provider="google", api_key="...", search_engine_id="...")
        WebSearchTool(backend=MyBackend(...))
    """

    def __init__(
        self,
        provider: str | WebSearchProvider = WebSearchProvider.DUCKDUCKGO,
        *,
        backend: WebSearchBackend | None = None,
        api_key: str | None = None,
        search_engine_id: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: float = 15.0,
        region: str = "us-en",
        safesearch: str = "moderate",
        allowed_domains: list[str] | tuple[str, ...] | None = None,
        blocked_domains: list[str] | tuple[str, ...] | None = None,
        name: str = "web_search",
        description: str = _DEFAULT_DESCRIPTION,
        **backend_kwargs: Any,
    ) -> None:
        from nucleusiq.agents.context.policy import ContextPolicy

        super().__init__(
            name=name,
            description=description,
            context_policy=ContextPolicy.EVIDENCE,
        )
        self.max_results = clamp_max_results(max_results, DEFAULT_MAX_RESULTS)
        self.allowed_domains = normalize_domains(allowed_domains)
        self.blocked_domains = normalize_domains(blocked_domains)
        if backend is not None:
            self.backend = backend
        else:
            self.backend = WebSearchBackendFactory.create(
                provider,
                api_key=api_key,
                search_engine_id=search_engine_id,
                timeout=timeout,
                region=region,
                safesearch=safesearch,
                include_domains=self.allowed_domains,
                **backend_kwargs,
            )

    async def initialize(self) -> None:
        return

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return "Error: 'query' parameter is required."

        max_results = clamp_max_results(
            kwargs.get("max_results", self.max_results),
            self.max_results,
        )

        try:
            # Over-fetch when an allowlist is set so local hostname
            # filtering still has enough candidates. Do not rely on
            # ``site:`` operators — engines ignore or fail them.
            fetch_n = HARD_MAX_RESULTS if self.allowed_domains else max_results
            hits = await self.backend.search(query, max_results=fetch_n)
        except Exception as exc:
            return _recoverable_error(query, exc)

        hits = sanitize_hits(
            hits,
            allowed_domains=self.allowed_domains,
            blocked_domains=self.blocked_domains,
        )[:max_results]
        if not hits and self.allowed_domains:
            allowed = ", ".join(self.allowed_domains)
            return (
                f'No web results for "{query}" on allowed sites ({allowed}). '
                "Broaden allowed_domains or try a different query."
            )
        return format_results(query, hits)

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to look up on the web.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum number of results to return "
                            f"(1–{HARD_MAX_RESULTS}). Default: {self.max_results}."
                        ),
                        "default": self.max_results,
                    },
                },
                "required": ["query"],
            },
        }


def _recoverable_error(query: str, exc: Exception) -> str:
    if isinstance(exc, WebSearchHttpError) and exc.status == 429:
        logger.warning("Web search rate-limited query=%r: %s", query, exc)
        return (
            "Error: the web search provider rate-limited this request. "
            "Wait a few seconds and retry with a narrower query."
        )
    name = type(exc).__name__
    msg = str(exc) or name
    if "ratelimit" in name.lower() or "ratelimit" in msg.lower():
        logger.warning("Web search rate-limited query=%r: %s", query, exc)
        return (
            "Error: the web search provider rate-limited this request. "
            "Wait a few seconds and retry with a narrower query."
        )
    logger.warning("Web search failed query=%r: %s", query, exc)
    return f"Error: web search failed ({name}: {msg})."
