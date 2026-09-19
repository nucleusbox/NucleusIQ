"""Registry for web-search adapters.

Built-ins are registered here. External adapters call
``WebSearchBackendFactory.register("myengine", MyBackend)``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from nucleusiq.tools.errors import ToolValidationError
from nucleusiq.tools.web_search.backends._ddgs import (
    BingSearchBackend,
    DuckDuckGoSearch,
)
from nucleusiq.tools.web_search.backends.brave import BraveSearchBackend
from nucleusiq.tools.web_search.backends.google import GoogleSearchBackend
from nucleusiq.tools.web_search.backends.serper import SerperSearchBackend
from nucleusiq.tools.web_search.backends.tavily import TavilySearchBackend
from nucleusiq.tools.web_search.protocol import WebSearchBackend

_ALIASES = {
    "ddg": "duckduckgo",
    "duckduckgo-search": "duckduckgo",
    "google_cse": "google",
    "google-cse": "google",
}


class WebSearchProvider(str, Enum):
    """Built-in search adapters."""

    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"
    BRAVE = "brave"
    TAVILY = "tavily"
    SERPER = "serper"


class WebSearchBackendFactory:
    """Create a :class:`WebSearchBackend` by provider name."""

    _registry: dict[str, type[WebSearchBackend]] = {
        WebSearchProvider.DUCKDUCKGO.value: DuckDuckGoSearch,
        WebSearchProvider.GOOGLE.value: GoogleSearchBackend,
        WebSearchProvider.BING.value: BingSearchBackend,
        WebSearchProvider.BRAVE.value: BraveSearchBackend,
        WebSearchProvider.TAVILY.value: TavilySearchBackend,
        WebSearchProvider.SERPER.value: SerperSearchBackend,
    }

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def register(cls, name: str, backend_cls: type[WebSearchBackend]) -> None:
        key = name.strip().lower()
        if not key:
            raise ToolValidationError(
                "Web-search provider name must be a non-empty string.",
                tool_name="web_search",
            )
        if key in cls._registry:
            raise ToolValidationError(
                f"Web-search provider '{key}' is already registered.",
                tool_name="web_search",
            )
        cls._registry[key] = backend_cls

    @classmethod
    def create(
        cls,
        provider: str | WebSearchProvider = WebSearchProvider.DUCKDUCKGO,
        **config: Any,
    ) -> WebSearchBackend:
        key = _normalize_provider(provider)
        backend_cls = cls._registry.get(key)
        if backend_cls is None:
            known = ", ".join(cls.available())
            raise ToolValidationError(
                f"Unknown web-search provider '{key}'. "
                f"Built-in: {known}. "
                "Register a custom adapter with "
                "WebSearchBackendFactory.register(...) "
                "or pass backend=... to WebSearchTool.",
                tool_name="web_search",
            )
        return backend_cls(**config)


def _normalize_provider(provider: str | WebSearchProvider) -> str:
    raw = provider.value if isinstance(provider, WebSearchProvider) else str(provider)
    key = raw.strip().lower()
    return _ALIASES.get(key, key)
