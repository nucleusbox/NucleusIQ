"""Built-in web-search adapters."""

from nucleusiq.tools.web_search.backends._ddgs import (
    BingSearchBackend,
    DuckDuckGoSearch,
)
from nucleusiq.tools.web_search.backends.brave import BraveSearchBackend
from nucleusiq.tools.web_search.backends.google import GoogleSearchBackend
from nucleusiq.tools.web_search.backends.serper import SerperSearchBackend
from nucleusiq.tools.web_search.backends.tavily import TavilySearchBackend

__all__ = [
    "BingSearchBackend",
    "BraveSearchBackend",
    "DuckDuckGoSearch",
    "GoogleSearchBackend",
    "SerperSearchBackend",
    "TavilySearchBackend",
]
