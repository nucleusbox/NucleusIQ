"""First-class web search: one tool, swappable backends.

Default::

    from nucleusiq.tools import WebSearchTool

    agent = Agent(llm=llm, tools=[WebSearchTool()])  # DuckDuckGo, no key

Paid / official engines require their own configuration::

    WebSearchTool(provider="google", api_key="...", search_engine_id="...")
    WebSearchTool(provider="brave", api_key="...")
    WebSearchTool(provider="tavily", api_key="...")
    WebSearchTool(provider="serper", api_key="...")
    WebSearchTool(provider="bing")  # free Bing index via ddgs; no official API
"""

from nucleusiq.tools.web_search.backends import (
    BingSearchBackend,
    BraveSearchBackend,
    DuckDuckGoSearch,
    GoogleSearchBackend,
    SerperSearchBackend,
    TavilySearchBackend,
)
from nucleusiq.tools.web_search.factory import (
    WebSearchBackendFactory,
    WebSearchProvider,
)
from nucleusiq.tools.web_search.protocol import WebSearchBackend
from nucleusiq.tools.web_search.tool import WebSearchTool
from nucleusiq.tools.web_search.types import SearchHit

__all__ = [
    "BingSearchBackend",
    "BraveSearchBackend",
    "DuckDuckGoSearch",
    "GoogleSearchBackend",
    "SearchHit",
    "SerperSearchBackend",
    "TavilySearchBackend",
    "WebSearchBackend",
    "WebSearchBackendFactory",
    "WebSearchProvider",
    "WebSearchTool",
]
