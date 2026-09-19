"""Web-search backend contract.

``WebSearchTool`` talks only to this ABC. Built-in adapters (DuckDuckGo,
Google CSE, Bing, Brave, Tavily, Serper) live in
:mod:`nucleusiq.tools.web_search.backends`. Third-party adapters register
with :class:`~nucleusiq.tools.web_search.factory.WebSearchBackendFactory`
or are passed as ``WebSearchTool(backend=...)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from nucleusiq.tools.web_search.types import SearchHit


class WebSearchBackend(ABC):
    """One search engine behind :class:`~nucleusiq.tools.web_search.WebSearchTool`.

    Implementations accept their credentials in ``__init__`` and raise
    :class:`~nucleusiq.tools.errors.ToolValidationError` immediately when
    required configuration is missing — do not defer that to ``search()``.
    """

    name: ClassVar[str]

    @abstractmethod
    async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        """Return normalized hits for *query*."""
