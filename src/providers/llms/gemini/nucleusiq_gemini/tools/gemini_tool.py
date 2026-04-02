"""Gemini-specific native tool factory.

Gemini supports several built-in tools that execute server-side:
1. Google Search — grounding with web search results
2. Code Execution — sandboxed Python execution
3. URL Context   — fetch and understand web page content
4. Google Maps   — location-aware grounding with Google Maps

**API constraint:** Google's API rejects requests that mix these **built-in** tools with
**function-declaration** tools (e.g. ``@tool`` callables) in the same call. Use native tools
in an agent that has **only** native tools, or use **only** custom tools — not both.

Usage — create via the ``GeminiTool`` factory and pass to an Agent::

    search = GeminiTool.google_search()
    code   = GeminiTool.code_execution()
    maps   = GeminiTool.google_maps()
    agent  = Agent(tools=[search, code, maps], ...)

Tool type identifiers are stored as class-level constants on ``GeminiTool``
so they can be overridden in one place when Google ships new type strings.
"""

from __future__ import annotations

from typing import Any

from nucleusiq.tools.base_tool import BaseTool

NATIVE_TOOL_TYPES: frozenset[str] = frozenset(
    {
        "google_search",
        "code_execution",
        "url_context",
        "google_maps",
    }
)


class _GeminiNativeTool(BaseTool):
    """Internal wrapper around a Gemini native tool specification.

    Created exclusively by :class:`GeminiTool` factory methods.  Native tools
    are executed **server-side** by Google — the local ``Executor`` never
    calls ``execute()`` on them.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        tool_type: str,
        tool_spec: dict[str, Any],
    ):
        super().__init__(name=name, description=description, version=None)
        self.tool_type = tool_type
        self.tool_spec = tool_spec
        self.is_native = True

    async def initialize(self) -> None:
        pass

    async def execute(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Gemini native tool '{self.name}' (type={self.tool_type}) is "
            f"executed server-side by Google. "
            f"It must not be called via Executor.execute()."
        )

    def get_spec(self) -> dict[str, Any]:
        return self.tool_spec


class GeminiTool:
    """Factory class for creating Gemini-specific native tools.

    **Adaptive design** — tool type identifiers live as class attributes
    so they act as a single source of truth.

    Usage::

        google_search  = GeminiTool.google_search()
        code_execution = GeminiTool.code_execution()
        url_context    = GeminiTool.url_context()

        agent = Agent(tools=[google_search, code_execution], ...)
    """

    GOOGLE_SEARCH_TYPE: str = "google_search"
    CODE_EXECUTION_TYPE: str = "code_execution"
    URL_CONTEXT_TYPE: str = "url_context"
    GOOGLE_MAPS_TYPE: str = "google_maps"

    @staticmethod
    def google_search(
        *,
        dynamic_retrieval_config: dict[str, Any] | None = None,
    ) -> BaseTool:
        """Create Gemini's Google Search grounding tool.

        Args:
            dynamic_retrieval_config: Optional config for dynamic retrieval
                threshold (e.g. ``{"mode": "MODE_DYNAMIC", "dynamic_threshold": 0.5}``).

        Returns:
            BaseTool instance for Google Search grounding.
        """
        spec: dict[str, Any] = {"type": GeminiTool.GOOGLE_SEARCH_TYPE}
        inner: dict[str, Any] = {}
        if dynamic_retrieval_config:
            inner["dynamic_retrieval_config"] = dynamic_retrieval_config
        spec["google_search"] = inner

        return _GeminiNativeTool(
            name="google_search",
            description="Ground responses with Google Search results",
            tool_type=GeminiTool.GOOGLE_SEARCH_TYPE,
            tool_spec=spec,
        )

    @staticmethod
    def code_execution() -> BaseTool:
        """Create Gemini's code execution tool.

        Returns:
            BaseTool instance for sandboxed Python execution.
        """
        return _GeminiNativeTool(
            name="code_execution",
            description="Execute Python code in a secure sandbox",
            tool_type=GeminiTool.CODE_EXECUTION_TYPE,
            tool_spec={"type": GeminiTool.CODE_EXECUTION_TYPE, "code_execution": {}},
        )

    @staticmethod
    def url_context() -> BaseTool:
        """Create Gemini's URL context tool.

        Returns:
            BaseTool instance for fetching and understanding web pages.
        """
        return _GeminiNativeTool(
            name="url_context",
            description="Fetch and understand web page content from URLs",
            tool_type=GeminiTool.URL_CONTEXT_TYPE,
            tool_spec={"type": GeminiTool.URL_CONTEXT_TYPE, "url_context": {}},
        )

    @staticmethod
    def google_maps() -> BaseTool:
        """Create Gemini's Google Maps grounding tool.

        Enables location-aware responses grounded with Google Maps data
        including places, directions, and geographic information.

        Returns:
            BaseTool instance for Google Maps grounding.
        """
        return _GeminiNativeTool(
            name="google_maps",
            description="Ground responses with Google Maps location data",
            tool_type=GeminiTool.GOOGLE_MAPS_TYPE,
            tool_spec={"type": GeminiTool.GOOGLE_MAPS_TYPE, "google_maps": {}},
        )
