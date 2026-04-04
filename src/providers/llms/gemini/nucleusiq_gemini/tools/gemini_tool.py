"""Gemini-specific native tool factory.

Gemini supports several built-in tools that execute server-side:
1. Google Search — grounding with web search results
2. Code Execution — sandboxed Python execution
3. URL Context   — fetch and understand web page content
4. Google Maps   — location-aware grounding with Google Maps

**API constraint (generateContent):** Google's ``generateContent`` API
rejects requests that mix these built-in tools with function-declaration
tools (``@tool`` callables).  When both types are needed, the framework
transparently enables **proxy mode** — native tools are presented to the
LLM as callable function declarations and executed via a separate API
sub-call.  See :mod:`tool_splitter` for details.

Usage — create via the ``GeminiTool`` factory and pass to an Agent::

    search = GeminiTool.google_search()
    code   = GeminiTool.code_execution()
    maps   = GeminiTool.google_maps()
    agent  = Agent(tools=[search, code, maps], ...)

Mixing native and custom tools is fully supported::

    search = GeminiTool.google_search()
    agent  = Agent(tools=[search, my_custom_tool], ...)
    result = await agent.execute("Search for X then process with my tool")

Tool type identifiers are stored as class-level constants on ``GeminiTool``
so they can be overridden in one place when Google ships new type strings.
"""

from __future__ import annotations

import logging
from typing import Any

from nucleusiq.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

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

    Created exclusively by :class:`GeminiTool` factory methods.

    **Normal mode** (``is_native=True``):
        The tool is sent as a native spec (``google_search``, etc.) and
        executed server-side by Google.  The core ``Executor`` never calls
        ``execute()``; it checks ``is_native`` and skips the tool.

    **Proxy mode** (``is_native=False``, activated by ``_enable_proxy_mode``):
        Used when both native and custom tools are configured on the same
        agent.  The tool appears as a function declaration so the LLM can
        call it like any custom tool.  ``execute()`` makes a separate
        ``generate_content`` sub-call with the *real* native tool to
        retrieve grounded results.

    Proxy mode is activated by ``BaseGemini.convert_tool_specs()`` and
    requires zero changes to core ``Agent``, ``Executor``, or any
    execution mode.
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
        self._proxy_llm: Any | None = None
        self._proxy_model: str | None = None
        self._proxy_spec: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Proxy mode lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def _enable_proxy_mode(self, llm: Any) -> None:
        """Switch to proxy mode for mixed-tool scenarios.

        After this call:
        - ``is_native`` becomes ``False`` so the Executor calls ``execute()``
        - ``get_spec()`` returns a function declaration instead of a native spec
        - ``execute()`` makes a sub-API-call with the real native tool

        Called by ``BaseGemini.convert_tool_specs()`` when mixed tools are
        detected.  This mutates the object in-place; the core ``Executor``
        holds a reference to the same object and sees the change.

        Args:
            llm: The ``BaseGemini`` instance (provides ``_client`` and
                ``model`` for sub-calls).
        """
        from nucleusiq_gemini.tools.tool_splitter import build_proxy_spec

        self._proxy_llm = llm
        self._proxy_model = getattr(llm, "model", None)
        self._proxy_spec = build_proxy_spec(self.tool_type, self.name)
        self.is_native = False
        logger.debug(
            "Proxy mode enabled for native tool '%s' (type=%s)",
            self.name,
            self.tool_type,
        )

    def _disable_proxy_mode(self) -> None:
        """Revert to native mode.

        Called when the tool list changes and mixing is no longer needed.
        """
        self._proxy_llm = None
        self._proxy_model = None
        self._proxy_spec = None
        self.is_native = True

    @property
    def is_proxy_mode(self) -> bool:
        """``True`` when this native tool is operating in proxy mode."""
        return self._proxy_llm is not None

    # ------------------------------------------------------------------ #
    # BaseTool interface                                                   #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        pass

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool.

        In native mode, raises ``NotImplementedError`` (server-side only).
        In proxy mode, makes a ``generate_content`` sub-call with the real
        native tool and returns the text result.
        """
        if self._proxy_llm is None:
            raise NotImplementedError(
                f"Gemini native tool '{self.name}' (type={self.tool_type}) is "
                f"executed server-side by Google. "
                f"It must not be called via Executor.execute()."
            )
        return await self._proxy_execute(**kwargs)

    def get_spec(self) -> dict[str, Any]:
        """Return the tool specification.

        In proxy mode returns a function-declaration spec; otherwise
        returns the native tool spec.
        """
        if self._proxy_spec is not None:
            return self._proxy_spec
        return self.tool_spec

    # ------------------------------------------------------------------ #
    # Proxy execution                                                      #
    # ------------------------------------------------------------------ #

    async def _proxy_execute(self, **kwargs: Any) -> Any:
        """Execute by making a ``generate_content`` call with the real native tool."""
        query = kwargs.get("query") or kwargs.get("task") or kwargs.get("url") or ""
        if not query:
            query = " ".join(str(v) for v in kwargs.values() if v)
        if not query:
            return f"No input provided for {self.tool_type}"

        model = self._proxy_model or "gemini-2.5-flash"
        native_tool_config = {self.tool_type: {}}

        logger.debug(
            "Proxy executing '%s' via native %s (model=%s, query=%s)",
            self.name,
            self.tool_type,
            model,
            query[:120],
        )

        try:
            llm = self._proxy_llm
            client = getattr(llm, "_client", None)
            if client is None:
                return f"Error: proxy LLM has no _client for {self.tool_type}"
            response = await client.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": query}]}],
                config={"tools": [native_tool_config]},
            )
        except Exception as exc:
            logger.error("Proxy sub-call failed for '%s': %s", self.name, exc)
            return f"Error executing {self.tool_type}: {exc}"

        return self._extract_proxy_response(response)

    @staticmethod
    def _extract_proxy_response(response: Any) -> str:
        """Extract text content from a ``generate_content`` response."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "(no results)"

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or [] if content else []

        texts: list[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
            code_result = getattr(part, "code_execution_result", None)
            if code_result:
                output = getattr(code_result, "output", "")
                texts.append(f"[Code output]: {output}")

        return "\n".join(texts) if texts else "(no text in response)"


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
