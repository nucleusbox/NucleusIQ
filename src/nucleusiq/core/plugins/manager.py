"""
PluginManager — orchestrates the plugin hook pipeline.

Iterates over registered plugins for each hook point, handling the
chain-of-responsibility pattern for ``wrap_model_call`` and
``wrap_tool_call`` using the (request, handler) pattern.

When a tracer is attached (via ``tracer`` property), every hook
execution records a :class:`PluginEvent` on the tracer for full
observability in the ``AgentResult``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, List

from nucleusiq.agents.agent_result import PluginEvent
from nucleusiq.plugins.base import (
    AgentContext,
    BasePlugin,
    ModelRequest,
    ToolRequest,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages plugin lifecycle and executes hook pipelines.

    Plugins are executed in registration order (first = first executed)
    for node hooks and outermost-first for wrap hooks.
    """

    def __init__(self, plugins: List[BasePlugin] | None = None) -> None:
        self._plugins: List[BasePlugin] = list(plugins or [])
        self._model_call_count: int = 0
        self._tool_call_count: int = 0
        self._tracer: Any | None = None

    @property
    def plugins(self) -> List[BasePlugin]:
        return self._plugins

    @property
    def tracer(self) -> Any | None:
        """The ``ExecutionTracerProtocol`` attached to this manager."""
        return self._tracer

    @tracer.setter
    def tracer(self, value: Any) -> None:
        self._tracer = value

    def reset_counters(self) -> None:
        """Reset per-execution counters (called at start of each execute())."""
        self._model_call_count = 0
        self._tool_call_count = 0

    @property
    def model_call_count(self) -> int:
        return self._model_call_count

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    def increment_model_calls(self) -> int:
        self._model_call_count += 1
        return self._model_call_count

    def increment_tool_calls(self) -> int:
        self._tool_call_count += 1
        return self._tool_call_count

    # ------------------------------------------------------------------ #
    # Plugin event recording                                               #
    # ------------------------------------------------------------------ #

    def _record_plugin_event(
        self,
        plugin: BasePlugin,
        hook: str,
        action: str,
        t0: float,
        detail: str | None = None,
    ) -> None:
        """Record a :class:`PluginEvent` on the tracer if one is attached."""
        if self._tracer is None:
            return
        try:
            self._tracer.record_plugin_event(
                PluginEvent(
                    plugin_name=type(plugin).__name__,
                    hook=hook,
                    action=action,
                    detail=detail,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
            )
        except Exception:
            logger.debug("Failed to record plugin event", exc_info=True)

    # ------------------------------------------------------------------ #
    # before_agent / after_agent                                          #
    # ------------------------------------------------------------------ #

    async def run_before_agent(self, ctx: AgentContext) -> AgentContext:
        """Execute ``before_agent`` hooks in order.

        Each plugin may return ``None`` (no change) or a modified context.
        Raises ``PluginHalt`` to abort.
        """
        for plugin in self._plugins:
            t0 = time.perf_counter()
            result = await plugin.before_agent(ctx)
            action = "modified" if result is not None else "passthrough"
            self._record_plugin_event(plugin, "before_agent", action, t0)
            if result is not None:
                ctx = result
        return ctx

    async def run_after_agent(self, ctx: AgentContext, result: Any) -> Any:
        """Execute ``after_agent`` hooks in order."""
        for plugin in self._plugins:
            t0 = time.perf_counter()
            result = await plugin.after_agent(ctx, result)
            self._record_plugin_event(plugin, "after_agent", "executed", t0)
        return result

    # ------------------------------------------------------------------ #
    # before_model / after_model                                          #
    # ------------------------------------------------------------------ #

    async def run_before_model(self, request: ModelRequest) -> ModelRequest:
        """Execute ``before_model`` hooks in order.

        Each plugin may return ``None`` (no change) or a ``ModelRequest``
        (via ``.with_()``). Raises ``PluginHalt`` to abort.
        """
        for plugin in self._plugins:
            t0 = time.perf_counter()
            result = await plugin.before_model(request)
            action = "modified" if result is not None else "passthrough"
            self._record_plugin_event(plugin, "before_model", action, t0)
            if result is not None:
                request = result
        return request

    async def run_after_model(self, request: ModelRequest, response: Any) -> Any:
        """Execute ``after_model`` hooks in order."""
        for plugin in self._plugins:
            t0 = time.perf_counter()
            response = await plugin.after_model(request, response)
            self._record_plugin_event(plugin, "after_model", "executed", t0)
        return response

    # ------------------------------------------------------------------ #
    # wrap_model_call (chain-of-responsibility, request/handler pattern)  #
    # ------------------------------------------------------------------ #

    async def execute_model_call(
        self,
        request: ModelRequest,
        final_call: Callable[..., Awaitable[Any]],
    ) -> Any:
        """Build and execute the model call chain.

        The innermost handler calls ``final_call(**request.to_call_kwargs())``.
        Each plugin wraps this with its ``wrap_model_call(request, handler)``.

        Args:
            request: ModelRequest with all call parameters
            final_call: The actual ``agent.llm.call`` coroutine factory
        """

        async def innermost(req: ModelRequest) -> Any:
            return await final_call(**req.to_call_kwargs())

        handler: Callable = innermost
        for plugin in reversed(self._plugins):
            prev = handler

            async def _make_handler(p: BasePlugin, nxt: Callable) -> Callable:
                async def h(r: ModelRequest) -> Any:
                    t0 = time.perf_counter()
                    result = await p.wrap_model_call(r, nxt)
                    self._record_plugin_event(p, "wrap_model_call", "wrapped", t0)
                    return result

                return h

            handler = await _make_handler(plugin, prev)

        return await handler(request)

    # ------------------------------------------------------------------ #
    # wrap_tool_call (chain-of-responsibility, request/handler pattern)   #
    # ------------------------------------------------------------------ #

    async def execute_tool_call(
        self,
        request: ToolRequest,
        final_call: Callable[..., Awaitable[Any]],
    ) -> Any:
        """Build and execute the tool call chain.

        The innermost handler reconstructs a ``ToolCallRequest`` from
        the ``ToolRequest`` and calls the executor.

        Args:
            request: ToolRequest describing the tool invocation
            final_call: The actual ``executor.execute(tc)`` coroutine factory
        """

        async def innermost(req: ToolRequest) -> Any:
            tc = req.to_tool_call_request()
            return await final_call(tc)

        handler: Callable = innermost
        for plugin in reversed(self._plugins):
            prev = handler

            async def _make_handler(p: BasePlugin, nxt: Callable) -> Callable:
                async def h(r: ToolRequest) -> Any:
                    t0 = time.perf_counter()
                    result = await p.wrap_tool_call(r, nxt)
                    self._record_plugin_event(p, "wrap_tool_call", "wrapped", t0)
                    return result

                return h

            handler = await _make_handler(plugin, prev)

        return await handler(request)

    def has_plugins(self) -> bool:
        return len(self._plugins) > 0
