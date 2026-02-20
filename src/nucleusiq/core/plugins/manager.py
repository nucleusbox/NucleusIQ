"""
PluginManager — orchestrates the plugin hook pipeline.

Iterates over registered plugins for each hook point, handling the
chain-of-responsibility pattern for ``wrap_model_call`` and
``wrap_tool_call`` using the (request, handler) pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, List, Optional

from nucleusiq.plugins.base import (
    BasePlugin,
    AgentContext,
    ModelRequest,
    ToolRequest,
    ModelHandler,
    ToolHandler,
)
from nucleusiq.plugins.errors import PluginHalt

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages plugin lifecycle and executes hook pipelines.

    Plugins are executed in registration order (first = first executed)
    for node hooks and outermost-first for wrap hooks.
    """

    def __init__(self, plugins: Optional[List[BasePlugin]] = None) -> None:
        self._plugins: List[BasePlugin] = list(plugins or [])
        self._model_call_count: int = 0
        self._tool_call_count: int = 0

    @property
    def plugins(self) -> List[BasePlugin]:
        return self._plugins

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
    # before_agent / after_agent                                          #
    # ------------------------------------------------------------------ #

    async def run_before_agent(self, ctx: AgentContext) -> AgentContext:
        """Execute ``before_agent`` hooks in order.

        Each plugin may return ``None`` (no change) or a modified context.
        Raises ``PluginHalt`` to abort.
        """
        for plugin in self._plugins:
            result = await plugin.before_agent(ctx)
            if result is not None:
                ctx = result
        return ctx

    async def run_after_agent(self, ctx: AgentContext, result: Any) -> Any:
        """Execute ``after_agent`` hooks in order."""
        for plugin in self._plugins:
            result = await plugin.after_agent(ctx, result)
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
            result = await plugin.before_model(request)
            if result is not None:
                request = result
        return request

    async def run_after_model(self, request: ModelRequest, response: Any) -> Any:
        """Execute ``after_model`` hooks in order."""
        for plugin in self._plugins:
            response = await plugin.after_model(request, response)
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

            async def _make_handler(
                p: BasePlugin, nxt: Callable
            ) -> Callable:
                async def h(r: ModelRequest) -> Any:
                    return await p.wrap_model_call(r, nxt)
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

            async def _make_handler(
                p: BasePlugin, nxt: Callable
            ) -> Callable:
                async def h(r: ToolRequest) -> Any:
                    return await p.wrap_tool_call(r, nxt)
                return h

            handler = await _make_handler(plugin, prev)

        return await handler(request)

    def has_plugins(self) -> bool:
        return len(self._plugins) > 0
