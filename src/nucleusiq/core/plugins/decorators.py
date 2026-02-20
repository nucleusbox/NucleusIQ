"""
Decorator API for creating NucleusIQ plugins from simple functions.

Each decorator creates a ``BasePlugin`` instance with a single hook
override, allowing concise plugin definitions::

    @before_model
    def log_calls(request: ModelRequest) -> None:
        print(f"LLM call #{request.call_count} to {request.model}")

    @wrap_model_call
    async def retry(request: ModelRequest, handler):
        try:
            return await handler(request)
        except Exception:
            return await handler(request.with_(model="gpt-4o-mini"))

    @wrap_tool_call
    async def approve_tools(request: ToolRequest, handler):
        if request.tool_name in DANGEROUS:
            return "Blocked"
        return await handler(request)

    agent = Agent(..., plugins=[log_calls, retry, approve_tools])
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Optional

from nucleusiq.plugins.base import (
    BasePlugin,
    AgentContext,
    ModelRequest,
    ToolRequest,
    ModelHandler,
    ToolHandler,
)


def _ensure_async(fn: Callable) -> Callable:
    """Wrap a sync function so it can be awaited."""
    if asyncio.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return wrapper


# ------------------------------------------------------------------ #
# Agent-level decorators                                               #
# ------------------------------------------------------------------ #


def before_agent(fn: Callable) -> BasePlugin:
    """Create a plugin with a ``before_agent`` hook.

    The function receives ``(ctx: AgentContext)`` and should return
    ``None`` (no change), an ``AgentContext``, or raise ``PluginHalt``.

    Example::

        @before_agent
        def validate(ctx: AgentContext) -> None:
            if not ctx.task.objective:
                raise PluginHalt("Empty task")
    """
    async_fn = _ensure_async(fn)
    plugin_name = getattr(fn, "__name__", "before_agent_hook")

    class _Plugin(BasePlugin):
        @property
        def name(self) -> str:
            return plugin_name

        async def before_agent(self, ctx: AgentContext) -> Optional[AgentContext]:
            return await async_fn(ctx)

    instance = _Plugin()
    functools.update_wrapper(instance, fn)
    return instance


def after_agent(fn: Callable) -> BasePlugin:
    """Create a plugin with an ``after_agent`` hook.

    The function receives ``(ctx: AgentContext, result: Any)``
    and should return the (possibly modified) result.

    Example::

        @after_agent
        def log_result(ctx: AgentContext, result: Any) -> Any:
            print(f"Result: {result}")
            return result
    """
    async_fn = _ensure_async(fn)
    plugin_name = getattr(fn, "__name__", "after_agent_hook")

    class _Plugin(BasePlugin):
        @property
        def name(self) -> str:
            return plugin_name

        async def after_agent(self, ctx: AgentContext, result: Any) -> Any:
            return await async_fn(ctx, result)

    instance = _Plugin()
    functools.update_wrapper(instance, fn)
    return instance


# ------------------------------------------------------------------ #
# Model-level decorators                                               #
# ------------------------------------------------------------------ #


def before_model(fn: Callable) -> BasePlugin:
    """Create a plugin with a ``before_model`` hook.

    The function receives ``(request: ModelRequest)`` and should return
    ``None`` (no change), a ``ModelRequest`` (via ``.with_()``),
    or raise ``PluginHalt``.

    Example::

        @before_model
        def log(request: ModelRequest) -> None:
            print(f"Call #{request.call_count} to {request.model}")

        @before_model
        def downgrade(request: ModelRequest) -> ModelRequest:
            if request.call_count > 5:
                return request.with_(model="gpt-4o-mini")
    """
    async_fn = _ensure_async(fn)
    plugin_name = getattr(fn, "__name__", "before_model_hook")

    class _Plugin(BasePlugin):
        @property
        def name(self) -> str:
            return plugin_name

        async def before_model(self, request: ModelRequest) -> Optional[ModelRequest]:
            return await async_fn(request)

    instance = _Plugin()
    functools.update_wrapper(instance, fn)
    return instance


def after_model(fn: Callable) -> BasePlugin:
    """Create a plugin with an ``after_model`` hook.

    The function receives ``(request: ModelRequest, response: Any)``
    and should return the (possibly modified) response.

    Example::

        @after_model
        def log_response(request: ModelRequest, response: Any) -> Any:
            print(f"Call #{request.call_count} returned {type(response).__name__}")
            return response
    """
    async_fn = _ensure_async(fn)
    plugin_name = getattr(fn, "__name__", "after_model_hook")

    class _Plugin(BasePlugin):
        @property
        def name(self) -> str:
            return plugin_name

        async def after_model(self, request: ModelRequest, response: Any) -> Any:
            return await async_fn(request, response)

    instance = _Plugin()
    functools.update_wrapper(instance, fn)
    return instance


# ------------------------------------------------------------------ #
# Wrap-style decorators (request, handler)                             #
# ------------------------------------------------------------------ #


def wrap_model_call(fn: Callable) -> BasePlugin:
    """Create a plugin with a ``wrap_model_call`` hook.

    The function receives ``(request: ModelRequest, handler)`` and must
    call ``await handler(request)`` to proceed or return directly to
    short-circuit.

    Example — retry with model fallback::

        @wrap_model_call
        async def retry(request: ModelRequest, handler):
            try:
                return await handler(request)
            except Exception:
                return await handler(request.with_(model="gpt-4o-mini"))
    """
    async_fn = _ensure_async(fn)
    plugin_name = getattr(fn, "__name__", "wrap_model_call_hook")

    class _Plugin(BasePlugin):
        @property
        def name(self) -> str:
            return plugin_name

        async def wrap_model_call(self, request: ModelRequest, handler: ModelHandler) -> Any:
            return await async_fn(request, handler)

    instance = _Plugin()
    functools.update_wrapper(instance, fn)
    return instance


def wrap_tool_call(fn: Callable) -> BasePlugin:
    """Create a plugin with a ``wrap_tool_call`` hook.

    The function receives ``(request: ToolRequest, handler)`` and must
    call ``await handler(request)`` to proceed or return directly to
    short-circuit.

    Example — block dangerous tools::

        @wrap_tool_call
        async def guard(request: ToolRequest, handler):
            if request.tool_name in DANGEROUS:
                return "Tool blocked by policy"
            return await handler(request)
    """
    async_fn = _ensure_async(fn)
    plugin_name = getattr(fn, "__name__", "wrap_tool_call_hook")

    class _Plugin(BasePlugin):
        @property
        def name(self) -> str:
            return plugin_name

        async def wrap_tool_call(self, request: ToolRequest, handler: ToolHandler) -> Any:
            return await async_fn(request, handler)

    instance = _Plugin()
    functools.update_wrapper(instance, fn)
    return instance
