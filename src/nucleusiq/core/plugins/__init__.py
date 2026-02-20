"""
NucleusIQ Plugin System

Provides a plugin pipeline for intercepting agent execution at 6 hook
points: before_agent, after_agent, before_model, after_model,
wrap_model_call, and wrap_tool_call.

Two APIs:
  - **Class-based**: Subclass ``BasePlugin`` for multi-hook plugins
  - **Decorator-based**: Use ``@before_model`` etc. for simple hooks

Request models use the **immutable override** pattern — call ``.with_()``
to create modified copies::

    request.with_(model="gpt-4o-mini", max_tokens=2048)

Example (decorator)::

    from nucleusiq.plugins import before_model, wrap_tool_call, ModelRequest

    @before_model
    def log(request: ModelRequest) -> None:
        print(f"LLM call #{request.call_count} to {request.model}")

    agent = Agent(..., plugins=[log])

Example (class-based)::

    from nucleusiq.plugins import BasePlugin, ModelRequest

    class RetryPlugin(BasePlugin):
        async def wrap_model_call(self, request, handler):
            try:
                return await handler(request)
            except Exception:
                return await handler(request.with_(model="gpt-4o-mini"))
"""

from nucleusiq.plugins.base import (
    BasePlugin,
    AgentContext,
    ModelRequest,
    ToolRequest,
    ModelHandler,
    ToolHandler,
    # Backward compat aliases
    ModelCallContext,
    ToolCallContext,
    CallNext,
)
from nucleusiq.plugins.errors import PluginHalt, PluginError
from nucleusiq.plugins.manager import PluginManager
from nucleusiq.plugins.decorators import (
    before_agent,
    after_agent,
    before_model,
    after_model,
    wrap_model_call,
    wrap_tool_call,
)

__all__ = [
    # Core
    "BasePlugin",
    "AgentContext",
    "ModelRequest",
    "ToolRequest",
    "ModelHandler",
    "ToolHandler",
    # Errors
    "PluginHalt",
    "PluginError",
    # Manager
    "PluginManager",
    # Decorators
    "before_agent",
    "after_agent",
    "before_model",
    "after_model",
    "wrap_model_call",
    "wrap_tool_call",
    # Backward compat
    "ModelCallContext",
    "ToolCallContext",
    "CallNext",
]
