"""
Plugin base classes and typed request models for NucleusIQ.

Core models:
    ``ModelRequest``  — immutable request for model hooks, with ``.with_()``
    ``ToolRequest``   — immutable request for tool hooks, with ``.with_()``
    ``AgentContext``   — context for agent-level hooks

Base class:
    ``BasePlugin``    — subclass to create multi-hook plugins

Example (class-based)::

    class RetryPlugin(BasePlugin):
        async def wrap_model_call(self, request, handler):
            for attempt in range(3):
                try:
                    return await handler(request)
                except Exception:
                    if attempt == 2:
                        raise
                    request = request.with_(metadata={**request.metadata, "retry": attempt})

Example (decorator)::

    @before_model
    def log_calls(request: ModelRequest) -> None:
        print(f"LLM call #{request.call_count} to {request.model}")
"""

from __future__ import annotations

import json
from abc import ABC
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from nucleusiq.agents.task import Task
    from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
    from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
    from nucleusiq.memory.base import BaseMemory


# ------------------------------------------------------------------ #
# Handler protocols                                                    #
# ------------------------------------------------------------------ #


class ModelHandler(Protocol):
    """Typed callable for wrap_model_call handlers."""

    async def __call__(self, request: "ModelRequest") -> Any: ...


class ToolHandler(Protocol):
    """Typed callable for wrap_tool_call handlers."""

    async def __call__(self, request: "ToolRequest") -> Any: ...


# ------------------------------------------------------------------ #
# Request models (immutable, Pydantic-first)                           #
# ------------------------------------------------------------------ #


class AgentContext(BaseModel):
    """Context for ``before_agent`` / ``after_agent`` hooks.

    Carries high-level information about the agent execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_name: str
    task: Any = Field(description="Task instance being executed")
    state: Any = Field(description="Current AgentState enum value")
    config: Any = Field(description="AgentConfig instance")
    memory: Optional[Any] = Field(default=None, description="BaseMemory instance if configured")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mutable dict for plugins to share data",
    )


class ModelRequest(BaseModel):
    """Immutable request object for all model-level hooks.

    Passed to ``before_model``, ``after_model``, and ``wrap_model_call``.
    Use ``.with_()`` to create a modified copy — the original is unchanged.

    Example::

        # Change the model for this call
        modified = request.with_(model="gpt-4o-mini")

        # Add metadata
        modified = request.with_(metadata={**request.metadata, "attempt": 2})

        # Override max_tokens
        modified = request.with_(max_tokens=4096)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "default"
    messages: List[Any] = Field(default_factory=list, description="ChatMessage list to be sent")
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="LLM-formatted tool specifications"
    )
    max_tokens: int = 1024
    call_count: int = Field(default=0, description="Number of LLM calls so far in this execution")
    agent_name: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    extra_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional LLM call kwargs (structured output, LLM params, etc.)",
    )

    def with_(self, **updates: Any) -> "ModelRequest":
        """Create an immutable copy with specific fields replaced.

        Uses Pydantic's native ``model_copy(update=...)`` under the hood,
        giving you validation on the new values for free.

        Example::

            new_req = request.with_(model="gpt-4o-mini", max_tokens=2048)
        """
        return self.model_copy(update=updates)

    def to_call_kwargs(self) -> Dict[str, Any]:
        """Serialize to the kwargs dict expected by ``BaseLLM.call()``.

        Merges the typed fields with ``extra_kwargs`` so the LLM receives
        everything it needs in a single dict.
        """
        from nucleusiq.agents.chat_models import messages_to_dicts, ChatMessage

        raw_msgs = []
        for m in self.messages:
            if isinstance(m, ChatMessage):
                raw_msgs.append(m.to_dict())
            elif isinstance(m, dict):
                raw_msgs.append(m)
            else:
                raw_msgs.append({"role": "user", "content": str(m)})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": raw_msgs,
            "tools": self.tools,
            "max_tokens": self.max_tokens,
        }
        kwargs.update(self.extra_kwargs)
        return kwargs


class ToolRequest(BaseModel):
    """Immutable request object for tool-level hooks.

    Passed to ``wrap_tool_call``.  Use ``.with_()`` for modifications.

    Example::

        # Override tool arguments
        modified = request.with_(tool_args={"query": "modified search"})
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None
    call_count: int = Field(default=0, description="Number of tool calls so far in this execution")
    agent_name: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    _tool_call_request: Any = PrivateAttr(default=None)

    def with_(self, **updates: Any) -> "ToolRequest":
        """Create an immutable copy with specific fields replaced."""
        new = self.model_copy(update=updates)
        new._tool_call_request = self._tool_call_request
        return new

    def to_tool_call_request(self) -> Any:
        """Reconstruct a ``ToolCallRequest`` from this request's fields."""
        from nucleusiq.agents.chat_models import ToolCallRequest

        return ToolCallRequest(
            id=self.tool_call_id,
            name=self.tool_name,
            arguments=json.dumps(self.tool_args),
        )


# Backward compat aliases
ModelCallContext = ModelRequest
ToolCallContext = ToolRequest

# Legacy type alias
CallNext = Callable[..., Awaitable[Any]]


# ------------------------------------------------------------------ #
# BasePlugin                                                           #
# ------------------------------------------------------------------ #


class BasePlugin(ABC):
    """Abstract base class for NucleusIQ plugins.

    Override any hook methods you need. Methods not overridden use the
    default pass-through implementation.

    **Node hooks** (before/after) — Return ``None`` for "no change" or
    a modified object to replace the current value::

        class LogPlugin(BasePlugin):
            async def before_model(self, request: ModelRequest) -> None:
                print(f"Call #{request.call_count} to {request.model}")
                # return None implicitly — no changes

    **Wrap hooks** — Receive ``(request, handler)`` and control execution::

        class RetryPlugin(BasePlugin):
            async def wrap_model_call(self, request: ModelRequest, handler: ModelHandler) -> Any:
                try:
                    return await handler(request)
                except Exception:
                    return await handler(request.with_(model="gpt-4o-mini"))

    The ``name`` property defaults to the class name. Override for custom naming.
    """

    @property
    def name(self) -> str:
        """Plugin identifier. Defaults to the class name."""
        return self.__class__.__name__

    # ------------------------------------------------------------------ #
    # Agent-level hooks                                                    #
    # ------------------------------------------------------------------ #

    async def before_agent(self, ctx: AgentContext) -> Optional[AgentContext]:
        """Called before the agent dispatches to an execution mode.

        Return ``None`` to leave the context unchanged, or a modified
        ``AgentContext`` to change behavior. Raise ``PluginHalt`` to
        abort execution with a result.
        """
        return None

    async def after_agent(self, ctx: AgentContext, result: Any) -> Any:
        """Called after the execution mode returns.

        Return the (possibly modified) result.
        """
        return result

    # ------------------------------------------------------------------ #
    # Model-level hooks                                                    #
    # ------------------------------------------------------------------ #

    async def before_model(self, request: ModelRequest) -> Optional[ModelRequest]:
        """Called before each LLM call.

        Return ``None`` for no change, a ``ModelRequest`` (via ``.with_()``)
        to modify the call, or raise ``PluginHalt`` to abort.
        """
        return None

    async def after_model(self, request: ModelRequest, response: Any) -> Any:
        """Called after each LLM call with the raw response.

        Return the (possibly modified) response.
        """
        return response

    async def wrap_model_call(
        self, request: ModelRequest, handler: ModelHandler
    ) -> Any:
        """Wrap the LLM call (chain-of-responsibility).

        Call ``await handler(request)`` to proceed to the next plugin
        or the actual LLM. Return directly to short-circuit.

        Example — retry with fallback model::

            try:
                return await handler(request)
            except Exception:
                return await handler(request.with_(model="gpt-4o-mini"))
        """
        return await handler(request)

    # ------------------------------------------------------------------ #
    # Tool-level hooks                                                     #
    # ------------------------------------------------------------------ #

    async def wrap_tool_call(
        self, request: ToolRequest, handler: ToolHandler
    ) -> Any:
        """Wrap a tool execution (chain-of-responsibility).

        Call ``await handler(request)`` to proceed or return directly
        to short-circuit.
        """
        return await handler(request)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r})>"
