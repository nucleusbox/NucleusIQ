"""
Base execution mode interface for NucleusIQ agents.

Each mode implements a distinct execution strategy:
- DirectMode: Fast, simple, no tools (Gear 1)
- StandardMode: Tool-enabled, linear execution (Gear 2)
- AutonomousMode: Full reasoning loop with planning (Gear 3)

New modes can be registered via ``Agent.register_mode()`` without
modifying the Agent class (Open/Closed Principle).
"""

from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.task import Task
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.chat_models import (
    ChatMessage,
    LLMCallKwargs,
    ToolCallRequest,
    messages_to_dicts,
)
from nucleusiq.plugins.base import ModelRequest, ToolRequest
from nucleusiq.plugins.errors import PluginHalt


class BaseExecutionMode(ABC):
    """
    Strategy interface for agent execution modes.

    Every mode receives the ``agent`` instance so it can access
    ``agent.llm``, ``agent.tools``, ``agent.config``, ``agent.memory``,
    ``agent._executor``, ``agent._logger``, and helper methods like
    ``agent._resolve_response_format()``.

    The mode does **not** own state — the Agent does.

    Shared helpers live here so that concrete modes stay DRY.
    """

    @abstractmethod
    async def run(
        self,
        agent: "Agent",
        task: Task,
    ) -> Any:
        """
        Execute a task using this mode's strategy.

        Args:
            agent: The Agent instance (provides access to LLM, tools, config, etc.)
            task: The task to execute

        Returns:
            Execution result
        """
        ...

    # ------------------------------------------------------------------ #
    # Shared helpers (used by DirectMode, StandardMode, etc.)            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_objective(task: Union[Task, Dict[str, Any]]) -> str:
        """Extract the objective string from a Task or dict.

        Accepts both forms for backward compatibility with external callers,
        but internal callers should always pass a ``Task`` instance.
        """
        if isinstance(task, Task):
            return task.objective
        return task.get("objective", "")

    def echo_fallback(
        self, agent: "Agent", task: Union[Task, Dict[str, Any]]
    ) -> Optional[str]:
        """Return an echo result when no LLM is configured, or ``None``."""
        if agent.llm:
            return None
        agent._logger.warning(
            "No LLM configured, falling back to echo mode"
        )
        agent.state = AgentState.COMPLETED
        objective = self.get_objective(task)
        return f"Echo: {objective}"

    def build_messages(
        self,
        agent: "Agent",
        task: Union[Task, Dict[str, Any]],
        plan: Any = None,
    ) -> List[ChatMessage]:
        """Convert task (and optional plan) into an LLM-ready message list.

        When agent has memory, prior conversation turns are injected
        between the system message and the current user message so the
        LLM has full conversational context.
        """
        task_dict = task.to_dict() if isinstance(task, Task) else task
        messages = MessageBuilder.build(
            task_dict,
            plan,
            prompt=agent.prompt,
            role=agent.role,
            objective=agent.objective,
            logger=agent._logger,
        )

        if agent.memory:
            memory_ctx = agent.memory.get_context()
            if memory_ctx:
                current_objective = task_dict.get("objective", "")
                filtered = [
                    m for m in memory_ctx
                    if not (
                        m["role"] == "user"
                        and m["content"] == current_objective
                        and m is memory_ctx[-1]
                    )
                ]
                if filtered:
                    insert_idx = 0
                    for i, m in enumerate(messages):
                        if m.role == "system":
                            insert_idx = i + 1
                        else:
                            break
                    for j, mem_msg in enumerate(filtered):
                        messages.insert(
                            insert_idx + j,
                            ChatMessage.from_dict(mem_msg),
                        )

        return messages

    def build_call_kwargs(
        self,
        agent: "Agent",
        messages: List[ChatMessage],
        tool_specs: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMCallKwargs:
        """Build the kwargs dict for ``agent.llm.call()``.

        Merges model name, messages, tool specs, max_tokens,
        per-execute LLM overrides, and structured-output kwargs.
        """
        output_config = agent._resolve_response_format()
        call_kwargs: Dict[str, Any] = {
            "model": getattr(agent.llm, "model_name", "default"),
            "messages": messages_to_dicts(messages),
            "tools": tool_specs if tool_specs else None,
            "max_tokens": max_tokens or getattr(
                agent.config, "llm_max_tokens", 1024
            ),
        }
        call_kwargs.update(getattr(agent, "_current_llm_overrides", {}))
        call_kwargs.update(
            agent._get_structured_output_kwargs(output_config)
        )
        return call_kwargs

    @staticmethod
    def validate_response(response: Any) -> None:
        """Raise ``ValueError`` if the LLM response is empty/malformed."""
        if (
            not response
            or not hasattr(response, "choices")
            or not response.choices
        ):
            raise ValueError("LLM returned empty response")

    @staticmethod
    def extract_content(msg: Any) -> Optional[str]:
        """Extract and normalise text content from an LLM message.

        Handles:
        - Plain string content
        - List-of-parts format ``[{"type": "text", "text": "..."}]``
        - ``None``
        """
        if isinstance(msg, dict):
            raw = msg.get("content")
        else:
            raw = getattr(msg, "content", None)

        if isinstance(raw, str) and raw.strip():
            return raw
        if isinstance(raw, list):
            parts: List[str] = []
            for part in raw:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
            return "\n".join(parts) if parts else None
        return None

    def handle_structured_output(
        self, agent: "Agent", response: Any
    ) -> Optional[Any]:
        """Return the wrapped structured-output result, or ``None``.

        When a structured-output result is detected the agent state is
        set to COMPLETED.
        """
        output_config = agent._resolve_response_format()
        wrapped = agent._wrap_structured_output_result(response, output_config)
        if isinstance(wrapped, dict) and "output" in wrapped:
            agent.state = AgentState.COMPLETED
            return wrapped
        return None

    # ------------------------------------------------------------------ #
    # Plugin-aware LLM and Tool invocation                               #
    # ------------------------------------------------------------------ #

    async def call_llm(
        self,
        agent: "Agent",
        call_kwargs: Dict[str, Any],
        messages: Optional[List[ChatMessage]] = None,
        tool_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Invoke ``agent.llm.call()`` with the full plugin pipeline.

        Constructs a ``ModelRequest`` and runs:
        before_model -> wrap_model_call chain -> after_model.
        Falls back to a direct call when no plugins are registered.
        """
        pm = getattr(agent, "_plugin_manager", None)

        if pm is None or not pm.has_plugins():
            return await agent.llm.call(**call_kwargs)

        reserved = {"model", "messages", "tools", "max_tokens"}
        extra = {k: v for k, v in call_kwargs.items() if k not in reserved}

        request = ModelRequest(
            model=call_kwargs.get("model", "default"),
            messages=messages or [],
            tools=tool_specs,
            max_tokens=call_kwargs.get("max_tokens", 1024),
            call_count=pm.increment_model_calls(),
            agent_name=agent.name,
            extra_kwargs=extra,
        )

        request = await pm.run_before_model(request)
        response = await pm.execute_model_call(request, agent.llm.call)
        response = await pm.run_after_model(request, response)
        return response

    async def call_tool(
        self,
        agent: "Agent",
        tc: ToolCallRequest,
    ) -> Any:
        """Invoke tool execution with the full plugin pipeline.

        Constructs a ``ToolRequest`` and runs the wrap_tool_call chain.
        Falls back to a direct call when no plugins are registered.
        """
        pm = getattr(agent, "_plugin_manager", None)

        if pm is None or not pm.has_plugins():
            return await agent._executor.execute(tc)

        tool_args: Dict[str, Any] = {}
        try:
            tool_args = json.loads(tc.arguments) if tc.arguments else {}
        except (json.JSONDecodeError, TypeError):
            pass

        request = ToolRequest(
            tool_name=tc.name or "",
            tool_args=tool_args,
            tool_call_id=tc.id,
            call_count=pm.increment_tool_calls(),
            agent_name=agent.name,
        )
        request._tool_call_request = tc

        return await pm.execute_tool_call(request, agent._executor.execute)
