"""
StandardMode — Gear 2: Tool-enabled, linear execution.

Logic: Input -> Decision -> Tool Execution -> Result

Use Cases: "Check the weather", "Query database", "Search information"

Characteristics:
- Tool execution enabled
- Linear flow (no loops)
- Fire-and-forget (tries once, returns error if fails)
- Optional memory
- Multiple tool calls supported
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.components.executor import Executor
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.plugins.errors import PluginHalt


class StandardMode(BaseExecutionMode):
    """Gear 2: Standard mode — tool-enabled, linear execution."""

    _DEFAULT_MAX_TOOL_CALLS = 50

    async def run(self, agent: "Agent", task: Task) -> Any:
        """Execute a task with tool-calling loop."""
        agent._logger.debug("Executing in STANDARD mode (tool-enabled, linear)")
        agent.state = AgentState.EXECUTING

        # Fast path: no LLM -> echo
        echo = self.echo_fallback(agent, task)
        if echo is not None:
            return echo

        # Ensure executor is ready
        self._ensure_executor(agent)

        # Convert tools to LLM-specific format
        tool_specs = self._get_tool_specs(agent)

        # Build initial messages
        messages = self.build_messages(agent, task)

        try:
            result = await self._tool_call_loop(agent, task, messages, tool_specs)
            agent._last_messages = messages
            return result
        except PluginHalt:
            raise
        except Exception as e:
            agent._logger.error("Error during standard execution: %s", str(e))
            agent.state = AgentState.ERROR
            return f"Error: Standard execution failed: {str(e)}"

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_executor(self, agent: "Agent") -> None:
        """Lazily create an Executor if the agent does not have one."""
        if not hasattr(agent, "_executor") or agent._executor is None:
            if agent.llm:
                agent._executor = Executor(agent.llm, agent.tools)
            else:
                raise RuntimeError("Cannot execute in standard mode: LLM not available")

    def _get_tool_specs(self, agent: "Agent") -> List[Dict[str, Any]]:
        """Return LLM-formatted tool specifications."""
        if agent.tools and agent.llm:
            return agent.llm.convert_tool_specs(agent.tools)
        return []

    async def _tool_call_loop(
        self,
        agent: "Agent",
        task: Task | Dict[str, Any],
        messages: List[ChatMessage],
        tool_specs: List[Dict[str, Any]],
    ) -> Any:
        """Core tool-calling loop: LLM -> tool -> LLM -> ... -> final answer."""
        tool_call_count = 0
        empty_retries_remaining = 1

        while tool_call_count < self._DEFAULT_MAX_TOOL_CALLS:
            call_kwargs = self.build_call_kwargs(
                agent, messages, tool_specs or None, max_tokens=2048
            )
            response = await self.call_llm(
                agent, call_kwargs, messages, tool_specs or None
            )

            structured = self.handle_structured_output(agent, response)
            if structured is not None:
                return structured

            self.validate_response(response)

            msg = response.choices[0].message
            tool_calls = self._get_tool_calls(msg)
            refusal = self._get_refusal(msg)
            content = self.extract_content(msg)

            if refusal:
                agent.state = AgentState.ERROR
                return f"Error: LLM refused request: {refusal}"

            if tool_calls:
                result = await self._process_tool_calls(
                    agent, msg, tool_calls, messages
                )
                if result is not None:
                    return result
                tool_call_count += len(tool_calls)
                continue

            if content:
                agent.state = AgentState.COMPLETED
                await self._store_in_memory(agent, task, content)
                return content

            if empty_retries_remaining > 0:
                empty_retries_remaining -= 1
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Your last message was empty. You MUST "
                            "either call a tool or provide a final answer."
                        ),
                    )
                )
                continue

            agent._logger.error("LLM returned no tool calls and no content after retry")
            agent.state = AgentState.ERROR
            objective = self.get_objective(task)
            return (
                f"Error: LLM did not respond. Task "
                f"'{objective[:80]}...' may require AUTONOMOUS mode "
                "for multi-step planning."
            )

        agent._logger.warning(
            "Maximum tool calls (%d) reached", self._DEFAULT_MAX_TOOL_CALLS
        )
        agent.state = AgentState.ERROR
        return f"Error: Maximum tool calls ({self._DEFAULT_MAX_TOOL_CALLS}) reached"

    # ------------------------------------------------------------------ #
    # Tool-call extraction helpers                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_tool_calls(msg: Any) -> list | None:
        """Extract tool_calls list from a message (dict or object)."""
        if isinstance(msg, dict):
            calls = msg.get("tool_calls")
        else:
            calls = getattr(msg, "tool_calls", None)
        if calls and isinstance(calls, list) and len(calls) > 0:
            return calls
        return None

    @staticmethod
    def _get_refusal(msg: Any) -> str | None:
        """Extract refusal string from a message (dict or object)."""
        if isinstance(msg, dict):
            return msg.get("refusal")
        return getattr(msg, "refusal", None)

    async def _process_tool_calls(
        self,
        agent: "Agent",
        msg: Any,
        tool_calls: list,
        messages: List[ChatMessage],
    ) -> str | None:
        """Execute tool calls and append results to the message list.

        Returns an error string if any tool fails (fire-and-forget),
        otherwise ``None`` (continue loop).
        """
        raw_content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        parsed_calls = [ToolCallRequest.from_raw(tc) for tc in tool_calls]
        messages.append(
            ChatMessage(
                role="assistant",
                content=raw_content,
                tool_calls=parsed_calls,
            )
        )

        for tc in parsed_calls:
            if not tc.name:
                agent._logger.warning("Tool call missing function name, skipping")
                continue

            agent._logger.info("Tool requested: %s", tc.name)

            try:
                tool_result = await self.call_tool(agent, tc)
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=tc.id,
                        content=json.dumps(tool_result),
                    )
                )
            except Exception as e:
                agent._logger.error("Tool execution failed: %s", e)
                agent.state = AgentState.ERROR
                return f"Error: Tool '{tc.name}' execution failed: {str(e)}"

        return None

    @staticmethod
    def _parse_tool_call(
        tool_call: Any,
    ) -> tuple:
        """Parse a single tool call into ``(id, name, arguments_str)``."""
        if isinstance(tool_call, dict):
            tc_id = tool_call.get("id")
            fn_info = tool_call.get("function", {})
            fn_name = fn_info.get("name") if isinstance(fn_info, dict) else None
            fn_args_str = (
                fn_info.get("arguments", "{}") if isinstance(fn_info, dict) else "{}"
            )
        else:
            tc_id = getattr(tool_call, "id", None)
            fn_info = getattr(tool_call, "function", None)
            fn_name = getattr(fn_info, "name", None) if fn_info else None
            fn_args_str = getattr(fn_info, "arguments", "{}") if fn_info else "{}"
        return tc_id, fn_name, fn_args_str

    async def _store_in_memory(self, agent: "Agent", task: Any, content: str) -> None:
        """Persist result in agent memory."""
        if agent.memory:
            try:
                await agent.memory.aadd_message("assistant", content)
            except Exception as e:
                agent._logger.warning("Failed to store in memory: %s", e)
