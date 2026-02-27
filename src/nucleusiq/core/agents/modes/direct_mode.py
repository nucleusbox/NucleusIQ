"""
DirectMode — Gear 1: Fast, simple, optional tools.

Logic: Input -> LLM (with optional tools, max 5 calls) -> Output

Use Cases: Chatbots, creative writing, quick lookups, simple Q&A

Characteristics:
- Minimal overhead (usually 1-2 LLM calls)
- Optional tool access (up to 5 tool calls by default)
- No retry logic on empty responses
"""

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.components.executor import Executor
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.streaming.events import StreamEvent


class DirectMode(BaseExecutionMode):
    """Gear 1: Direct mode — fast, optional tools (max 5 by default)."""

    async def run(self, agent: "Agent", task: Task) -> Any:
        """Execute a task with minimal LLM calls and optional tool support."""
        agent._logger.debug("Executing in DIRECT mode (fast, optional tools)")
        agent.state = AgentState.EXECUTING

        echo = self.echo_fallback(agent, task)
        if echo is not None:
            return echo

        has_tools = bool(agent.tools and agent.llm)

        try:
            messages = self.build_messages(agent, task)
            tool_specs = self._get_tool_specs(agent) if has_tools else None

            call_kwargs = self.build_call_kwargs(
                agent, messages, tool_specs, max_tokens=1024
            )
            response = await self.call_llm(agent, call_kwargs, messages, tool_specs)

            structured = self.handle_structured_output(agent, response)
            if structured is not None:
                return structured

            self.validate_response(response)

            msg = response.choices[0].message
            tool_calls = self._get_tool_calls(msg)
            content = self.extract_content(msg)

            if tool_calls and has_tools:
                return await self._handle_tool_calls(
                    agent, task, messages, tool_specs, msg, tool_calls
                )

            if content:
                agent.state = AgentState.COMPLETED
                return content

            agent._logger.warning(
                "LLM returned no content in DIRECT mode (task may require tools)"
            )
            agent.state = AgentState.COMPLETED
            objective = self.get_objective(task)
            return (
                f"No response from LLM. The task '{objective[:100]}...' "
                "may require a higher execution mode. "
                "Try using STANDARD or AUTONOMOUS execution mode."
            )

        except PluginHalt:
            raise
        except Exception as e:
            agent._logger.error("Error during direct execution: %s", str(e))
            agent.state = AgentState.ERROR
            return f"Echo: {self.get_objective(task)}"

    # ------------------------------------------------------------------ #
    # Streaming                                                           #
    # ------------------------------------------------------------------ #

    async def run_stream(
        self, agent: "Agent", task: Task
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a Direct mode execution.

        Delegates to the shared ``_streaming_tool_call_loop`` with
        the mode's own ``max_tool_calls`` limit (default 5).
        """
        agent._logger.debug("Streaming in DIRECT mode (fast, optional tools)")
        agent.state = AgentState.EXECUTING

        echo = self.echo_fallback(agent, task)
        if echo is not None:
            yield StreamEvent.complete_event(echo)
            return

        has_tools = bool(agent.tools and agent.llm)
        if has_tools:
            self._ensure_executor(agent)
        tool_specs = self._get_tool_specs(agent) if has_tools else None
        messages = self.build_messages(agent, task)
        max_tool_calls = agent.config.get_effective_max_tool_calls()

        try:
            async for event in self._streaming_tool_call_loop(
                agent,
                messages,
                tool_specs,
                max_tool_calls=max_tool_calls,
                max_tokens=1024,
            ):
                yield event

            agent.state = AgentState.COMPLETED
        except PluginHalt:
            raise
        except Exception as e:
            agent._logger.error("Streaming error in direct mode: %s", e)
            agent.state = AgentState.ERROR
            yield StreamEvent.error_event(str(e))

    # ------------------------------------------------------------------ #
    # Tool helpers                                                        #
    # ------------------------------------------------------------------ #

    async def _handle_tool_calls(
        self,
        agent: "Agent",
        task: Task,
        messages: List[ChatMessage],
        tool_specs: List[Dict[str, Any]] | None,
        msg: Any,
        tool_calls: list,
    ) -> Any:
        """Execute tool calls and make a single follow-up LLM call."""
        self._ensure_executor(agent)

        max_tool_calls = agent.config.get_effective_max_tool_calls()
        total_calls = 0

        raw_content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        parsed_calls = [ToolCallRequest.from_raw(tc) for tc in tool_calls]
        messages.append(
            ChatMessage(role="assistant", content=raw_content, tool_calls=parsed_calls)
        )

        for tc in parsed_calls:
            if total_calls >= max_tool_calls:
                agent._logger.warning(
                    "Direct mode tool call limit (%d) reached", max_tool_calls
                )
                break
            if not tc.name:
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
                total_calls += 1
            except Exception as e:
                agent._logger.error("Tool execution failed: %s", e)
                agent.state = AgentState.ERROR
                return f"Error: Tool '{tc.name}' execution failed: {str(e)}"

        call_kwargs = self.build_call_kwargs(
            agent, messages, tool_specs, max_tokens=1024
        )
        response = await self.call_llm(agent, call_kwargs, messages, tool_specs)

        structured = self.handle_structured_output(agent, response)
        if structured is not None:
            return structured

        self.validate_response(response)
        content = self.extract_content(response.choices[0].message)

        agent.state = AgentState.COMPLETED
        return content or f"Echo: {self.get_objective(task)}"

    @staticmethod
    def _get_tool_calls(msg: Any) -> list | None:
        if isinstance(msg, dict):
            calls = msg.get("tool_calls")
        else:
            calls = getattr(msg, "tool_calls", None)
        if calls and isinstance(calls, list) and len(calls) > 0:
            return calls
        return None

    @staticmethod
    def _get_tool_specs(agent: "Agent") -> List[Dict[str, Any]]:
        if agent.tools and agent.llm:
            return agent.llm.convert_tool_specs(agent.tools)
        return []

    @staticmethod
    def _ensure_executor(agent: "Agent") -> None:
        if not hasattr(agent, "_executor") or agent._executor is None:
            if agent.llm:
                agent._executor = Executor(agent.llm, agent.tools)
            else:
                raise RuntimeError("Cannot execute tools: LLM not available")
