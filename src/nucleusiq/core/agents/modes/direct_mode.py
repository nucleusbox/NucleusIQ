"""
DirectMode — Gear 1: Fast, simple, no tools.

Logic: Input -> LLM -> Output

Use Cases: Chatbots, creative writing, simple explanations

Characteristics:
- Near-zero overhead
- No tool execution
- No planning
- Single LLM call
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.plugins.errors import PluginHalt


class DirectMode(BaseExecutionMode):
    """Gear 1: Direct mode — fast, simple, no tools."""

    async def run(self, agent: "Agent", task: Task) -> Any:
        """Execute a task with a single LLM call (no tools, no planning)."""
        agent._logger.debug("Executing in DIRECT mode (fast, no tools)")
        agent.state = AgentState.EXECUTING

        # Fast path: no LLM → echo
        echo = self.echo_fallback(agent, task)
        if echo is not None:
            return echo

        try:
            messages = self.build_messages(agent, task)
            call_kwargs = self.build_call_kwargs(agent, messages)
            response = await self.call_llm(agent, call_kwargs, messages)

            # Structured output short-circuit
            structured = self.handle_structured_output(agent, response)
            if structured is not None:
                return structured

            self.validate_response(response)

            content = self.extract_content(response.choices[0].message)
            if content:
                agent.state = AgentState.COMPLETED
                return content

            # Model returned empty content — task may need tools/planning
            agent._logger.warning(
                "LLM returned no content in DIRECT mode (task may require tools)"
            )
            agent.state = AgentState.COMPLETED
            objective = self.get_objective(task)
            return (
                f"No response from LLM. The task '{objective[:100]}...' "
                "may require tools or planning. "
                "Try using STANDARD or AUTONOMOUS execution mode."
            )

        except PluginHalt:
            raise
        except Exception as e:
            agent._logger.error("Error during direct execution: %s", str(e))
            agent.state = AgentState.ERROR
            return f"Echo: {self.get_objective(task)}"
