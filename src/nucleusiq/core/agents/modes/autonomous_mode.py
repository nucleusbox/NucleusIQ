"""
AutonomousMode — Gear 3: Full reasoning loop with planning and self-correction.

Logic: Input → Plan → Execute Plan → Self-Correct → Result

Use Cases: Complex multi-step tasks, research, analysis, problem-solving

Characteristics:
- Automatic planning (calls plan() internally)
- Multi-step execution following the plan
- Context building across steps
- Self-correction capabilities (future enhancement)
- Memory enabled for context retention
- Iterative refinement (future enhancement)
"""

import asyncio
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.planning.planner import Planner


class AutonomousMode(BaseExecutionMode):
    """Gear 3: Autonomous mode — planning + execution loop."""

    def __init__(
        self,
        fallback_mode: Optional[BaseExecutionMode] = None,
    ) -> None:
        self._fallback = fallback_mode

    def _get_fallback(self) -> BaseExecutionMode:
        if self._fallback is not None:
            return self._fallback
        from nucleusiq.agents.modes.standard_mode import StandardMode
        return StandardMode()

    async def run(self, agent: "Agent", task: Task) -> Any:
        """Execute a task with automatic planning and multi-step execution."""
        agent._logger.debug(
            "Executing in AUTONOMOUS mode (planning + execution)"
        )
        agent.state = AgentState.PLANNING

        # Check if LLM is available (required for planning)
        if not agent.llm:
            agent._logger.warning(
                "No LLM configured for autonomous mode, "
                "falling back to standard mode"
            )
            return await self._get_fallback().run(agent, task)

        try:
            # Step 1: Generate plan automatically
            agent._logger.info(
                "Autonomous mode: Generating execution plan..."
            )

            # Get timeout from config
            planning_timeout = getattr(
                agent.config, "planning_timeout", 120
            )
            max_retries = getattr(agent.config, "max_retries", 3)

            planner = Planner(agent)
            context = await planner.get_context(task)

            # Try LLM-based planning with timeout and retry
            plan = None
            last_error = None
            for attempt in range(max_retries):
                try:
                    plan = await asyncio.wait_for(
                        planner.create_plan(task, context),
                        timeout=planning_timeout,
                    )
                    agent._logger.info(
                        "Generated plan with %d steps using LLM", len(plan)
                    )
                    break
                except asyncio.TimeoutError:
                    last_error = (
                        f"Planning timed out after {planning_timeout}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    agent._logger.warning(last_error)
                    if attempt < max_retries - 1:
                        agent._logger.info(
                            "Retrying planning (attempt %d/%d)...",
                            attempt + 2,
                            max_retries,
                        )
                        await asyncio.sleep(1)
                except Exception as e:
                    last_error = str(e)
                    agent._logger.warning(
                        "LLM planning failed (attempt %d): %s",
                        attempt + 1,
                        e,
                    )
                    break  # Don't retry on non-timeout errors

            # Fallback to basic plan if LLM planning failed
            if plan is None:
                agent._logger.warning(
                    "LLM planning failed after retries: %s. "
                    "Falling back to basic plan.",
                    last_error,
                )
                plan = await agent.plan(task)

            # Log plan details
            if len(plan.steps) > 1:
                agent._logger.info("Multi-step plan generated:")
                for step in plan.steps:
                    agent._logger.debug(
                        "  Step %d: %s%s",
                        step.step,
                        step.action,
                        f" - {step.details}" if step.details else "",
                    )
            else:
                agent._logger.debug("Single-step plan (direct execution)")

            # Step 2: Execute the plan
            agent._logger.info("Autonomous mode: Executing plan...")
            result = await planner.execute_plan(task, plan)

            # Check if plan execution failed
            if isinstance(result, str) and result.strip().startswith(
                "Error:"
            ):
                # State was already set to ERROR by execute_plan
                return result

            # Step 3: Store in memory
            if agent.memory:
                try:
                    await agent.memory.aadd_message("assistant", result)
                except Exception as e:
                    agent._logger.warning(
                        "Failed to store in memory: %s", e
                    )

            # Step 4: Wrap result with structured output if configured
            output_config = agent._resolve_response_format()
            if output_config is not None:
                # For AUTONOMOUS mode, the result is already computed by
                # tool execution.  We wrap it in structured output format
                # for consistency.
                agent.state = AgentState.COMPLETED
                return {
                    "output": result,
                    "schema": (
                        output_config.schema_name
                        if hasattr(output_config, "schema_name")
                        else "Result"
                    ),
                    "mode": "autonomous",
                }

            agent.state = AgentState.COMPLETED
            return result

        except Exception as e:
            agent._logger.error(
                "Error during autonomous execution: %s", str(e)
            )
            agent.state = AgentState.ERROR
            # Fallback to standard mode on error
            agent._logger.warning(
                "Falling back to standard mode due to error"
            )
            return await self._get_fallback().run(agent, task)
