"""
AutonomousMode — Gear 3: Structured orchestration over Standard mode.

Adds capabilities Standard mode cannot provide alone:
- **Parallel execution** via isolated sub-agents (for independent sub-tasks)
- **Plugin-based validation** with structured retry
- **Progress tracking** per step
- **Context curation** — each sub-agent sees only what it needs

Task routing (via Decomposer's 3-gate checklist):

**Simple tasks** → Standard mode + validate + retry
**Parallel tasks** → Decompose → parallel Standard agents → synthesize + validate + retry

Validation pipeline (3 layers, short-circuits on failure):
    Layer 1: Tool output checks (free, deterministic)
    Layer 2: Plugin validators (user-provided, via existing plugin system)
    Layer 3: LLM review (opt-in only)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.task import Task
from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.decomposer import Decomposer, TaskAnalysis
from nucleusiq.agents.components.validation import ValidationPipeline, ValidationResult
from nucleusiq.agents.components.progress import ExecutionProgress, StepRecord
from nucleusiq.plugins.errors import PluginHalt


class AutonomousMode(BaseExecutionMode):
    """Gear 3: Autonomous mode — structured orchestration over Standard mode.

    For simple tasks: runs Standard mode directly, validates, retries if needed.
    For parallel tasks: decomposes, runs sub-agents in parallel, synthesizes.
    All paths: external validation (tool checks + plugins) with structured retry.
    """

    async def run(self, agent: "Agent", task: Any) -> Any:
        if isinstance(task, dict):
            task = Task.from_dict(task)

        agent._logger.debug("Executing in AUTONOMOUS mode")

        if not agent.llm:
            agent._logger.warning("No LLM — falling back to standard mode")
            return await StandardMode().run(agent, task)

        decomposer = Decomposer(logger=agent._logger)
        analysis = await decomposer.analyze(agent, task)

        if analysis.is_complex and len(analysis.sub_tasks) >= 2:
            agent._logger.info(
                "Task classified as COMPLEX (%d sub-tasks) — decomposing",
                len(analysis.sub_tasks),
            )
            return await self._run_complex(agent, task, decomposer, analysis)

        agent._logger.info("Task classified as SIMPLE — standard + validate")
        return await self._run_simple(agent, task)

    # ------------------------------------------------------------------ #
    # Simple path: Standard mode + validate + retry                        #
    # ------------------------------------------------------------------ #

    async def _run_simple(self, agent: "Agent", task: Task) -> Any:
        """Execute via Standard mode with validation and structured retry."""
        max_retries = getattr(agent.config, "max_retries", 3)
        validation = ValidationPipeline(logger=agent._logger)
        progress = ExecutionProgress(task_id=task.id)
        std_mode = StandardMode()

        std_mode._ensure_executor(agent)
        tool_specs = std_mode._get_tool_specs(agent)
        messages = std_mode.build_messages(agent, task)

        step = progress.add_step("execute", task.objective)

        result = None
        for attempt in range(max_retries):
            label = "EXECUTE" if attempt == 0 else "RETRY"
            agent._logger.info(
                "Attempt %d/%d [%s]", attempt + 1, max_retries, label,
            )

            step.mark_executing()
            agent.state = AgentState.EXECUTING

            try:
                result = await std_mode._tool_call_loop(
                    agent, task, messages, tool_specs,
                )
            except PluginHalt:
                raise
            except Exception as e:
                step.mark_failed(str(e))
                agent._logger.error("Execution error: %s", e)
                agent.state = AgentState.ERROR
                return f"Error: {e}"

            agent._last_messages = messages

            if self._is_error(result):
                step.mark_failed(str(result))
                agent.state = AgentState.COMPLETED
                return result

            vr = await validation.validate(agent, result, messages)
            agent._logger.info(
                "Attempt %d/%d [VALIDATE]: valid=%s layer=%s",
                attempt + 1, max_retries, vr.valid, vr.layer,
            )

            if vr.valid:
                step.mark_completed(str(result))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return result

            if attempt < max_retries - 1:
                agent._logger.info(
                    "Validation failed: %s — retrying with error context",
                    vr.reason,
                )
                retry_msg = self._build_retry_message(vr)
                messages.append(ChatMessage(role="user", content=retry_msg))

        step.mark_completed(str(result))
        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress
        return result

    # ------------------------------------------------------------------ #
    # Complex path: Decompose → Parallel → Synthesize + validate           #
    # ------------------------------------------------------------------ #

    async def _run_complex(
        self,
        agent: "Agent",
        task: Task,
        decomposer: Decomposer,
        analysis: TaskAnalysis,
    ) -> Any:
        """Decomposition: parallel sub-agents → synthesize → validate → retry."""
        max_sub = getattr(agent.config, "max_sub_agents", 5)
        max_retries = getattr(agent.config, "max_retries", 3)
        validation = ValidationPipeline(logger=agent._logger)
        progress = ExecutionProgress(task_id=task.id)
        std_mode = StandardMode()
        std_mode._ensure_executor(agent)

        # Step 1: Run sub-tasks in parallel
        sub_step = progress.add_step("decompose", "Run parallel sub-tasks")
        sub_step.mark_executing()

        findings = await decomposer.run_sub_tasks(
            parent=agent,
            sub_tasks=analysis.sub_tasks,
            max_sub_agents=max_sub,
        )

        sub_step.mark_completed(f"{len(findings)} findings collected")
        agent._logger.info(
            "Decomposition complete: %d findings collected", len(findings),
        )

        # Step 2: Synthesize with validation + retry
        synth_step = progress.add_step("synthesize", "Combine findings")
        synth_prompt = decomposer.build_synthesis_prompt(
            task.objective, findings,
        )
        tool_specs = std_mode._get_tool_specs(agent)
        messages = std_mode.build_messages(
            agent, Task(id=f"{task.id}-synth", objective=synth_prompt),
        )

        result = None
        for attempt in range(max_retries):
            label = "SYNTHESIZE" if attempt == 0 else "RETRY"
            agent._logger.info(
                "Synthesis attempt %d/%d [%s]",
                attempt + 1, max_retries, label,
            )

            synth_step.mark_executing()
            agent.state = AgentState.EXECUTING

            try:
                result = await std_mode._tool_call_loop(
                    agent, task, messages, tool_specs,
                )
            except PluginHalt:
                raise
            except Exception as e:
                synth_step.mark_failed(str(e))
                agent._logger.error("Synthesis error: %s", e)
                agent.state = AgentState.ERROR
                return f"Error: {e}"

            agent._last_messages = messages

            vr = await validation.validate(agent, result, messages)
            agent._logger.info(
                "Synthesis attempt %d/%d [VALIDATE]: valid=%s layer=%s",
                attempt + 1, max_retries, vr.valid, vr.layer,
            )

            if vr.valid:
                synth_step.mark_completed(str(result))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return result

            if attempt < max_retries - 1:
                agent._logger.info(
                    "Synthesis validation failed: %s — retrying",
                    vr.reason,
                )
                retry_msg = self._build_retry_message(vr)
                messages.append(ChatMessage(role="user", content=retry_msg))

        synth_step.mark_completed(str(result))
        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_retry_message(vr: ValidationResult) -> str:
        """Build a retry message from a validation failure."""
        parts = [f"Your previous answer had an issue: {vr.reason}"]
        if vr.details:
            parts.append(f"Details: {'; '.join(vr.details)}")
        parts.append("Please fix the issue and provide a corrected answer.")
        return "\n".join(parts)

    @staticmethod
    def _is_error(result: Any) -> bool:
        return isinstance(result, str) and result.strip().startswith("Error:")
