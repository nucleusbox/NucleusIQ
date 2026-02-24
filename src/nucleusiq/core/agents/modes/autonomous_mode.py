"""
AutonomousMode — Gear 3: Structured orchestration over Standard mode.

Adds capabilities Standard mode cannot provide alone:
- **Parallel execution** via isolated sub-agents (for independent sub-tasks)
- **Plugin-based validation** with structured retry
- **Critic** (independent verification) + **Refiner** (targeted correction)
- **Progress tracking** per step
- **Context curation** — each sub-agent sees only what it needs

Task routing (via Decomposer's 3-gate checklist):

**Simple tasks** -> Standard mode + validate + Critic/Refiner loop
**Parallel tasks** -> Decompose -> parallel Standard agents -> synthesize + validate + Critic/Refiner loop

Validation + Verification pipeline:
    Layer 1: Tool output checks (free, deterministic)
    Layer 2: Plugin validators (user-provided, via existing plugin system)
    Critic:  Independent verification (builds adaptive prompt, parses CritiqueResult)
    Refiner: Targeted correction on FAIL (specific issues + suggestions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import Critic, CritiqueResult, Verdict
from nucleusiq.agents.components.decomposer import Decomposer, TaskAnalysis
from nucleusiq.agents.components.progress import ExecutionProgress
from nucleusiq.agents.components.refiner import Refiner
from nucleusiq.agents.components.validation import ValidationPipeline, ValidationResult
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.task import Task
from nucleusiq.plugins.errors import PluginHalt

_UNCERTAIN_ACCEPT_THRESHOLD = 0.7


class AutonomousMode(BaseExecutionMode):
    """Gear 3: Autonomous mode — structured orchestration over Standard mode.

    For simple tasks: runs Standard mode directly, validates, retries if needed.
    For parallel tasks: decomposes, runs sub-agents in parallel, synthesizes.
    All paths: Layer 1+2 validation, then Critic for independent verification,
    and Refiner for targeted correction on failure.
    """

    async def run(self, agent: Agent, task: Any) -> Any:
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

        agent._logger.info("Task classified as SIMPLE — standard + validate + Critic")
        return await self._run_simple(agent, task)

    # ------------------------------------------------------------------ #
    # Simple path: Standard mode + validate + Critic/Refiner               #
    # ------------------------------------------------------------------ #

    async def _run_simple(self, agent: Agent, task: Task) -> Any:
        """Execute via Standard mode with validation, Critic, and Refiner."""
        max_retries = getattr(agent.config, "max_retries", 3)
        validation = ValidationPipeline(logger=agent._logger)
        critic = Critic(logger=agent._logger)
        refiner = Refiner(logger=agent._logger)
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
                "Attempt %d/%d [%s]",
                attempt + 1,
                max_retries,
                label,
            )

            step.mark_executing()
            agent.state = AgentState.EXECUTING

            try:
                result = await std_mode._tool_call_loop(
                    agent,
                    task,
                    messages,
                    tool_specs,
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

            # Layer 1+2: Deterministic + plugin validation
            vr = await validation.validate(agent, result, messages)
            agent._logger.info(
                "Attempt %d/%d [VALIDATE]: valid=%s layer=%s",
                attempt + 1,
                max_retries,
                vr.valid,
                vr.layer,
            )

            if not vr.valid:
                if attempt < max_retries - 1:
                    retry_msg = self._build_validation_retry(vr)
                    messages.append(ChatMessage(role="user", content=retry_msg))
                continue

            # Critic: Independent verification
            critique = await self._run_critic(
                agent, critic, task.objective, result, messages
            )

            if critique.verdict == Verdict.PASS:
                agent._logger.info(
                    "Attempt %d/%d [CRITIC]: PASS (score=%.2f)",
                    attempt + 1,
                    max_retries,
                    critique.score,
                )
                step.mark_completed(str(result))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return result

            if (
                critique.verdict == Verdict.UNCERTAIN
                and critique.score >= _UNCERTAIN_ACCEPT_THRESHOLD
            ):
                agent._logger.info(
                    "Attempt %d/%d [CRITIC]: UNCERTAIN but score=%.2f >= %.2f — accepting",
                    attempt + 1,
                    max_retries,
                    critique.score,
                    _UNCERTAIN_ACCEPT_THRESHOLD,
                )
                step.mark_completed(str(result))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return result

            agent._logger.info(
                "Attempt %d/%d [CRITIC]: %s (score=%.2f) — %s",
                attempt + 1,
                max_retries,
                critique.verdict.value,
                critique.score,
                critique.feedback[:100] if critique.feedback else "no feedback",
            )

            # Refiner: Build targeted correction
            if attempt < max_retries - 1:
                revision_msg = refiner.build_revision_message(critique)
                messages.append(ChatMessage(role="user", content=revision_msg))

        step.mark_completed(str(result))
        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress
        return result

    # ------------------------------------------------------------------ #
    # Complex path: Decompose -> Parallel -> Synthesize + validate          #
    # ------------------------------------------------------------------ #

    async def _run_complex(
        self,
        agent: Agent,
        task: Task,
        decomposer: Decomposer,
        analysis: TaskAnalysis,
    ) -> Any:
        """Decomposition: parallel sub-agents -> synthesize -> validate -> Critic/Refiner."""
        max_sub = getattr(agent.config, "max_sub_agents", 5)
        max_retries = getattr(agent.config, "max_retries", 3)
        validation = ValidationPipeline(logger=agent._logger)
        critic = Critic(logger=agent._logger)
        refiner = Refiner(logger=agent._logger)
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
            "Decomposition complete: %d findings collected",
            len(findings),
        )

        # Step 2: Synthesize with validation + Critic/Refiner
        synth_step = progress.add_step("synthesize", "Combine findings")
        synth_prompt = decomposer.build_synthesis_prompt(
            task.objective,
            findings,
        )
        tool_specs = std_mode._get_tool_specs(agent)
        messages = std_mode.build_messages(
            agent,
            Task(id=f"{task.id}-synth", objective=synth_prompt),
        )

        result = None
        for attempt in range(max_retries):
            label = "SYNTHESIZE" if attempt == 0 else "RETRY"
            agent._logger.info(
                "Synthesis attempt %d/%d [%s]",
                attempt + 1,
                max_retries,
                label,
            )

            synth_step.mark_executing()
            agent.state = AgentState.EXECUTING

            try:
                result = await std_mode._tool_call_loop(
                    agent,
                    task,
                    messages,
                    tool_specs,
                )
            except PluginHalt:
                raise
            except Exception as e:
                synth_step.mark_failed(str(e))
                agent._logger.error("Synthesis error: %s", e)
                agent.state = AgentState.ERROR
                return f"Error: {e}"

            agent._last_messages = messages

            # Layer 1+2 validation
            vr = await validation.validate(agent, result, messages)
            agent._logger.info(
                "Synthesis attempt %d/%d [VALIDATE]: valid=%s layer=%s",
                attempt + 1,
                max_retries,
                vr.valid,
                vr.layer,
            )

            if not vr.valid:
                if attempt < max_retries - 1:
                    retry_msg = self._build_validation_retry(vr)
                    messages.append(ChatMessage(role="user", content=retry_msg))
                continue

            # Critic verification
            critique = await self._run_critic(
                agent, critic, task.objective, result, messages
            )

            if critique.verdict == Verdict.PASS:
                synth_step.mark_completed(str(result))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return result

            if (
                critique.verdict == Verdict.UNCERTAIN
                and critique.score >= _UNCERTAIN_ACCEPT_THRESHOLD
            ):
                synth_step.mark_completed(str(result))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return result

            agent._logger.info(
                "Synthesis attempt %d/%d [CRITIC]: %s (score=%.2f)",
                attempt + 1,
                max_retries,
                critique.verdict.value,
                critique.score,
            )

            if attempt < max_retries - 1:
                revision_msg = refiner.build_revision_message(critique)
                messages.append(ChatMessage(role="user", content=revision_msg))

        synth_step.mark_completed(str(result))
        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress
        return result

    # ------------------------------------------------------------------ #
    # Critic execution                                                     #
    # ------------------------------------------------------------------ #

    async def _run_critic(
        self,
        agent: Agent,
        critic: Critic,
        task_objective: str,
        result: Any,
        messages: List[ChatMessage],
    ) -> CritiqueResult:
        """Run the Critic to independently verify the result.

        Builds a verification prompt, calls the LLM, and parses
        the response into a CritiqueResult.  Falls back to PASS
        if the Critic itself errors (non-fatal).
        """
        try:
            verification_prompt = critic.build_verification_prompt(
                task_objective=task_objective,
                final_result=result,
                generator_messages=messages,
            )

            response = await agent.llm.call(
                model=getattr(agent.llm, "model_name", "default"),
                messages=[{"role": "user", "content": verification_prompt}],
                max_tokens=1024,
            )

            text = ""
            if hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                text = getattr(msg, "content", "") or ""
            elif isinstance(response, str):
                text = response

            return critic.parse_result_text(text)

        except Exception as e:
            agent._logger.warning("Critic failed (non-fatal, accepting result): %s", e)
            return CritiqueResult(
                verdict=Verdict.PASS,
                score=0.5,
                feedback=f"Critic error: {e}",
            )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_validation_retry(vr: ValidationResult) -> str:
        """Build a retry message from Layer 1/2 validation failure."""
        parts = [f"Your previous answer had an issue: {vr.reason}"]
        if vr.details:
            parts.append(f"Details: {'; '.join(vr.details)}")
        parts.append("Please fix the issue and provide a corrected answer.")
        return "\n".join(parts)

    @staticmethod
    def _is_error(result: Any) -> bool:
        return isinstance(result, str) and result.strip().startswith("Error:")
