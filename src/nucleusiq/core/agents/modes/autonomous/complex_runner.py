"""
``ComplexRunner`` — owns the decomposition-and-synthesis path.

Task flow:
  1. Decomposer runs the parallel sub-tasks.
  2. Sub-agent metrics are rolled up into the parent tracer.
  3. A synthesis prompt is built and fed to the same
     primary-agent -> Critic -> Refiner loop used by ``SimpleRunner``.

The synthesis loop is deliberately kept close to ``SimpleRunner``'s
loop (same structure, same helpers) so the two remain easy to
cross-reference; factoring the loop into a shared object is deferred
until F3/F4 introduce ``ComputeBudget`` and best-of-N (at which point
the shared shape will be obvious).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from nucleusiq.agents.agent_result import AbstentionSignal
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.compute_budget import (
    Action,
    ComputeBudget,
    decide_next_action,
)
from nucleusiq.agents.components.critic import Critic, CritiqueResult, Verdict
from nucleusiq.agents.components.decomposer import Decomposer, TaskAnalysis
from nucleusiq.agents.components.progress import ExecutionProgress
from nucleusiq.agents.components.refiner import Refiner
from nucleusiq.agents.components.validation import ValidationPipeline
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.modes.autonomous import helpers, telemetry
from nucleusiq.agents.modes.autonomous.critic_runner import CriticRunner
from nucleusiq.agents.modes.autonomous.refiner_runner import RefinerRunner
from nucleusiq.agents.modes.autonomous.simple_runner import SimpleRunner
from nucleusiq.agents.task import Task
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.streaming.events import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
    from nucleusiq.agents.modes.standard_mode import StandardMode


# F2: single threshold for UNCERTAIN verdicts; no per-attempt relaxation.
_UNCERTAIN_ACCEPT_THRESHOLD = 0.7


class ComplexRunner:
    """Executes the decompose → parallel → synthesize path."""

    def __init__(
        self,
        mode: "AutonomousMode",
        std_mode: "StandardMode",
        validation: ValidationPipeline,
        critic: Critic,
        refiner: Refiner,
    ) -> None:
        self._mode = mode
        self._std_mode = std_mode
        self._validation = validation
        self._critic = critic
        self._refiner = refiner
        # Actual Critic / Refiner execution is routed back through the mode
        # (``mode._run_critic`` / ``mode._run_refiner``) so tests that patch
        # those methods on ``AutonomousMode`` continue to work.  The bare
        # runners are kept here as a fallback / documentation of intent.
        self._critic_runner = CriticRunner(mode, critic)
        self._refiner_runner = RefinerRunner(refiner)

    # ------------------------------------------------------------------ #
    # Sync entrypoint                                                     #
    # ------------------------------------------------------------------ #

    async def run_sync(
        self,
        agent: "Agent",
        task: Task,
        decomposer: Decomposer,
        analysis: TaskAnalysis,
    ) -> Any:
        max_sub = getattr(agent.config, "max_sub_agents", 5)
        budget = ComputeBudget.from_config(agent.config)
        cfg_snapshot = SimpleRunner._snapshot_budget_on_config(agent)
        progress = ExecutionProgress(task_id=task.id)
        self._std_mode._ensure_executor(agent)

        # --- Step 1: Run sub-tasks in parallel ----------------------
        sub_step = progress.add_step("decompose", "Run parallel sub-tasks")
        sub_step.mark_executing()

        findings = await decomposer.run_sub_tasks(
            parent=agent,
            sub_tasks=analysis.sub_tasks,
            max_sub_agents=max_sub,
        )

        telemetry.rollup_sub_agent_metrics(agent, decomposer._sub_agent_results)

        sub_step.mark_completed(f"{len(findings)} findings collected")
        agent._logger.info(
            "Decomposition complete: %d findings collected", len(findings)
        )

        # --- Step 2: Synthesize with validation + Critic/Refiner ----
        synth_step = progress.add_step("synthesize", "Combine findings")
        synth_prompt = decomposer.build_synthesis_prompt(task.objective, findings)
        tool_specs = self._std_mode._get_tool_specs(agent)
        messages = self._std_mode.build_messages(
            agent, Task(id=f"{task.id}-synth", objective=synth_prompt)
        )

        sub_task_names = tuple(
            s.get("objective", s.get("id", "")) if isinstance(s, dict) else str(s)
            for s in analysis.sub_tasks
        )

        result: Any = None
        last_critique: CritiqueResult | None = None
        tool_summary_cache: str | None = None
        refined_at_least_once = False
        critique_history: list[CritiqueResult] = []
        best_content: Any = None
        best_critique: CritiqueResult | None = None

        attempt = 0
        final_attempt_index = 0
        try:
            while attempt < budget.max_retries:
                final_attempt_index = attempt
                SimpleRunner._apply_budget_to_config(agent, budget)

                use_refiner = (
                    attempt > 0
                    and result is not None
                    and not helpers.is_error_result(result)
                    and last_critique is not None
                )

                label = (
                    "SYNTHESIZE"
                    if attempt == 0
                    else ("REFINE" if use_refiner else "RETRY")
                )
                agent._logger.info(
                    "Synthesis attempt %d/%d [%s]",
                    attempt + 1,
                    budget.max_retries,
                    label,
                )

                synth_step.mark_executing()
                agent.state = AgentState.EXECUTING

                if use_refiner:
                    if tool_summary_cache is None:
                        tool_summary_cache = helpers.summarize_tool_results(
                            messages
                        )
                    revision = await self._mode._run_refiner(
                        agent,
                        self._refiner,
                        task.objective,
                        result,
                        last_critique,  # type: ignore[arg-type]
                        messages,
                    )
                    if revision is not None:
                        refined_at_least_once = True
                        telemetry.record_revision(
                            agent,
                            attempt + 1,
                            last_critique,  # type: ignore[arg-type]
                            revision,
                        )
                        prior_len = len(str(result or ""))
                        new_len = len(str(revision.content or ""))
                        agent._logger.info(
                            "Synthesis attempt %d/%d [REFINE]: chars %d → %d "
                            "(delta=%+d), tool_calls=%d",
                            attempt + 1,
                            budget.max_retries,
                            prior_len,
                            new_len,
                            new_len - prior_len,
                            getattr(revision, "tool_calls_made", 0) or 0,
                        )
                        result = revision.content
                    else:
                        fallback_msg = helpers.build_fallback_revision_message(
                            last_critique  # type: ignore[arg-type]
                        )
                        messages.append(
                            ChatMessage(role="user", content=fallback_msg)
                        )
                        last_critique = None
                        tool_summary_cache = None
                        try:
                            result = await self._std_mode._tool_call_loop(
                                agent, task, messages, tool_specs
                            )
                        except PluginHalt:
                            raise
                        except Exception as e:
                            synth_step.mark_failed(str(e))
                            agent._logger.error("Synthesis error: %s", e)
                            agent.state = AgentState.ERROR
                            return f"Error: {e}"
                        agent._last_messages = messages
                else:
                    try:
                        result = await self._std_mode._tool_call_loop(
                            agent, task, messages, tool_specs
                        )
                    except PluginHalt:
                        raise
                    except Exception as e:
                        synth_step.mark_failed(str(e))
                        agent._logger.error("Synthesis error: %s", e)
                        agent.state = AgentState.ERROR
                        return f"Error: {e}"
                    agent._last_messages = messages
                    tool_summary_cache = None

                budget = SimpleRunner._sync_budget_with_tracer(agent, budget)

                vr = await self._validation.validate(agent, result, messages)
                telemetry.record_validation(
                    agent, attempt + 1, vr.valid, vr.layer, vr.reason
                )
                agent._logger.info(
                    "Synthesis attempt %d/%d [VALIDATE]: valid=%s layer=%s",
                    attempt + 1,
                    budget.max_retries,
                    vr.valid,
                    vr.layer,
                )

                if not vr.valid:
                    if attempt < budget.max_retries - 1:
                        retry_msg = helpers.build_validation_retry(vr)
                        messages.append(
                            ChatMessage(role="user", content=retry_msg)
                        )
                    last_critique = None
                    attempt += 1
                    continue

                critique = await self._mode._run_critic(
                    agent, self._critic, task.objective, result, messages
                )
                telemetry.record_validation(
                    agent,
                    attempt + 1,
                    critique.verdict in (Verdict.PASS, Verdict.UNCERTAIN),
                    "critic",
                    f"{critique.verdict.value} (score={critique.score:.2f})",
                )
                telemetry.record_critic_verdict(agent, attempt + 1, critique)
                critique_history.append(critique)

                # Fix 2/4 — uniform verdict logging with delta.
                delta_str = ""
                if len(critique_history) >= 2:
                    delta = critique.score - critique_history[-2].score
                    delta_str = f" (delta={delta:+.2f})"
                agent._logger.info(
                    "Synthesis attempt %d/%d [CRITIC]: %s (score=%.2f)%s "
                    "— %s",
                    attempt + 1,
                    budget.max_retries,
                    critique.verdict.value,
                    critique.score,
                    delta_str,
                    (critique.feedback or "no feedback")[:120],
                )

                # F5 — track best candidate.
                if SimpleRunner._is_better_critique(critique, best_critique):
                    best_content = result
                    best_critique = critique

                budget = SimpleRunner._sync_budget_with_tracer(agent, budget)

                decision = decide_next_action(
                    critique,
                    critique_history,
                    budget,
                    attempt,
                    _UNCERTAIN_ACCEPT_THRESHOLD,
                )

                if decision.action == Action.STOP_ACCEPT:
                    agent._logger.info(
                        "Synthesis attempt %d/%d [ACCEPT]: %s "
                        "(score=%.2f) — returning result",
                        attempt + 1,
                        budget.max_retries,
                        critique.verdict.value,
                        critique.score,
                    )
                    synth_step.mark_completed(str(result))
                    agent.state = AgentState.COMPLETED
                    agent._execution_progress = progress
                    telemetry.set_autonomous_detail(
                        agent,
                        attempts=attempt + 1,
                        max_attempts=budget.max_retries,
                        sub_tasks=sub_task_names,
                        complexity="complex",
                        refined=refined_at_least_once,
                        cumulative_tokens=budget.cumulative_tokens_spent,
                    )
                    return result

                if decision.action == Action.STOP_ABSTAIN:
                    last_critique = critique
                    final_attempt_index = attempt
                    break

                if decision.escalation_reason is not None:
                    telemetry.record_escalation(
                        agent,
                        attempt + 1,
                        decision.escalation_reason,
                        budget,
                        decision.budget,
                    )

                budget = decision.budget
                last_critique = critique
                final_attempt_index = attempt
                attempt += 1
        finally:
            SimpleRunner._restore_config(agent, cfg_snapshot)

        budget = SimpleRunner._sync_budget_with_tracer(agent, budget)
        attempts_completed = (
            final_attempt_index + 1 if last_critique is not None else attempt
        )
        telemetry.set_autonomous_detail(
            agent,
            attempts=attempts_completed,
            max_attempts=budget.max_retries,
            sub_tasks=sub_task_names,
            complexity="complex",
            refined=refined_at_least_once,
            cumulative_tokens=budget.cumulative_tokens_spent,
        )

        # F5/F6 — accept-with-warning or abstain with best candidate.
        if last_critique is not None:
            assert best_critique is not None
            first_critique = critique_history[0]
            if SimpleRunner._should_accept_with_warning(
                best_critique, first_critique
            ):
                warning = SimpleRunner._build_warning(
                    best_critique, first_critique, attempts_completed
                )
                agent._logger.warning(
                    "Synthesis attempt %d/%d [ACCEPT-WITH-WARNING]: "
                    "best=%s (score=%.2f, improvement=%+.2f) — accepting "
                    "best candidate",
                    attempts_completed,
                    budget.max_retries,
                    best_critique.verdict.value,
                    best_critique.score,
                    best_critique.score - first_critique.score,
                )
                tracer = getattr(agent, "_tracer", None)
                if tracer is not None:
                    try:
                        tracer.record_warning(warning)
                    except Exception:
                        pass
                synth_step.mark_completed(str(best_content))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return best_content

            agent._logger.info(
                "Synthesis attempt %d/%d [ABSTAIN]: Critic refused all "
                "retries — returning ResultStatus.ABSTAINED (best "
                "score=%.2f, verdict=%s)",
                attempts_completed,
                budget.max_retries,
                best_critique.score,
                best_critique.verdict.value,
            )
            synth_step.mark_completed(str(best_content))
            agent.state = AgentState.COMPLETED
            agent._execution_progress = progress
            raise AbstentionSignal(
                best_candidate=best_content,
                critique=best_critique,
                reason=(
                    best_critique.feedback
                    or "Critic rejected all synthesis candidates across the "
                    "retry budget"
                ),
            )

        synth_step.mark_completed(str(result))
        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress
        return result

    # ------------------------------------------------------------------ #
    # Streaming entrypoint                                                #
    # ------------------------------------------------------------------ #

    async def run_stream(
        self,
        agent: "Agent",
        task: Task,
        decomposer: Decomposer,
        analysis: TaskAnalysis,
    ) -> AsyncGenerator[StreamEvent, None]:
        max_sub = getattr(agent.config, "max_sub_agents", 5)
        budget = ComputeBudget.from_config(agent.config)
        cfg_snapshot = SimpleRunner._snapshot_budget_on_config(agent)
        progress = ExecutionProgress(task_id=task.id)
        self._std_mode._ensure_executor(agent)

        yield StreamEvent.thinking_event(
            f"Decomposing into {len(analysis.sub_tasks)} sub-tasks…"
        )

        sub_step = progress.add_step("decompose", "Run parallel sub-tasks")
        sub_step.mark_executing()

        findings = await decomposer.run_sub_tasks(
            parent=agent,
            sub_tasks=analysis.sub_tasks,
            max_sub_agents=max_sub,
        )

        telemetry.rollup_sub_agent_metrics(agent, decomposer._sub_agent_results)

        sub_step.mark_completed(f"{len(findings)} findings collected")
        yield StreamEvent.thinking_event(
            f"Sub-tasks complete — synthesizing {len(findings)} findings…"
        )

        synth_prompt = decomposer.build_synthesis_prompt(task.objective, findings)
        tool_specs = self._std_mode._get_tool_specs(agent)
        messages = self._std_mode.build_messages(
            agent, Task(id=f"{task.id}-synth", objective=synth_prompt)
        )

        final_content: str | None = None
        last_critique: CritiqueResult | None = None
        tool_summary_cache: str | None = None
        critique_history: list[CritiqueResult] = []
        best_content: str | None = None
        best_critique: CritiqueResult | None = None
        refined_at_least_once = False

        attempt = 0
        final_attempt_index = 0
        try:
            while attempt < budget.max_retries:
                final_attempt_index = attempt
                SimpleRunner._apply_budget_to_config(agent, budget)

                use_refiner = (
                    attempt > 0
                    and final_content is not None
                    and not helpers.is_error_result(final_content)
                    and last_critique is not None
                )

                label = (
                    "SYNTHESIZE"
                    if attempt == 0
                    else ("REFINE" if use_refiner else "RETRY")
                )
                agent._logger.info(
                    "Synthesis attempt %d/%d [%s]",
                    attempt + 1,
                    budget.max_retries,
                    label,
                )
                agent.state = AgentState.EXECUTING

                if use_refiner:
                    yield StreamEvent.thinking_event(
                        f"Refiner: revising synthesis (verdict="
                        f"{last_critique.verdict.value}, "  # type: ignore[union-attr]
                        f"score={last_critique.score:.2f})…"  # type: ignore[union-attr]
                    )
                    if tool_summary_cache is None:
                        tool_summary_cache = helpers.summarize_tool_results(
                            messages
                        )
                    revision = await self._mode._run_refiner(
                        agent,
                        self._refiner,
                        task.objective,
                        final_content,
                        last_critique,  # type: ignore[arg-type]
                        messages,
                    )
                    if revision is not None:
                        refined_at_least_once = True
                        telemetry.record_revision(
                            agent,
                            attempt + 1,
                            last_critique,  # type: ignore[arg-type]
                            revision,
                        )
                        prior_len = len(str(final_content or ""))
                        new_len = len(str(revision.content or ""))
                        agent._logger.info(
                            "Synthesis attempt %d/%d [REFINE]: chars %d → %d "
                            "(delta=%+d), tool_calls=%d",
                            attempt + 1,
                            budget.max_retries,
                            prior_len,
                            new_len,
                            new_len - prior_len,
                            getattr(revision, "tool_calls_made", 0) or 0,
                        )
                        final_content = revision.content
                    else:
                        yield StreamEvent.thinking_event(
                            "Refiner failed, falling back to synthesis retry…"
                        )
                        fallback_msg = helpers.build_fallback_revision_message(
                            last_critique  # type: ignore[arg-type]
                        )
                        messages.append(
                            ChatMessage(role="user", content=fallback_msg)
                        )
                        last_critique = None
                        tool_summary_cache = None
                        final_content = None
                        attempt += 1
                        continue
                else:
                    final_content = None
                    try:
                        async for event in self._mode._streaming_tool_call_loop(
                            agent,
                            messages,
                            tool_specs,
                            max_tool_calls=budget.max_tool_calls,
                            max_output_tokens=budget.max_output_tokens,
                        ):
                            if event.type == StreamEventType.COMPLETE:
                                final_content = event.content
                            elif event.type == StreamEventType.ERROR:
                                yield event
                                return
                            yield event
                    except PluginHalt:
                        raise
                    except Exception as e:
                        agent._logger.error("Synthesis error: %s", e)
                        agent.state = AgentState.ERROR
                        yield StreamEvent.error_event(str(e))
                        return

                    agent._last_messages = messages
                    tool_summary_cache = None

                budget = SimpleRunner._sync_budget_with_tracer(agent, budget)

                vr = await self._validation.validate(
                    agent, final_content, messages
                )
                if not vr.valid:
                    if attempt < budget.max_retries - 1:
                        yield StreamEvent.thinking_event(
                            f"Validation failed ({vr.reason}), retrying…"
                        )
                        retry_msg = helpers.build_validation_retry(vr)
                        messages.append(
                            ChatMessage(role="user", content=retry_msg)
                        )
                    last_critique = None
                    attempt += 1
                    continue

                yield StreamEvent.thinking_event(
                    "Verifying synthesis with Critic…"
                )
                critique = await self._mode._run_critic(
                    agent, self._critic, task.objective, final_content, messages
                )
                telemetry.record_critic_verdict(agent, attempt + 1, critique)
                critique_history.append(critique)

                # Fix 2/4 — uniform verdict log with delta.
                delta_str = ""
                if len(critique_history) >= 2:
                    delta = critique.score - critique_history[-2].score
                    delta_str = f" (delta={delta:+.2f})"
                agent._logger.info(
                    "Synthesis attempt %d/%d [CRITIC]: %s (score=%.2f)%s "
                    "— %s",
                    attempt + 1,
                    budget.max_retries,
                    critique.verdict.value,
                    critique.score,
                    delta_str,
                    (critique.feedback or "no feedback")[:120],
                )

                # F5 — best-candidate tracking.
                if SimpleRunner._is_better_critique(critique, best_critique):
                    best_content = final_content
                    best_critique = critique

                budget = SimpleRunner._sync_budget_with_tracer(agent, budget)

                decision = decide_next_action(
                    critique,
                    critique_history,
                    budget,
                    attempt,
                    _UNCERTAIN_ACCEPT_THRESHOLD,
                )

                if decision.action == Action.STOP_ACCEPT:
                    agent.state = AgentState.COMPLETED
                    agent._execution_progress = progress
                    return

                if decision.action == Action.STOP_ABSTAIN:
                    last_critique = critique
                    final_attempt_index = attempt
                    break

                if decision.escalation_reason is not None:
                    telemetry.record_escalation(
                        agent,
                        attempt + 1,
                        decision.escalation_reason,
                        budget,
                        decision.budget,
                    )

                budget = decision.budget

                if attempt < budget.max_retries - 1:
                    yield StreamEvent.thinking_event(
                        f"Critic: {critique.verdict.value} "
                        f"(score={critique.score:.2f}) — "
                        f"scheduling Refiner pass…"
                    )
                last_critique = critique
                final_attempt_index = attempt
                attempt += 1
        finally:
            SimpleRunner._restore_config(agent, cfg_snapshot)

        attempts_completed = (
            final_attempt_index + 1 if last_critique is not None else attempt
        )

        if last_critique is not None:
            assert best_critique is not None
            first_critique = critique_history[0]
            if SimpleRunner._should_accept_with_warning(
                best_critique, first_critique
            ):
                warning = SimpleRunner._build_warning(
                    best_critique, first_critique, attempts_completed
                )
                agent._logger.warning(
                    "Synthesis attempt %d/%d [ACCEPT-WITH-WARNING]: "
                    "best=%s (score=%.2f, improvement=%+.2f) — accepting "
                    "best candidate",
                    attempts_completed,
                    budget.max_retries,
                    best_critique.verdict.value,
                    best_critique.score,
                    best_critique.score - first_critique.score,
                )
                tracer = getattr(agent, "_tracer", None)
                if tracer is not None:
                    try:
                        tracer.record_warning(warning)
                    except Exception:
                        pass
                yield StreamEvent.thinking_event(
                    f"Accepting best synthesis candidate despite Critic "
                    f"uncertainty (score={best_critique.score:.2f})"
                )
                yield StreamEvent.complete_event(best_content or "")
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return

            agent._logger.info(
                "Synthesis attempt %d/%d [ABSTAIN]: Critic refused all "
                "retries — returning ResultStatus.ABSTAINED (best "
                "score=%.2f, verdict=%s)",
                attempts_completed,
                budget.max_retries,
                best_critique.score,
                best_critique.verdict.value,
            )
            agent.state = AgentState.COMPLETED
            agent._execution_progress = progress
            raise AbstentionSignal(
                best_candidate=best_content,
                critique=best_critique,
                reason=(
                    best_critique.feedback
                    or "Critic rejected all synthesis candidates across the "
                    "retry budget"
                ),
            )

        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress


__all__ = ["ComplexRunner"]
