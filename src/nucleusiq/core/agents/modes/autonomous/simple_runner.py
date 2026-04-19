"""
``SimpleRunner`` — owns the simple-task execution path.

Encapsulates the Aletheia-style loop:

    primary agent pass (StandardMode tool loop)
        -> Layer 1+2 validation
        -> Critic verification
        -> Refiner on Critic FAIL (F1)
        -> loop until PASS / UNCERTAIN+accepted / max_retries

Split out of ``autonomous_mode.py`` so ``AutonomousMode`` stays a thin
dispatcher.  The runner takes already-instantiated dependencies so that
test patches on ``autonomous_mode.StandardMode`` /
``autonomous_mode.ValidationPipeline`` apply transparently.
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
from nucleusiq.agents.components.progress import ExecutionProgress
from nucleusiq.agents.components.refiner import Refiner
from nucleusiq.agents.components.validation import ValidationPipeline
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.modes.autonomous import helpers, telemetry
from nucleusiq.agents.modes.autonomous.critic_runner import CriticRunner
from nucleusiq.agents.modes.autonomous.refiner_runner import RefinerRunner
from nucleusiq.agents.task import Task
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.streaming.events import StreamEvent, StreamEventType

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
    from nucleusiq.agents.modes.standard_mode import StandardMode


# F2: single threshold for UNCERTAIN verdicts, no per-attempt relaxation.
# An UNCERTAIN verdict with score >= this threshold is accepted; anything
# below it keeps the retry loop going and ultimately triggers abstention
# if the loop runs out of budget.
_UNCERTAIN_ACCEPT_THRESHOLD = 0.7

# F6: "accept-with-warning" policy.  When the retry budget is exhausted
# and the Critic never issued PASS, we still accept the run's best
# candidate IFF:
#     * best Critic score >= _ACCEPT_WITH_WARNING_FLOOR
#     * best score is at least _ACCEPT_WITH_WARNING_IMPROVEMENT above
#       the first attempt's score (demonstrates Refiner actually helped)
#     * best verdict is UNCERTAIN (not FAIL — FAIL means the Critic
#       found a concrete error we should not paper over)
# Anything else falls through to hard abstention as before.
_ACCEPT_WITH_WARNING_FLOOR = 0.55
_ACCEPT_WITH_WARNING_IMPROVEMENT = 0.10


class SimpleRunner:
    """Executes the simple-task path (no decomposition)."""

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
        # NB: actual Critic / Refiner execution is routed back through the
        # mode (``mode._run_critic`` / ``mode._run_refiner``) so tests that
        # patch those methods on ``AutonomousMode`` continue to work and
        # subclasses retain the ability to override either pass.
        self._critic_runner = CriticRunner(mode, critic)
        self._refiner_runner = RefinerRunner(refiner)

    # ------------------------------------------------------------------ #
    # ComputeBudget plumbing (F3)                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _snapshot_budget_on_config(agent: "Agent") -> dict[str, Any]:
        """Snapshot budget-relevant ``AgentConfig`` values for restore.

        The runner mutates ``llm_max_output_tokens`` and
        ``max_tool_calls`` on the config so ``StandardMode._tool_call_loop``
        and its call-kwargs builders pick up escalated limits.  We
        restore these at the end of the run so the config object the
        user passed in is never permanently changed.
        """
        cfg = agent.config
        return {
            "llm_max_output_tokens": getattr(cfg, "llm_max_output_tokens", None),
            "max_tool_calls": getattr(cfg, "max_tool_calls", None),
        }

    @staticmethod
    def _restore_config(agent: "Agent", snapshot: dict[str, Any]) -> None:
        cfg = agent.config
        for key, value in snapshot.items():
            try:
                setattr(cfg, key, value)
            except Exception:
                pass

    @staticmethod
    def _apply_budget_to_config(agent: "Agent", budget: ComputeBudget) -> None:
        """Mirror the current budget onto ``agent.config``.

        ``StandardMode._tool_call_loop`` reads these values to build its
        LLM call kwargs, so mutating the config is the least-invasive way
        to let escalations take effect for the next attempt.
        """
        cfg = agent.config
        try:
            cfg.llm_max_output_tokens = budget.max_output_tokens
        except Exception:
            pass
        try:
            cfg.max_tool_calls = budget.max_tool_calls
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Best-candidate + accept-with-warning helpers (F5 + F6)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_better_critique(
        new: CritiqueResult, current_best: CritiqueResult | None
    ) -> bool:
        """Monotonic "betterness" ordering used by F5 best-tracking.

        * PASS verdicts always beat non-PASS, regardless of score.
        * Within the same verdict tier, higher score wins.
        * UNCERTAIN beats FAIL at equal score.
        """
        if current_best is None:
            return True
        # PASS dominates.
        if new.verdict == Verdict.PASS and current_best.verdict != Verdict.PASS:
            return True
        if current_best.verdict == Verdict.PASS and new.verdict != Verdict.PASS:
            return False
        # UNCERTAIN dominates FAIL at equal score.
        if (
            new.verdict == Verdict.UNCERTAIN
            and current_best.verdict == Verdict.FAIL
            and new.score >= current_best.score
        ):
            return True
        if (
            current_best.verdict == Verdict.UNCERTAIN
            and new.verdict == Verdict.FAIL
            and current_best.score >= new.score
        ):
            return False
        # Otherwise: higher score wins.
        return new.score > current_best.score

    @staticmethod
    def _should_accept_with_warning(
        best_critique: CritiqueResult,
        first_critique: CritiqueResult,
    ) -> bool:
        """F6 policy check — is the best candidate good enough to accept?

        See module-level ``_ACCEPT_WITH_WARNING_*`` constants for the
        thresholds.  Returns False when the best candidate is still a
        FAIL (Critic cited concrete errors) or when the Refiner didn't
        demonstrably improve over the first attempt.
        """
        if best_critique.verdict != Verdict.UNCERTAIN:
            return False
        if best_critique.score < _ACCEPT_WITH_WARNING_FLOOR:
            return False
        improvement = best_critique.score - first_critique.score
        return improvement >= _ACCEPT_WITH_WARNING_IMPROVEMENT

    @staticmethod
    def _build_warning(
        best_critique: CritiqueResult,
        first_critique: CritiqueResult,
        attempts: int,
    ) -> str:
        """Human-readable accept-with-warning message surfaced via
        ``tracer.record_warning`` and ``AgentResult.warnings``.
        """
        return (
            f"Critic could not fully verify the output "
            f"(verdict={best_critique.verdict.value}, "
            f"score={best_critique.score:.2f}) after {attempts} attempt(s); "
            f"best candidate improved by "
            f"{best_critique.score - first_critique.score:+.2f} vs. the "
            f"first attempt. Review the output before relying on it."
        )

    @staticmethod
    def _sync_budget_with_tracer(
        agent: "Agent", budget: ComputeBudget
    ) -> ComputeBudget:
        """Update ``budget.cumulative_tokens_spent`` from the tracer.

        The tracer already tracks per-call token usage (used for usage
        reports and cost accounting); we reuse it as the single source
        of truth for budget accounting instead of double-counting.
        """
        tracer = getattr(agent, "_tracer", None)
        if tracer is None:
            return budget
        try:
            total = sum(lc.total_tokens for lc in tracer.llm_calls)
        except Exception:
            return budget
        if total <= budget.cumulative_tokens_spent:
            return budget
        return budget.model_copy(update={"cumulative_tokens_spent": total})

    # ------------------------------------------------------------------ #
    # Sync entrypoint                                                     #
    # ------------------------------------------------------------------ #

    async def run_sync(self, agent: "Agent", task: Task) -> Any:
        """Execute the simple path, returning the final result.

        F3 — the outer loop is driven by ``ComputeBudget``.  Each Critic
        verdict feeds ``decide_next_action`` which may escalate the
        budget (growing ``max_retries`` + ``max_output_tokens`` +
        ``max_tool_calls``) before the next attempt runs.
        """
        budget = ComputeBudget.from_config(agent.config)
        cfg_snapshot = self._snapshot_budget_on_config(agent)
        progress = ExecutionProgress(task_id=task.id)

        self._std_mode._ensure_executor(agent)
        tool_specs = self._std_mode._get_tool_specs(agent)
        messages = self._std_mode.build_messages(agent, task)

        step = progress.add_step("execute", task.objective)

        result: Any = None
        last_critique: CritiqueResult | None = None
        tool_summary_cache: str | None = None
        refined_at_least_once = False
        critique_history: list[CritiqueResult] = []
        # F5: track the best candidate across attempts so we return the
        # highest-quality output (or surface it on abstention), not just
        # the most recent one.
        best_content: Any = None
        best_critique: CritiqueResult | None = None

        attempt = 0
        final_attempt_index = 0
        try:
            while attempt < budget.max_retries:
                final_attempt_index = attempt
                self._apply_budget_to_config(agent, budget)

                use_refiner = (
                    attempt > 0
                    and result is not None
                    and not helpers.is_error_result(result)
                    and last_critique is not None
                )

                label = (
                    "EXECUTE"
                    if attempt == 0
                    else ("REFINE" if use_refiner else "RETRY")
                )
                agent._logger.info(
                    "Attempt %d/%d [%s]", attempt + 1, budget.max_retries, label
                )

                step.mark_executing()
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
                            "Attempt %d/%d [REFINE]: chars %d → %d "
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
                            step.mark_failed(str(e))
                            agent._logger.error("Execution error: %s", e)
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
                        step.mark_failed(str(e))
                        agent._logger.error("Execution error: %s", e)
                        agent.state = AgentState.ERROR
                        return f"Error: {e}"
                    agent._last_messages = messages
                    tool_summary_cache = None

                budget = self._sync_budget_with_tracer(agent, budget)

                if helpers.is_error_result(result):
                    agent._logger.warning(
                        "Attempt %d/%d: primary agent returned error — "
                        "treating as FAIL for retry (not bailing out). "
                        "Error: %s",
                        attempt + 1,
                        budget.max_retries,
                        str(result)[:200],
                    )
                    if attempt < budget.max_retries - 1:
                        messages.append(
                            ChatMessage(
                                role="user",
                                content=(
                                    "Your previous attempt produced an error "
                                    "or empty response. Please try again "
                                    "from scratch and provide a complete "
                                    "answer to the task."
                                ),
                            )
                        )
                    last_critique = None
                    attempt += 1
                    continue

                vr = await self._validation.validate(agent, result, messages)
                telemetry.record_validation(
                    agent, attempt + 1, vr.valid, vr.layer, vr.reason
                )
                agent._logger.info(
                    "Attempt %d/%d [VALIDATE]: valid=%s layer=%s",
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

                # Fix 2/4 — log verdict uniformly on EVERY path with
                # delta-vs-prior for trajectory observability, BEFORE
                # control-flow branching (STOP_ACCEPT / STOP_ABSTAIN /
                # RETRY all need to be visible).
                delta_str = ""
                if len(critique_history) >= 2:
                    delta = critique.score - critique_history[-2].score
                    delta_str = f" (delta={delta:+.2f})"
                agent._logger.info(
                    "Attempt %d/%d [CRITIC]: %s (score=%.2f)%s — %s",
                    attempt + 1,
                    budget.max_retries,
                    critique.verdict.value,
                    critique.score,
                    delta_str,
                    (critique.feedback or "no feedback")[:120],
                )

                # F5 — track the best candidate + critique across
                # attempts so abstention/accept-with-warning can return
                # the highest-quality output, not the most recent.
                if self._is_better_critique(critique, best_critique):
                    best_content = result
                    best_critique = critique

                budget = self._sync_budget_with_tracer(agent, budget)

                decision = decide_next_action(
                    critique,
                    critique_history,
                    budget,
                    attempt,
                    _UNCERTAIN_ACCEPT_THRESHOLD,
                )

                if decision.action == Action.STOP_ACCEPT:
                    agent._logger.info(
                        "Attempt %d/%d [ACCEPT]: %s (score=%.2f) — "
                        "returning result",
                        attempt + 1,
                        budget.max_retries,
                        critique.verdict.value,
                        critique.score,
                    )
                    step.mark_completed(str(result))
                    agent.state = AgentState.COMPLETED
                    agent._execution_progress = progress
                    telemetry.set_autonomous_detail(
                        agent,
                        attempts=attempt + 1,
                        max_attempts=budget.max_retries,
                        complexity="simple",
                        refined=refined_at_least_once,
                        cumulative_tokens=budget.cumulative_tokens_spent,
                    )
                    return result

                if decision.action == Action.STOP_ABSTAIN:
                    last_critique = critique
                    final_attempt_index = attempt
                    break

                if decision.escalation_reason is not None:
                    agent._logger.info(
                        "Attempt %d/%d [ESCALATE]: %s — retries %d→%d, "
                        "tokens %d→%d",
                        attempt + 1,
                        budget.max_retries,
                        decision.escalation_reason,
                        budget.max_retries,
                        decision.budget.max_retries,
                        budget.max_output_tokens,
                        decision.budget.max_output_tokens,
                    )
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
            self._restore_config(agent, cfg_snapshot)

        budget = self._sync_budget_with_tracer(agent, budget)
        attempts_completed = (
            final_attempt_index + 1 if last_critique is not None else attempt
        )
        telemetry.set_autonomous_detail(
            agent,
            attempts=attempts_completed,
            max_attempts=budget.max_retries,
            complexity="simple",
            refined=refined_at_least_once,
            cumulative_tokens=budget.cumulative_tokens_spent,
        )

        # F5/F6 — last_critique being non-None means the Critic ran on
        # at least one attempt and the loop exited via STOP_ABSTAIN (or
        # natural exhaustion with the Critic having rejected the
        # latest candidate).  We now decide between three outcomes:
        #     (a) Accept best candidate (F6: soft accept-with-warning)
        #     (b) Abstain and surface best candidate (F5 + F2)
        #     (c) Natural return of the plain result (Critic never ran)
        if last_critique is not None:
            assert best_critique is not None  # last_critique implies at least one run
            first_critique = critique_history[0]
            if self._should_accept_with_warning(best_critique, first_critique):
                warning = self._build_warning(
                    best_critique, first_critique, attempts_completed
                )
                agent._logger.warning(
                    "Attempt %d/%d [ACCEPT-WITH-WARNING]: best=%s "
                    "(score=%.2f, improvement=%+.2f) — accepting best "
                    "candidate",
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
                step.mark_completed(str(best_content))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return best_content

            agent._logger.info(
                "Attempt %d/%d [ABSTAIN]: Critic refused all retries — "
                "returning ResultStatus.ABSTAINED (best score=%.2f, "
                "verdict=%s)",
                attempts_completed,
                budget.max_retries,
                best_critique.score,
                best_critique.verdict.value,
            )
            step.mark_completed(str(best_content))
            agent.state = AgentState.COMPLETED
            agent._execution_progress = progress
            raise AbstentionSignal(
                best_candidate=best_content,
                critique=best_critique,
                reason=(
                    best_critique.feedback
                    or "Critic rejected all candidates across the retry budget"
                ),
            )

        step.mark_completed(str(result))
        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress
        return result

    # ------------------------------------------------------------------ #
    # Streaming entrypoint                                                #
    # ------------------------------------------------------------------ #

    async def run_stream(
        self, agent: "Agent", task: Task
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream the simple path: primary agent → validate → Critic → Refiner.

        F3 — mirrors ``run_sync``: ``ComputeBudget`` drives the outer
        loop and escalations emit ``EscalationRecord`` telemetry.
        """
        budget = ComputeBudget.from_config(agent.config)
        cfg_snapshot = self._snapshot_budget_on_config(agent)
        progress = ExecutionProgress(task_id=task.id)

        self._std_mode._ensure_executor(agent)
        tool_specs = self._std_mode._get_tool_specs(agent)
        messages = self._std_mode.build_messages(agent, task)

        step = progress.add_step("execute", task.objective)

        final_content: str | None = None
        last_critique: CritiqueResult | None = None
        tool_summary_cache: str | None = None
        critique_history: list[CritiqueResult] = []
        # F5: streaming path also tracks best-of-attempts.
        best_content: str | None = None
        best_critique: CritiqueResult | None = None
        refined_at_least_once = False

        attempt = 0
        final_attempt_index = 0
        try:
            while attempt < budget.max_retries:
                final_attempt_index = attempt
                self._apply_budget_to_config(agent, budget)

                use_refiner = (
                    attempt > 0
                    and final_content is not None
                    and not helpers.is_error_result(final_content)
                    and last_critique is not None
                )

                label = (
                    "EXECUTE"
                    if attempt == 0
                    else ("REFINE" if use_refiner else "RETRY")
                )
                agent._logger.info(
                    "Attempt %d/%d [%s]", attempt + 1, budget.max_retries, label
                )

                step.mark_executing()
                agent.state = AgentState.EXECUTING

                if use_refiner:
                    yield StreamEvent.thinking_event(
                        f"Refiner: revising candidate (verdict="
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
                            "Attempt %d/%d [REFINE]: chars %d → %d "
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
                            "Refiner failed, falling back to primary-agent retry…"
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
                        step.mark_failed(str(e))
                        agent._logger.error("Execution error: %s", e)
                        agent.state = AgentState.ERROR
                        yield StreamEvent.error_event(str(e))
                        return

                    agent._last_messages = messages
                    tool_summary_cache = None

                budget = self._sync_budget_with_tracer(agent, budget)

                if helpers.is_error_result(final_content):
                    agent._logger.warning(
                        "Attempt %d/%d: primary agent returned error — "
                        "treating as FAIL for retry (not bailing out). "
                        "Error: %s",
                        attempt + 1,
                        budget.max_retries,
                        str(final_content)[:200],
                    )
                    if attempt < budget.max_retries - 1:
                        yield StreamEvent.thinking_event(
                            "Primary-agent error — retrying with fresh prompt…"
                        )
                        messages.append(
                            ChatMessage(
                                role="user",
                                content=(
                                    "Your previous attempt produced an error "
                                    "or empty response. Please try again "
                                    "from scratch and provide a complete "
                                    "answer to the task."
                                ),
                            )
                        )
                    last_critique = None
                    attempt += 1
                    continue

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

                yield StreamEvent.thinking_event("Verifying result with Critic…")
                critique = await self._mode._run_critic(
                    agent, self._critic, task.objective, final_content, messages
                )
                telemetry.record_critic_verdict(agent, attempt + 1, critique)
                critique_history.append(critique)

                # Fix 2/4 — uniform verdict log with delta, BEFORE
                # branching so STOP_ACCEPT / STOP_ABSTAIN / RETRY all
                # appear in logs.
                delta_str = ""
                if len(critique_history) >= 2:
                    delta = critique.score - critique_history[-2].score
                    delta_str = f" (delta={delta:+.2f})"
                agent._logger.info(
                    "Attempt %d/%d [CRITIC]: %s (score=%.2f)%s — %s",
                    attempt + 1,
                    budget.max_retries,
                    critique.verdict.value,
                    critique.score,
                    delta_str,
                    (critique.feedback or "no feedback")[:120],
                )

                # F5 — best-candidate tracking.
                if self._is_better_critique(critique, best_critique):
                    best_content = final_content
                    best_critique = critique

                budget = self._sync_budget_with_tracer(agent, budget)

                decision = decide_next_action(
                    critique,
                    critique_history,
                    budget,
                    attempt,
                    _UNCERTAIN_ACCEPT_THRESHOLD,
                )

                if decision.action == Action.STOP_ACCEPT:
                    step.mark_completed(str(final_content))
                    agent.state = AgentState.COMPLETED
                    agent._execution_progress = progress
                    telemetry.set_autonomous_detail(
                        agent,
                        attempts=attempt + 1,
                        max_attempts=budget.max_retries,
                        complexity="simple",
                        refined=refined_at_least_once,
                        cumulative_tokens=budget.cumulative_tokens_spent,
                    )
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
            self._restore_config(agent, cfg_snapshot)

        attempts_completed = (
            final_attempt_index + 1 if last_critique is not None else attempt
        )
        telemetry.set_autonomous_detail(
            agent,
            attempts=attempts_completed,
            max_attempts=budget.max_retries,
            complexity="simple",
            refined=refined_at_least_once,
            cumulative_tokens=budget.cumulative_tokens_spent,
        )

        # F5/F6 — mirror the sync-path policy for streaming.
        if last_critique is not None:
            assert best_critique is not None
            first_critique = critique_history[0]
            if self._should_accept_with_warning(best_critique, first_critique):
                warning = self._build_warning(
                    best_critique, first_critique, attempts_completed
                )
                agent._logger.warning(
                    "Attempt %d/%d [ACCEPT-WITH-WARNING]: best=%s "
                    "(score=%.2f, improvement=%+.2f) — accepting best "
                    "candidate",
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
                    f"Accepting best candidate despite Critic uncertainty "
                    f"(score={best_critique.score:.2f})"
                )
                yield StreamEvent.complete_event(best_content or "")
                step.mark_completed(str(best_content))
                agent.state = AgentState.COMPLETED
                agent._execution_progress = progress
                return

            agent._logger.info(
                "Attempt %d/%d [ABSTAIN]: Critic refused all retries — "
                "returning ResultStatus.ABSTAINED (best score=%.2f, "
                "verdict=%s)",
                attempts_completed,
                budget.max_retries,
                best_critique.score,
                best_critique.verdict.value,
            )
            step.mark_completed(str(best_content))
            agent.state = AgentState.COMPLETED
            agent._execution_progress = progress
            raise AbstentionSignal(
                best_candidate=best_content,
                critique=best_critique,
                reason=(
                    best_critique.feedback
                    or "Critic rejected all candidates across the retry budget"
                ),
            )

        step.mark_completed(str(final_content))
        agent.state = AgentState.COMPLETED
        agent._execution_progress = progress


__all__ = ["SimpleRunner"]
