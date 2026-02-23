"""
Refiner Component — The Reviser SubAgent in the Generate → Verify → Revise loop.

The Refiner acts as part of the Reviser subagent.  It translates the Critic's
structured feedback into a targeted correction, which the Generator then
executes with full tool access.  Together (Refiner message + Generator re-run)
they form the Reviser subagent — capable of multiple LLM calls and tool use.

Architecture:
    Critic (Verifier SubAgent) produces CritiqueResult(verdict=FAIL, ...)
        ↓
    Refiner builds a targeted correction message
        ↓
    Generator SubAgent re-runs with the correction injected
        ↓
    Corrected result goes back to Critic for re-verification

Revision strategies:
    1. Conversation injection (autonomous mode) — the Refiner builds a
       correction message that gets injected into the Generator's conversation.
       The Generator re-runs with full tool access to fix the error.
    2. Tool re-execution (plan-step mode) — the Refiner asks the LLM to
       infer better arguments and re-executes the tool.
    3. LLM revision (plan-step mode) — the Refiner asks the LLM to produce
       a corrected result incorporating the Critic's feedback.

Design Principles:
    1. Directed revision — uses Critic's specific feedback, not blind retry
    2. Builds on previous attempt — doesn't start from scratch
    3. Strategy selection — picks the right revision approach automatically
    4. Graceful fallback — returns original result if revision fails
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.chat_models import ToolCallRequest
from nucleusiq.agents.components.critic import CritiqueResult


class Refiner:
    """Reviser component for the autonomous execution loop.

    Usage::

        refiner = Refiner()
        corrected = await refiner.refine_step(
            agent, task_objective, step, bad_result, critique, context,
        )
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def build_revision_message(
        self,
        critique: CritiqueResult,
    ) -> str:
        """Build a targeted correction message from Critic feedback.

        Used by AutonomousMode to inject a revision directive into the
        conversation.  The message tells the model exactly what went
        wrong and what to fix, without redoing correct work.

        Args:
            critique: The Critic's structured feedback (verdict=FAIL).

        Returns:
            A user-role message string for conversation injection.
        """
        issues = "\n".join(
            f"- {i}" for i in critique.issues
        ) if critique.issues else "- error detected in your answer"
        suggestions = "\n".join(
            f"- {s}" for s in critique.suggestions
        ) if critique.suggestions else ""

        msg = (
            "VERIFICATION FOUND A SPECIFIC ERROR in your answer.\n\n"
            f"Issues identified:\n{issues}\n"
        )
        if suggestions:
            msg += f"\nSuggested fixes:\n{suggestions}\n"
        msg += (
            "\nREVISION INSTRUCTIONS:\n"
            "- Fix ONLY the specific error(s) listed above.\n"
            "- Do NOT redo work that was already correct.\n"
            "- Re-check against the tool results in the conversation above.\n"
            "- If you need to re-call a tool with different arguments, do so.\n"
            "- Provide your corrected final answer."
        )
        return msg

    async def refine_step(
        self,
        agent: "Agent",
        task_objective: str,
        step: PlanStep,
        previous_result: Any,
        critique: CritiqueResult,
        context: Dict[str, Any],
    ) -> Any:
        """Produce a corrected result for a single step.

        Automatically selects the revision strategy:
        - Tool steps → re-infer arguments and re-execute the tool
        - LLM / execute steps → ask the LLM for a corrected response

        Args:
            agent: Agent instance
            task_objective: The user's original task text
            step: The PlanStep being revised
            previous_result: The result that the Critic rejected
            critique: The Critic's structured feedback
            context: Accumulated results from prior steps

        Returns:
            Corrected result (or original if revision fails)
        """
        tool_names = {
            t.name for t in (agent.tools or []) if hasattr(t, "name")
        }
        is_tool_step = step.action in tool_names

        try:
            if is_tool_step and agent.llm:
                return await self._refine_tool_step(
                    agent, task_objective, step, previous_result,
                    critique, context,
                )
            elif agent.llm:
                return await self._refine_llm_step(
                    agent, task_objective, step, previous_result,
                    critique, context,
                )
            else:
                self._logger.warning(
                    "No LLM available for refinement — returning original"
                )
                return previous_result
        except Exception as e:
            self._logger.warning(
                "Refiner failed for step %d: %s. "
                "Returning original result.",
                step.step, e,
            )
            return previous_result

    async def refine_final(
        self,
        agent: "Agent",
        task_objective: str,
        plan: Plan,
        results: List[Any],
        final_result: Any,
        critique: CritiqueResult,
    ) -> Any:
        """Produce a corrected final result using the Critic's feedback.

        Args:
            agent: Agent instance
            task_objective: The user's original task text
            plan: The execution plan
            results: All step results
            final_result: The final result that the Critic rejected
            critique: The Critic's structured feedback

        Returns:
            Corrected final result
        """
        if not agent.llm:
            return final_result

        try:
            prompt = self._build_final_revision_prompt(
                task_objective, plan, results, final_result, critique,
            )
            response = await self._call_llm(agent, prompt)
            content = self._extract_content(response)
            return content if content else final_result
        except Exception as e:
            self._logger.warning(
                "Final refinement failed: %s. Returning original.", e,
            )
            return final_result

    # ------------------------------------------------------------------ #
    # Strategy 1: Tool re-execution                                       #
    # ------------------------------------------------------------------ #

    async def _refine_tool_step(
        self,
        agent: "Agent",
        task_objective: str,
        step: PlanStep,
        previous_result: Any,
        critique: CritiqueResult,
        context: Dict[str, Any],
    ) -> Any:
        """Re-infer tool arguments based on critique, then re-execute."""
        self._logger.info(
            "Refiner: re-inferring args for tool '%s' (step %d)",
            step.action, step.step,
        )

        prompt = self._build_tool_arg_revision_prompt(
            task_objective, step, previous_result, critique, context, agent,
        )

        tool_specs = (
            agent.llm.convert_tool_specs(agent.tools)
            if agent.tools else []
        )

        call_kwargs: Dict[str, Any] = {
            "model": getattr(agent.llm, "model_name", "default"),
            "messages": [{"role": "user", "content": prompt}],
            "tools": tool_specs if tool_specs else None,
            "max_tokens": getattr(
                agent.config, "step_inference_max_tokens", 2048,
            ),
        }
        call_kwargs.update(getattr(agent, "_current_llm_overrides", {}))

        response = await agent.llm.call(**call_kwargs)
        new_args = self._extract_tool_args(response, step.action)

        if new_args is not None:
            fn_call = ToolCallRequest(
                name=step.action,
                arguments=json.dumps(new_args),
            )
            self._logger.info(
                "Refiner: re-executing tool '%s' with revised args",
                step.action,
            )
            return await agent._executor.execute(fn_call)

        self._logger.warning(
            "Refiner: could not infer new args for '%s', "
            "falling back to LLM revision",
            step.action,
        )
        return await self._refine_llm_step(
            agent, task_objective, step, previous_result, critique, context,
        )

    # ------------------------------------------------------------------ #
    # Strategy 2: LLM-based revision                                      #
    # ------------------------------------------------------------------ #

    async def _refine_llm_step(
        self,
        agent: "Agent",
        task_objective: str,
        step: PlanStep,
        previous_result: Any,
        critique: CritiqueResult,
        context: Dict[str, Any],
    ) -> Any:
        """Ask the LLM to produce a corrected result from the critique."""
        self._logger.info(
            "Refiner: LLM revision for step %d (%s)",
            step.step, step.action,
        )

        prompt = self._build_llm_revision_prompt(
            task_objective, step, previous_result, critique, context,
        )
        response = await self._call_llm(agent, prompt)
        content = self._extract_content(response)
        return content if content else previous_result

    # ------------------------------------------------------------------ #
    # Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_tool_arg_revision_prompt(
        task_objective: str,
        step: PlanStep,
        previous_result: Any,
        critique: CritiqueResult,
        context: Dict[str, Any],
        agent: "Agent",
    ) -> str:
        issues_list = "\n".join(
            f"  - {i}" for i in critique.issues
        ) or "  (none specified)"
        suggestions_list = "\n".join(
            f"  - {s}" for s in critique.suggestions
        ) or "  (none specified)"
        ctx_str = json.dumps(
            {k: str(v)[:200] for k, v in context.items()},
            indent=2,
        ) if context else "{}"

        tool_specs = (
            agent.llm.convert_tool_specs(agent.tools)
            if agent.tools else []
        )
        target_spec = ""
        for spec in tool_specs:
            fn = spec.get("function", {}) if isinstance(spec, dict) else {}
            if fn.get("name") == step.action:
                target_spec = json.dumps(fn, indent=2)
                break

        return (
            "A previous tool call produced an incorrect result. "
            "You must call the tool again with BETTER arguments.\n\n"
            f"## Original Task\n{task_objective}\n\n"
            f"## Tool\n{step.action}\n"
            f"{f'Spec: {target_spec}' if target_spec else ''}\n\n"
            f"## Previous Arguments\n{json.dumps(step.args or {})}\n\n"
            f"## Previous Result\n{str(previous_result)[:1000]}\n\n"
            f"## Reviewer Feedback\n"
            f"Verdict: {critique.verdict.value}\n"
            f"Score: {critique.score}\n"
            f"Feedback: {critique.feedback}\n"
            f"Issues:\n{issues_list}\n"
            f"Suggestions:\n{suggestions_list}\n\n"
            f"## Context From Previous Steps\n{ctx_str}\n\n"
            f"Now call the tool '{step.action}' with corrected arguments "
            "that address the reviewer's feedback.\n"
        )

    @staticmethod
    def _build_llm_revision_prompt(
        task_objective: str,
        step: PlanStep,
        previous_result: Any,
        critique: CritiqueResult,
        context: Dict[str, Any],
    ) -> str:
        step_details = step.details or step.action
        issues_list = "\n".join(
            f"  - {i}" for i in critique.issues
        ) or "  (none specified)"
        suggestions_list = "\n".join(
            f"  - {s}" for s in critique.suggestions
        ) or "  (none specified)"
        ctx_summary = ""
        if context:
            lines = []
            for k, v in context.items():
                if k.endswith("_action"):
                    continue
                lines.append(f"  - {k}: {str(v)[:300]}")
            if lines:
                ctx_summary = "\n\nPrevious Steps:\n" + "\n".join(lines)

        return (
            "You are a revision specialist. A previous attempt at "
            "completing a step had issues identified by a quality reviewer.\n\n"
            f"## Original Task\n{task_objective}\n\n"
            f"## Step to Revise\nStep {step.step}: {step_details}\n\n"
            f"## Previous Attempt's Result\n"
            f"{str(previous_result)[:2000]}\n\n"
            f"## Reviewer Feedback\n"
            f"Verdict: {critique.verdict.value}\n"
            f"Score: {critique.score}\n"
            f"Overall: {critique.feedback}\n"
            f"Issues Found:\n{issues_list}\n"
            f"Suggested Improvements:\n{suggestions_list}\n"
            f"{ctx_summary}\n\n"
            "## Your Job\n"
            "Produce a CORRECTED result that addresses all the issues "
            "raised by the reviewer. Focus specifically on the problems "
            "identified — build upon and improve the previous attempt, "
            "do not start from scratch.\n\n"
            "Provide your corrected result:\n"
        )

    @staticmethod
    def _build_final_revision_prompt(
        task_objective: str,
        plan: Plan,
        results: List[Any],
        final_result: Any,
        critique: CritiqueResult,
    ) -> str:
        plan_lines = []
        for s in plan.steps:
            plan_lines.append(f"  Step {s.step}: {s.action}")
        plan_summary = "\n".join(plan_lines)

        results_lines = []
        for i, r in enumerate(results, 1):
            results_lines.append(f"  Step {i}: {str(r)[:500]}")
        results_summary = "\n".join(results_lines)

        issues_list = "\n".join(
            f"  - {i}" for i in critique.issues
        ) or "  (none specified)"
        suggestions_list = "\n".join(
            f"  - {s}" for s in critique.suggestions
        ) or "  (none specified)"

        return (
            "You are a revision specialist. The final output of an "
            "AI agent was reviewed and found lacking. Your job is to "
            "produce an improved version.\n\n"
            f"## Original Task\n{task_objective}\n\n"
            f"## Plan Followed\n{plan_summary}\n\n"
            f"## Step Results\n{results_summary}\n\n"
            f"## Current Final Result\n"
            f"{str(final_result)[:2000]}\n\n"
            f"## Reviewer Feedback\n"
            f"Verdict: {critique.verdict.value}\n"
            f"Score: {critique.score}\n"
            f"Overall: {critique.feedback}\n"
            f"Issues:\n{issues_list}\n"
            f"Suggestions:\n{suggestions_list}\n\n"
            "## Your Job\n"
            "Produce a CORRECTED and COMPLETE final result that "
            "addresses the reviewer's concerns and fully satisfies "
            "the original task.\n"
        )

    # ------------------------------------------------------------------ #
    # LLM helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _call_llm(agent: "Agent", prompt: str) -> Any:
        call_kwargs: Dict[str, Any] = {
            "model": getattr(agent.llm, "model_name", "default"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": getattr(
                agent.config, "step_inference_max_tokens", 2048,
            ),
        }
        call_kwargs.update(getattr(agent, "_current_llm_overrides", {}))
        return await agent.llm.call(**call_kwargs)

    @staticmethod
    def _extract_content(response: Any) -> Optional[str]:
        """Pull text content from an LLM response object."""
        if not response or not hasattr(response, "choices") or not response.choices:
            return None
        msg = response.choices[0].message
        if isinstance(msg, dict):
            return msg.get("content")
        return getattr(msg, "content", None)

    @staticmethod
    def _extract_tool_args(response: Any, action: str) -> Optional[Dict[str, Any]]:
        """Extract tool arguments from an LLM response for re-execution."""
        if not response or not hasattr(response, "choices") or not response.choices:
            return None
        msg = response.choices[0].message
        tool_calls = (
            msg.get("tool_calls")
            if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None)
        )
        if not tool_calls or not isinstance(tool_calls, list):
            return None

        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                fn_name = fn.get("name") if isinstance(fn, dict) else None
                fn_args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
            else:
                fn_info = getattr(tc, "function", None)
                fn_name = getattr(fn_info, "name", None) if fn_info else None
                fn_args = getattr(fn_info, "arguments", "{}") if fn_info else "{}"

            if fn_name == action:
                try:
                    return (
                        json.loads(fn_args)
                        if isinstance(fn_args, str)
                        else (fn_args or {})
                    )
                except (json.JSONDecodeError, TypeError):
                    return None
        return None
