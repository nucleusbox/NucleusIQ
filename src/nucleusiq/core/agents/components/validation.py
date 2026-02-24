"""
ValidationPipeline — External validation for Autonomous Mode.

Three-layer validation:

Layer 1 (built-in): Tool output checks — did tools error? Was output empty?
Layer 2 (plugins):  User-provided validators via the plugin system.
Layer 3 (opt-in):   LLM review — only when explicitly enabled.

The pipeline returns a ValidationResult that the orchestrator uses
to decide whether to accept the result or retry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.agents.chat_models import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of the validation pipeline."""

    valid: bool
    layer: str = ""
    reason: str = ""
    details: List[str] = field(default_factory=list)


class ValidationPipeline:
    """Runs layered validation on execution results.

    Instantiate once per autonomous run; call ``validate()`` after
    each execution to decide accept-or-retry.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        llm_review: bool = False,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._llm_review = llm_review

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def validate(
        self,
        agent: Agent,
        result: Any,
        messages: List[ChatMessage],
    ) -> ValidationResult:
        """Run all applicable validation layers in order.

        Short-circuits on the first failure — no point running
        expensive layers if a cheap one already caught an issue.
        """
        # Layer 1: Tool output checks (free, deterministic)
        l1 = self._check_tool_outputs(result, messages)
        if not l1.valid:
            self._logger.info("Validation FAIL (Layer 1): %s", l1.reason)
            return l1

        # Layer 2: Plugin validators (user-provided)
        l2 = await self._run_plugin_validators(agent, result)
        if not l2.valid:
            self._logger.info("Validation FAIL (Layer 2): %s", l2.reason)
            return l2

        # Layer 3: LLM review (opt-in only)
        if self._llm_review:
            l3 = await self._run_llm_review(agent, result, messages)
            if not l3.valid:
                self._logger.info("Validation FAIL (Layer 3): %s", l3.reason)
                return l3

        return ValidationResult(valid=True, layer="all", reason="All checks passed")

    # ------------------------------------------------------------------ #
    # Layer 1: Tool output checks                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_tool_outputs(
        result: Any,
        messages: List[ChatMessage],
    ) -> ValidationResult:
        """Built-in checks on tool outputs — no LLM, no plugins."""

        result_str = str(result).strip() if result is not None else ""

        if not result_str:
            return ValidationResult(
                valid=False,
                layer="tool_output",
                reason="Empty result — execution produced no output",
            )

        if result_str.startswith("Error:"):
            return ValidationResult(
                valid=False,
                layer="tool_output",
                reason=f"Execution returned an error: {result_str[:200]}",
            )

        tool_errors = []
        for msg in messages:
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.startswith("Error executing tool"):
                tool_errors.append(content[:200])

        if tool_errors:
            return ValidationResult(
                valid=False,
                layer="tool_output",
                reason=f"{len(tool_errors)} tool error(s) detected",
                details=tool_errors,
            )

        return ValidationResult(valid=True, layer="tool_output", reason="OK")

    # ------------------------------------------------------------------ #
    # Layer 2: Plugin validators                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _run_plugin_validators(
        agent: Agent,
        result: Any,
    ) -> ValidationResult:
        """Run ResultValidatorPlugin instances registered on the agent.

        Scans the agent's plugin list for plugins that implement
        ``validate_result(result, context)`` and calls each one.
        First failure short-circuits — remaining validators are skipped.

        This is separate from ``after_agent`` hooks (which run in
        ``Agent.execute()`` after the mode finishes) to avoid double
        invocation.
        """
        from nucleusiq.plugins.builtin.result_validator import ResultValidatorPlugin

        plugins = getattr(agent, "plugins", None) or []
        validators = [p for p in plugins if isinstance(p, ResultValidatorPlugin)]

        if not validators:
            return ValidationResult(valid=True, layer="plugin", reason="No validators")

        context = {
            "agent_name": getattr(agent, "name", ""),
            "task_objective": "",
        }
        current_task = getattr(agent, "_current_task", None)
        if isinstance(current_task, dict):
            context["task_objective"] = current_task.get("objective", "")

        for validator in validators:
            try:
                valid, reason = await validator.validate_result(result, context)
                if not valid:
                    return ValidationResult(
                        valid=False,
                        layer="plugin",
                        reason=f"[{validator.name}] {reason}",
                    )
            except Exception as e:
                logger.warning(
                    "Validator %s error (non-fatal): %s",
                    validator.name,
                    e,
                )

        return ValidationResult(valid=True, layer="plugin", reason="Validators passed")

    # ------------------------------------------------------------------ #
    # Layer 3: LLM review (opt-in)                                         #
    # ------------------------------------------------------------------ #

    async def _run_llm_review(
        self,
        agent: Agent,
        result: Any,
        messages: List[ChatMessage],
    ) -> ValidationResult:
        """Optional LLM-based review — only runs when explicitly enabled.

        Uses the agent's own LLM to do a quick sanity check.
        This is the lightest possible version — one call, no sub-agent.
        """
        if not agent.llm:
            return ValidationResult(valid=True, layer="llm_review", reason="No LLM")

        task_objective = ""
        for msg in messages:
            role = getattr(msg, "role", "")
            if role == "system":
                task_objective = getattr(msg, "content", "") or ""
                break

        prompt = (
            "Review this result for obvious errors, contradictions, "
            "or incomplete answers.\n\n"
            f"## TASK\n{task_objective[:1000]}\n\n"
            f"## RESULT\n{str(result)[:2000]}\n\n"
            "Respond with ONLY 'PASS' if the result looks reasonable, "
            "or 'FAIL: <brief reason>' if there's an obvious problem."
        )

        try:
            response = await agent.llm.call(
                model=getattr(agent.llm, "model_name", "default"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = ""
            if hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                text = getattr(msg, "content", "") or ""
            elif isinstance(response, str):
                text = response

            text = text.strip().upper()
            if text.startswith("FAIL"):
                return ValidationResult(
                    valid=False,
                    layer="llm_review",
                    reason=text[:200],
                )
            return ValidationResult(
                valid=True, layer="llm_review", reason="LLM approved"
            )
        except Exception as e:
            self._logger.warning("LLM review failed (non-fatal): %s", e)
            return ValidationResult(
                valid=True, layer="llm_review", reason="LLM error (skipped)"
            )
