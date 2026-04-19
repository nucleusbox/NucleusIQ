"""
``CriticRunner`` — encapsulates one Critic verification pass.

SRP: takes an agent, a prepared result and message list, drives the
``Critic``'s prompt builder + an LLM call, parses the response into a
``CritiqueResult``, and falls back to PASS on non-fatal Critic errors
(we never want a Critic bug to block the user's task).

Decoupling: does NOT import ``AutonomousMode`` — it only depends on the
``BaseExecutionMode.call_llm`` wrapping for usage-tracker hooks.  The
caller passes the mode instance in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nucleusiq.agents.components.critic import (
    Critic,
    CritiqueResult,
    Verdict,
)
from nucleusiq.agents.usage.usage_tracker import CallPurpose

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.modes.base_mode import BaseExecutionMode


class CriticRunner:
    """Runs the ``Critic`` role against a candidate answer.

    Args:
        mode: The enclosing execution mode (supplies ``call_llm`` for
            usage-tracker-aware LLM invocation).
        critic: The ``Critic`` component (owns prompt/parse logic).
    """

    def __init__(self, mode: "BaseExecutionMode", critic: Critic) -> None:
        self._mode = mode
        self._critic = critic

    async def run(
        self,
        agent: "Agent",
        task_objective: str,
        result: Any,
        messages: list["ChatMessage"],
    ) -> CritiqueResult:
        """Build the verification prompt, call the LLM, parse the result.

        Single LLM call, no tool access — we use the reasoning-verification
        prompt (no "call tools" instruction) to avoid confusing models
        that take instructions literally.  Token budget is the agent's
        configured ``llm_max_output_tokens``.

        On *any* exception the Critic is treated as non-fatal: we log a
        warning and return a synthetic ``PASS`` verdict so the candidate
        is accepted rather than the whole task failing on a verifier
        infrastructure problem.
        """
        try:
            verification_prompt = self._critic.build_verification_prompt(
                task_objective=task_objective,
                final_result=result,
                generator_messages=messages,
                allow_tool_instructions=False,
            )

            model_name = getattr(agent.llm, "model_name", "default")
            token_budget = agent.config.llm_max_output_tokens

            call_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [{"role": "user", "content": verification_prompt}],
                "max_output_tokens": token_budget,
            }
            call_kwargs.update(getattr(agent, "_current_llm_overrides", {}))

            response = await self._mode.call_llm(
                agent, call_kwargs, purpose=CallPurpose.CRITIC
            )

            text = ""
            if hasattr(response, "choices") and response.choices:
                msg = response.choices[0].message
                text = getattr(msg, "content", "") or ""
            elif isinstance(response, str):
                text = response

            return self._critic.parse_result_text(text)

        except Exception as e:
            agent._logger.warning(
                "Critic failed (non-fatal, accepting result): %s", e
            )
            return CritiqueResult(
                verdict=Verdict.PASS,
                score=0.5,
                feedback=f"Critic error: {e}",
            )


__all__ = ["CriticRunner"]
