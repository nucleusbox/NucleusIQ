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

    def __init__(self, mode: BaseExecutionMode, critic: Critic) -> None:
        self._mode = mode
        self._critic = critic

    async def run(
        self,
        agent: Agent,
        task_objective: str,
        result: Any,
        messages: list[ChatMessage],
    ) -> CritiqueResult:
        """Build the verification prompt, call the LLM, parse the result.

        Single LLM call, no tool access — we use the reasoning-verification
        prompt (no "call tools" instruction) to avoid confusing models
        that take instructions literally.  Token budget is the agent's
        configured ``llm_max_output_tokens``.

        On *any* exception the Critic is treated as non-fatal: we log a
        warning and return ``UNCERTAIN`` with score ``0.0`` so the
        orchestrator can retry or abstain instead of falsely passing.
        """
        phase_controller = getattr(agent, "_phase_controller", None)
        if phase_controller is not None:
            phase_controller.enter("VALIDATE")
        try:
            engine = getattr(agent, "_context_engine", None)
            content_store = getattr(engine, "store", None) if engine else None
            per_tool_cap = _compute_critic_per_tool_cap(agent, messages)
            build_package_messages = getattr(
                agent, "_build_synthesis_messages_from_context", None
            )
            package_messages = (
                build_package_messages(
                    task=task_objective,
                    output_shape=(
                        "Use this curated package as the primary verification "
                        "context before considering any raw trace."
                    ),
                )
                if build_package_messages is not None
                else None
            )
            package_text = (
                package_messages[0].content
                if package_messages and package_messages[0].content
                else ""
            )
            last_pkg = getattr(agent, "_last_synthesis_package", None)
            omitted = (
                last_pkg.metadata.get("omitted_sections") or []
                if last_pkg is not None
                else []
            )
            evidence_not_in_package = "Supported Evidence" in omitted

            if package_text and not evidence_not_in_package:
                activator = getattr(agent, "_context_state_activator", None)
                if activator is not None:
                    activator.metrics.critic_used_package = True
                if phase_controller is not None:
                    phase_controller.critic_used_package = True
                task_for_critic = (
                    f"{task_objective}\n\n"
                    "## CURATED SYNTHESIS PACKAGE FOR VERIFICATION\n"
                    f"{package_text}"
                )
                generator_messages = None
            else:
                activator = getattr(agent, "_context_state_activator", None)
                if activator is not None:
                    activator.metrics.raw_trace_fallback_used = True
                task_for_critic = task_objective
                generator_messages = messages
            verification_prompt = self._critic.build_verification_prompt(
                task_objective=task_for_critic,
                final_result=result,
                generator_messages=generator_messages,
                allow_tool_instructions=False,
                content_store=content_store,
                per_tool_char_cap=per_tool_cap,
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
            agent._logger.warning("Critic failed (non-fatal, uncertain verdict): %s", e)
            return CritiqueResult(
                verdict=Verdict.UNCERTAIN,
                score=0.0,
                feedback=f"Critic infrastructure error: {e}",
            )


def _compute_critic_per_tool_cap(
    agent: Agent,
    messages: list[ChatMessage],
) -> int | None:
    """Resolve the adaptive per-tool-result char cap for the Critic.

    v0.7.8 — the Critic's rehydrated tool-result trace is capped by a
    **runtime-computed** budget, not the legacy fixed
    ``CriticLimits.tool_result`` (3K/5K chars).  The budget shrinks as
    the trace grows and scales with the LLM's actual context window,
    so the same framework works for 32K open-source models and 2M
    Gemini without any model-specific tuning.

    Returns ``None`` when the agent has no context config or no LLM,
    in which case the Critic falls back to the legacy fixed limits
    for full backward compatibility.
    """
    cfg = getattr(getattr(agent, "config", None), "context", None)
    if cfg is None or agent.llm is None:
        return None

    from nucleusiq.agents.context.store import compute_per_tool_cap

    context_window = agent.llm.get_context_window()
    num_tool_results = sum(1 for m in messages if getattr(m, "role", None) == "tool")

    return compute_per_tool_cap(
        context_window=context_window,
        prompt_overhead_tokens=cfg.critic_prompt_overhead_tokens,
        response_reserve_tokens=cfg.critic_response_reserve_tokens,
        num_tool_results=num_tool_results,
        min_chars=cfg.tool_result_per_call_min_chars,
        max_chars=cfg.tool_result_per_call_max_chars,
        purpose="critic",
    )


__all__ = ["CriticRunner"]
