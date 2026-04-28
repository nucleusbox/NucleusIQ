"""
``RefinerRunner`` — orchestrates one ``Refiner.revise`` pass.

Responsibility:
  1. Build a bounded tool-result summary from the primary agent's
     conversation (so the ``Refiner`` can re-synthesise against the
     same evidence without seeing the full, potentially-biased
     context).
  2. Invoke ``Refiner.revise`` with the bounded inputs.
  3. Downgrade any exception to a ``None`` return so ``AutonomousMode``
     can fall back to the legacy "inject correction + re-run primary
     agent" path instead of failing the whole task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nucleusiq.agents.components.critic import CritiqueResult
from nucleusiq.agents.components.refiner import Refiner, RevisionCandidate
from nucleusiq.agents.modes.autonomous.helpers import summarize_tool_results

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.agents.chat_models import ChatMessage


class RefinerRunner:
    """Runs one ``Refiner.revise`` pass with graceful degradation."""

    def __init__(self, refiner: Refiner) -> None:
        self._refiner = refiner

    async def run(
        self,
        agent: Agent,
        task_objective: str,
        candidate: Any,
        critique: CritiqueResult,
        messages: list[ChatMessage],
    ) -> RevisionCandidate | None:
        """Return a ``RevisionCandidate`` or ``None`` on non-fatal failure.

        The ``Refiner`` sees:

        * The original task objective.
        * The previous candidate answer (truncated inside ``Refiner``).
        * The ``Critic``'s structured critique.
        * A bounded tool-result summary (``summarize_tool_results``).

        It does NOT see the primary agent's raw conversation — that
        would re-introduce the same bias the ``Critic`` just flagged
        (see the Aletheia alignment design doc).
        """
        try:
            engine = getattr(agent, "_context_engine", None)
            content_store = getattr(engine, "store", None) if engine else None
            per_tool_cap, total_cap = _compute_refiner_char_caps(agent, messages)
            tool_summary = summarize_tool_results(
                messages,
                per_tool_char_cap=per_tool_cap,
                total_char_cap=total_cap,
                content_store=content_store,
            )
            return await self._refiner.revise(
                agent=agent,
                task_objective=task_objective,
                candidate=candidate,
                critique=critique,
                tool_result_summary=tool_summary,
            )
        except Exception as e:
            agent._logger.warning(
                "Refiner failed (non-fatal, falling back to primary-agent retry): %s",
                e,
            )
            return None


def _compute_refiner_char_caps(
    agent: Agent,
    messages: list[ChatMessage],
) -> tuple[int | None, int | None]:
    """Resolve ``(per_tool_cap, total_cap)`` for the Refiner's rehydration.

    v0.7.8 — mirrors ``CriticRunner``'s adaptive budget so the Refiner
    and Critic see symmetric evidence.  The Refiner's prompt/response
    reserves are larger than the Critic's (see ``ContextConfig``
    defaults) because the Refiner prompt includes the full prior
    candidate and its response must contain the full revised answer.

    Returns ``(None, None)`` when the agent has no context config or
    no LLM, in which case ``summarize_tool_results`` applies no
    per-tool truncation (full backward compatibility).
    """
    cfg = getattr(getattr(agent, "config", None), "context", None)
    if cfg is None or agent.llm is None:
        return None, None

    from nucleusiq.agents.context.store import compute_per_tool_cap

    context_window = agent.llm.get_context_window()
    num_tool_results = sum(1 for m in messages if getattr(m, "role", None) == "tool")

    per_tool_cap = compute_per_tool_cap(
        context_window=context_window,
        prompt_overhead_tokens=cfg.refiner_prompt_overhead_tokens,
        response_reserve_tokens=cfg.refiner_response_reserve_tokens,
        num_tool_results=num_tool_results,
        min_chars=cfg.tool_result_per_call_min_chars,
        max_chars=cfg.tool_result_per_call_max_chars,
        purpose="refiner",
    )

    # Aggregate ceiling: chars equivalent of the total rehydration
    # budget.  Prevents pathological many-tool traces from overshooting
    # the Refiner's prompt budget even when each per-tool cap is fine.
    available_tokens = (
        context_window
        - cfg.refiner_prompt_overhead_tokens
        - cfg.refiner_response_reserve_tokens
    )
    total_cap = max(0, available_tokens) * 4  # chars/token
    return per_tool_cap, total_cap


__all__ = ["RefinerRunner"]
