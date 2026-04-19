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
        agent: "Agent",
        task_objective: str,
        candidate: Any,
        critique: CritiqueResult,
        messages: list["ChatMessage"],
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
            tool_summary = summarize_tool_results(messages)
            return await self._refiner.revise(
                agent=agent,
                task_objective=task_objective,
                candidate=candidate,
                critique=critique,
                tool_result_summary=tool_summary,
            )
        except Exception as e:
            agent._logger.warning(
                "Refiner failed (non-fatal, falling back to "
                "primary-agent retry): %s",
                e,
            )
            return None


__all__ = ["RefinerRunner"]
