"""
AutonomousMode — Gear 3: structured orchestration over Standard mode.

Implements NucleusIQ's Aletheia-aligned loop, mapping paper roles onto
actual framework classes:

    primary agent pass (``StandardMode._tool_call_loop``)
        -> ``Critic``   (Verifier role — independent verification)
        -> ``Refiner``  (Reviser role — targeted correction)
        -> loop until PASS / UNCERTAIN-accepted / max_retries

Key behaviours:

1. **Explicit verification separation** — the ``Critic`` runs
   independently from the primary agent so it can catch errors the
   primary pass cannot self-detect.
2. **Rich context for the Critic** — adaptive context limits via
   ``CriticLimits`` (reasoning vs standard models).
3. **Targeted revision (F1)** — the ``Refiner`` corrects only what the
   ``Critic`` flagged, re-synthesising from an existing tool-result
   summary rather than re-exploring from scratch.
4. **Uncertain-with-high-score acceptance** — an UNCERTAIN verdict with
   ``score >= 0.7`` is accepted on the first attempt; later attempts
   use a looser ``0.3`` threshold.

Task routing (via ``Decomposer``'s 3-gate checklist):

* **Simple tasks**  — primary agent + validate + Critic/Refiner loop
* **Parallel tasks** — decompose -> parallel sub-agents -> synthesize,
  then the same validate + Critic/Refiner loop.

Modularisation
--------------
Execution detail lives in the ``autonomous`` sub-package so this file
stays a thin dispatcher:

* ``autonomous.telemetry``      — tracer record helpers.
* ``autonomous.helpers``        — pure utility functions.
* ``autonomous.critic_runner``  — one ``Critic`` pass.
* ``autonomous.refiner_runner`` — one ``Refiner`` pass + fallback.
* ``autonomous.simple_runner``  — simple-path orchestration.
* ``autonomous.complex_runner`` — complex-path orchestration.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import (
    Critic,
    CritiqueResult,
)
from nucleusiq.agents.components.decomposer import Decomposer, TaskAnalysis
from nucleusiq.agents.components.refiner import Refiner, RevisionCandidate
from nucleusiq.agents.components.validation import ValidationPipeline
from nucleusiq.agents.modes.autonomous import helpers, telemetry
from nucleusiq.agents.modes.autonomous.complex_runner import ComplexRunner
from nucleusiq.agents.modes.autonomous.critic_runner import CriticRunner
from nucleusiq.agents.modes.autonomous.parallel_runner import ParallelRunner
from nucleusiq.agents.modes.autonomous.refiner_runner import RefinerRunner
from nucleusiq.agents.modes.autonomous.simple_runner import SimpleRunner
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.task import Task
from nucleusiq.streaming.events import StreamEvent


class AutonomousMode(BaseExecutionMode):
    """Gear 3 — structured orchestration over Standard mode.

    Public API:
      * ``run``         — dispatch sync.
      * ``run_stream``  — dispatch streaming.

    Internal methods (``_run_simple``, ``_run_complex``, ``_run_critic``,
    ``_run_refiner``) are thin delegators kept for test compatibility
    and to document the main collaboration points; all real work lives
    in the ``autonomous`` sub-package.
    """

    # ------------------------------------------------------------------ #
    # Static delegators (test compatibility + documentation)              #
    # ------------------------------------------------------------------ #
    #
    # The original monolithic ``AutonomousMode`` exposed these as static
    # methods; preserving them avoids breaking callers / tests that
    # reference ``AutonomousMode._is_error``, etc.

    _is_error = staticmethod(helpers.is_error_result)
    _build_validation_retry = staticmethod(helpers.build_validation_retry)
    _build_fallback_revision_message = staticmethod(
        helpers.build_fallback_revision_message
    )
    _select_critic_limits = staticmethod(helpers.select_critic_limits)
    _summarize_tool_results = staticmethod(helpers.summarize_tool_results)

    _record_validation = staticmethod(telemetry.record_validation)
    _record_revision = staticmethod(telemetry.record_revision)
    _record_critic_verdict = staticmethod(telemetry.record_critic_verdict)
    _set_autonomous_detail = staticmethod(telemetry.set_autonomous_detail)
    _rollup_sub_agent_metrics = staticmethod(telemetry.rollup_sub_agent_metrics)

    # ------------------------------------------------------------------ #
    # Dispatcher: sync                                                    #
    # ------------------------------------------------------------------ #

    async def run(self, agent: "Agent", task: Any) -> Any:
        if isinstance(task, dict):
            task = Task.from_dict(task)

        agent._logger.debug("Executing in AUTONOMOUS mode")

        if not agent.llm:
            agent._logger.warning("No LLM — falling back to standard mode")
            return await StandardMode().run(agent, task)

        await self.store_task_in_memory(agent, task)

        decomposer = Decomposer(logger=agent._logger)
        analysis = await decomposer.analyze(agent, task)

        n = int(getattr(agent.config, "n_parallel_attempts", 1) or 1)
        is_complex = analysis.is_complex and len(analysis.sub_tasks) >= 2

        if is_complex:
            agent._logger.info(
                "Task classified as COMPLEX (%d sub-tasks) — decomposing",
                len(analysis.sub_tasks),
            )
            if n <= 1:
                return await self._run_complex(agent, task, decomposer, analysis)
            agent._logger.info(
                "F4 Best-of-%d enabled — running %d independent complex "
                "attempts sequentially",
                n,
                n,
            )
            runner = ParallelRunner(
                n=n,
                run_one_sync=lambda: self._run_complex(
                    agent, task, decomposer, analysis
                ),
            )
            return await runner.run_sync(agent)

        agent._logger.info(
            "Task classified as SIMPLE — standard + validate + Critic"
        )
        if n <= 1:
            return await self._run_simple(agent, task)
        agent._logger.info(
            "F4 Best-of-%d enabled — running %d independent simple attempts "
            "sequentially",
            n,
            n,
        )
        runner = ParallelRunner(
            n=n,
            run_one_sync=lambda: self._run_simple(agent, task),
        )
        return await runner.run_sync(agent)

    # ------------------------------------------------------------------ #
    # Dispatcher: streaming                                               #
    # ------------------------------------------------------------------ #

    async def run_stream(
        self, agent: "Agent", task: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        if isinstance(task, dict):
            task = Task.from_dict(task)

        agent._logger.debug("Streaming in AUTONOMOUS mode")

        if not agent.llm:
            agent._logger.warning("No LLM — falling back to standard streaming")
            async for event in StandardMode().run_stream(agent, task):
                yield event
            return

        await self.store_task_in_memory(agent, task)

        yield StreamEvent.thinking_event("Analyzing task complexity…")

        decomposer = Decomposer(logger=agent._logger)
        analysis = await decomposer.analyze(agent, task)

        n = int(getattr(agent.config, "n_parallel_attempts", 1) or 1)
        is_complex = analysis.is_complex and len(analysis.sub_tasks) >= 2

        if is_complex:
            agent._logger.info(
                "Task classified as COMPLEX (%d sub-tasks) — decomposing",
                len(analysis.sub_tasks),
            )
            if n <= 1:
                async for event in self._stream_complex(
                    agent, task, decomposer, analysis
                ):
                    yield event
                return
            runner = ParallelRunner(
                n=n,
                run_one_sync=lambda: self._run_complex(
                    agent, task, decomposer, analysis
                ),
                run_one_stream=lambda: self._stream_complex(
                    agent, task, decomposer, analysis
                ),
            )
            async for event in runner.run_stream(agent, task):
                yield event
            return

        agent._logger.info(
            "Task classified as SIMPLE — streaming standard + Critic"
        )
        if n <= 1:
            async for event in self._stream_simple(agent, task):
                yield event
            return
        runner = ParallelRunner(
            n=n,
            run_one_sync=lambda: self._run_simple(agent, task),
            run_one_stream=lambda: self._stream_simple(agent, task),
        )
        async for event in runner.run_stream(agent, task):
            yield event

    # ------------------------------------------------------------------ #
    # Simple path                                                         #
    # ------------------------------------------------------------------ #

    async def _run_simple(self, agent: "Agent", task: Task) -> Any:
        """Simple path sync entrypoint. Delegates to ``SimpleRunner``."""
        runner = self._build_simple_runner(agent)
        return await runner.run_sync(agent, task)

    async def _stream_simple(
        self, agent: "Agent", task: Task
    ) -> AsyncGenerator[StreamEvent, None]:
        """Simple path streaming entrypoint."""
        runner = self._build_simple_runner(agent)
        async for event in runner.run_stream(agent, task):
            yield event

    def _build_simple_runner(self, agent: "Agent") -> SimpleRunner:
        """Instantiate ``SimpleRunner`` with module-level collaborators.

        Using the module-level ``StandardMode`` / ``ValidationPipeline``
        references (not the sub-package ones) keeps the existing test
        patch points functional.
        """
        return SimpleRunner(
            mode=self,
            std_mode=StandardMode(),
            validation=ValidationPipeline(logger=agent._logger),
            critic=Critic(
                logger=agent._logger,
                limits=helpers.select_critic_limits(agent),
            ),
            refiner=Refiner(logger=agent._logger),
        )

    # ------------------------------------------------------------------ #
    # Complex path                                                        #
    # ------------------------------------------------------------------ #

    async def _run_complex(
        self,
        agent: "Agent",
        task: Task,
        decomposer: Decomposer,
        analysis: TaskAnalysis,
    ) -> Any:
        """Complex path sync entrypoint. Delegates to ``ComplexRunner``."""
        runner = self._build_complex_runner(agent)
        return await runner.run_sync(agent, task, decomposer, analysis)

    async def _stream_complex(
        self,
        agent: "Agent",
        task: Task,
        decomposer: Decomposer,
        analysis: TaskAnalysis,
    ) -> AsyncGenerator[StreamEvent, None]:
        runner = self._build_complex_runner(agent)
        async for event in runner.run_stream(agent, task, decomposer, analysis):
            yield event

    def _build_complex_runner(self, agent: "Agent") -> ComplexRunner:
        return ComplexRunner(
            mode=self,
            std_mode=StandardMode(),
            validation=ValidationPipeline(logger=agent._logger),
            critic=Critic(
                logger=agent._logger,
                limits=helpers.select_critic_limits(agent),
            ),
            refiner=Refiner(logger=agent._logger),
        )

    # ------------------------------------------------------------------ #
    # Critic / Refiner — kept as methods so tests can patch them          #
    # ------------------------------------------------------------------ #

    async def _run_critic(
        self,
        agent: "Agent",
        critic: Critic,
        task_objective: str,
        result: Any,
        messages: list[ChatMessage],
    ) -> CritiqueResult:
        """Run one Critic verification pass. Thin delegate to ``CriticRunner``."""
        return await CriticRunner(self, critic).run(
            agent, task_objective, result, messages
        )

    async def _run_refiner(
        self,
        agent: "Agent",
        refiner: Refiner,
        task_objective: str,
        candidate: Any,
        critique: CritiqueResult,
        messages: list[ChatMessage],
    ) -> RevisionCandidate | None:
        """Run one Refiner pass. Thin delegate to ``RefinerRunner``."""
        return await RefinerRunner(refiner).run(
            agent, task_objective, candidate, critique, messages
        )
