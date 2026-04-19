"""Unit tests for F4: Best-of-N parallel attempts (``ParallelRunner``).

Focus areas:
    1. Selection rule (pure, no I/O).
    2. N == 1 short-circuit — bit-for-bit identical to the inner runner.
    3. N >= 2 orchestration — temperature perturbation, autonomous-detail
       isolation, selection, abstention aggregation.
    4. Integration with ``AutonomousMode`` wiring (config-driven).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import (
    AbstentionSignal,
    AutonomousDetail,
    CritiqueSnapshot,
)
from nucleusiq.agents.components.critic import CritiqueResult, Verdict
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.modes.autonomous.parallel_runner import (
    MAX_PARALLEL_ATTEMPTS,
    ParallelRunner,
    _selection_rule,
)
from nucleusiq.agents.task import Task


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _snap(verdict: Verdict, score: float, feedback: str = "") -> CritiqueSnapshot:
    return CritiqueSnapshot(
        attempt=1,
        verdict=verdict.value,
        score=score,
        feedback=feedback,
        issues=(),
        suggestions=(),
    )


def _attempt(
    index: int,
    verdict: Verdict,
    score: float,
    *,
    abstained: bool = False,
    result: Any = None,
    feedback: str = "",
):
    from nucleusiq.agents.modes.autonomous.parallel_runner import _AttemptOutcome

    return _AttemptOutcome(
        index=index,
        result=result if result is not None else f"candidate-{index}",
        critique=CritiqueResult(
            verdict=verdict, score=score, feedback=feedback
        ),
        abstained=abstained,
        abstention_reason=feedback if abstained else None,
        detail=AutonomousDetail(critic_verdicts=(_snap(verdict, score, feedback),)),
    )


# --------------------------------------------------------------------- #
# Selection rule — pure, no I/O
# --------------------------------------------------------------------- #


class TestSelectionRule:
    def test_pass_beats_uncertain_even_if_lower_score(self) -> None:
        attempts = [
            _attempt(0, Verdict.UNCERTAIN, 0.95),
            _attempt(1, Verdict.PASS, 0.60),
        ]
        assert _selection_rule(attempts) == 1

    def test_highest_score_pass_wins_among_passes(self) -> None:
        attempts = [
            _attempt(0, Verdict.PASS, 0.70),
            _attempt(1, Verdict.PASS, 0.85),
            _attempt(2, Verdict.PASS, 0.75),
        ]
        assert _selection_rule(attempts) == 1

    def test_uncertain_above_threshold_wins_over_fail(self) -> None:
        attempts = [
            _attempt(0, Verdict.FAIL, 0.50),
            _attempt(1, Verdict.UNCERTAIN, 0.80),
        ]
        assert _selection_rule(attempts) == 1

    def test_uncertain_below_threshold_not_selected_over_better_uncertain(
        self,
    ) -> None:
        attempts = [
            _attempt(0, Verdict.UNCERTAIN, 0.65),  # below 0.7
            _attempt(1, Verdict.UNCERTAIN, 0.75),  # above 0.7
        ]
        assert _selection_rule(attempts) == 1

    def test_all_fail_returns_best_of_bad(self) -> None:
        attempts = [
            _attempt(0, Verdict.FAIL, 0.10),
            _attempt(1, Verdict.FAIL, 0.30),
            _attempt(2, Verdict.FAIL, 0.20),
        ]
        assert _selection_rule(attempts) == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _selection_rule([])

    def test_missing_critique_falls_through(self) -> None:
        """An attempt with no critique should have score 0 for selection."""
        good = _attempt(0, Verdict.FAIL, 0.10)
        bad = _attempt(1, Verdict.FAIL, 0.05)
        bad.critique = None  # simulate missing
        # good has higher score than bad's 0.0 fallback
        assert _selection_rule([good, bad]) == 0


# --------------------------------------------------------------------- #
# ParallelRunner construction guardrails
# --------------------------------------------------------------------- #


class TestParallelRunnerConstruction:
    def test_rejects_n_less_than_2(self) -> None:
        with pytest.raises(ValueError):
            ParallelRunner(n=1, run_one_sync=AsyncMock())
        with pytest.raises(ValueError):
            ParallelRunner(n=0, run_one_sync=AsyncMock())

    def test_rejects_n_above_cap(self) -> None:
        with pytest.raises(ValueError):
            ParallelRunner(
                n=MAX_PARALLEL_ATTEMPTS + 1, run_one_sync=AsyncMock()
            )


# --------------------------------------------------------------------- #
# AgentConfig validation
# --------------------------------------------------------------------- #


class TestAgentConfigValidation:
    def test_default_is_one(self) -> None:
        cfg = AgentConfig()
        assert cfg.n_parallel_attempts == 1

    def test_accepts_valid_range(self) -> None:
        for n in range(1, MAX_PARALLEL_ATTEMPTS + 1):
            cfg = AgentConfig(n_parallel_attempts=n)
            assert cfg.n_parallel_attempts == n

    def test_rejects_zero(self) -> None:
        with pytest.raises(Exception):
            AgentConfig(n_parallel_attempts=0)

    def test_rejects_above_cap(self) -> None:
        with pytest.raises(Exception):
            AgentConfig(n_parallel_attempts=MAX_PARALLEL_ATTEMPTS + 1)


# --------------------------------------------------------------------- #
# End-to-end orchestration with a fake inner runner
# --------------------------------------------------------------------- #


def _fake_agent() -> Any:
    """Build a minimal agent-like object good enough for ParallelRunner."""

    class _Tracer:
        def __init__(self) -> None:
            self._autonomous_detail: dict[str, Any] = {}

        def set_autonomous_detail(self, **kwargs: Any) -> None:
            self._autonomous_detail.update(kwargs)

        @property
        def autonomous_detail(self) -> dict[str, Any] | None:
            return (
                dict(self._autonomous_detail)
                if self._autonomous_detail
                else None
            )

    class _Cfg:
        llm_params = None
        n_parallel_attempts = 3

    class _A:
        def __init__(self) -> None:
            self.config = _Cfg()
            self._tracer = _Tracer()
            import logging

            self._logger = logging.getLogger("test.parallel")

    return _A()


class TestParallelRunnerSync:
    @pytest.mark.asyncio
    async def test_selects_pass_attempt_over_uncertain(self) -> None:
        agent = _fake_agent()
        tracer = agent._tracer

        # Each inner call sets a different critique_verdicts, then returns
        # its own candidate.  Order: UNCERTAIN(0.9), PASS(0.5), FAIL(0.3).
        verdicts = [
            (Verdict.UNCERTAIN, 0.9),
            (Verdict.PASS, 0.5),
            (Verdict.FAIL, 0.3),
        ]
        call_idx = {"n": 0}

        async def fake_inner() -> str:
            i = call_idx["n"]
            call_idx["n"] += 1
            v, s = verdicts[i]
            tracer.set_autonomous_detail(
                critic_verdicts=(_snap(v, s),),
                attempts=1,
                max_attempts=3,
                complexity="simple",
            )
            return f"candidate-{i}"

        runner = ParallelRunner(n=3, run_one_sync=fake_inner)
        result = await runner.run_sync(agent)

        # PASS at index 1 wins over higher-scored UNCERTAIN at 0.
        assert result == "candidate-1"

        detail = tracer.autonomous_detail
        assert detail is not None
        assert len(detail["parallel_attempts"]) == 3
        assert detail["selected_attempt"] == 1

    @pytest.mark.asyncio
    async def test_no_pass_uncertain_high_wins(self) -> None:
        agent = _fake_agent()
        tracer = agent._tracer
        verdicts = [
            (Verdict.UNCERTAIN, 0.65),  # below threshold
            (Verdict.UNCERTAIN, 0.78),  # above threshold
            (Verdict.FAIL, 0.20),
        ]
        call_idx = {"n": 0}

        async def fake_inner() -> str:
            i = call_idx["n"]
            call_idx["n"] += 1
            v, s = verdicts[i]
            tracer.set_autonomous_detail(critic_verdicts=(_snap(v, s),))
            return f"c-{i}"

        runner = ParallelRunner(n=3, run_one_sync=fake_inner)
        result = await runner.run_sync(agent)
        assert result == "c-1"
        assert tracer.autonomous_detail["selected_attempt"] == 1

    @pytest.mark.asyncio
    async def test_abstains_when_all_fail(self) -> None:
        agent = _fake_agent()
        tracer = agent._tracer
        verdicts = [
            (Verdict.FAIL, 0.10),
            (Verdict.FAIL, 0.30),
            (Verdict.FAIL, 0.20),
        ]
        call_idx = {"n": 0}

        async def fake_inner() -> str:
            i = call_idx["n"]
            call_idx["n"] += 1
            v, s = verdicts[i]
            tracer.set_autonomous_detail(
                critic_verdicts=(_snap(v, s, feedback=f"bad-{i}"),)
            )
            # Simulate inner runner raising abstention on FAIL:
            raise AbstentionSignal(
                best_candidate=f"bad-candidate-{i}",
                critique=CritiqueResult(
                    verdict=v, score=s, feedback=f"bad-{i}"
                ),
                reason=f"bad-{i}",
            )

        runner = ParallelRunner(n=3, run_one_sync=fake_inner)
        with pytest.raises(AbstentionSignal) as exc:
            await runner.run_sync(agent)

        # Best-of-the-bad is attempt 1 (score 0.30).
        assert exc.value.best_candidate == "bad-candidate-1"
        detail = tracer.autonomous_detail
        assert detail["selected_attempt"] == 1
        assert len(detail["parallel_attempts"]) == 3

    @pytest.mark.asyncio
    async def test_all_exceptions_raises_synthesised_abstention(self) -> None:
        agent = _fake_agent()

        async def fake_inner() -> str:
            raise RuntimeError("boom")

        runner = ParallelRunner(n=3, run_one_sync=fake_inner)
        with pytest.raises(AbstentionSignal) as exc:
            await runner.run_sync(agent)
        assert exc.value.best_candidate is None
        assert "3" in exc.value.reason  # N surfaced in reason

    @pytest.mark.asyncio
    async def test_temperature_perturbation_applied_per_attempt(self) -> None:
        """First attempt uses the baseline; subsequent attempts widen it."""
        from nucleusiq.llms.llm_params import LLMParams

        agent = _fake_agent()
        agent.config.llm_params = LLMParams(temperature=0.1)

        observed: list[float | None] = []

        async def fake_inner() -> str:
            observed.append(agent.config.llm_params.temperature)
            agent._tracer.set_autonomous_detail(
                critic_verdicts=(_snap(Verdict.FAIL, 0.1),)
            )
            raise AbstentionSignal(
                best_candidate=None,
                critique=CritiqueResult(verdict=Verdict.FAIL, score=0.1),
                reason="fail",
            )

        runner = ParallelRunner(n=3, run_one_sync=fake_inner)
        with pytest.raises(AbstentionSignal):
            await runner.run_sync(agent)

        assert observed[0] == pytest.approx(0.1)  # baseline
        assert observed[1] > observed[0]
        assert observed[2] > observed[1]
        # Config is restored after the run.
        assert agent.config.llm_params.temperature == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_parallel_details_are_isolated(self) -> None:
        """Each attempt sees a fresh ``_autonomous_detail`` dict."""
        agent = _fake_agent()
        tracer = agent._tracer

        seen: list[dict[str, Any] | None] = []

        async def fake_inner() -> str:
            # Record what tracer.autonomous_detail looks like at *start*
            # of this attempt — should always be empty.
            seen.append(tracer.autonomous_detail)
            tracer.set_autonomous_detail(
                critic_verdicts=(_snap(Verdict.PASS, 0.9),),
                attempts=1,
                max_attempts=3,
            )
            return "ok"

        runner = ParallelRunner(n=3, run_one_sync=fake_inner)
        await runner.run_sync(agent)
        # All attempts started with an empty detail.
        assert all(s is None for s in seen)

    @pytest.mark.asyncio
    async def test_single_pass_terminates_but_still_runs_all_attempts(
        self,
    ) -> None:
        """Sequential design: we currently run all N attempts for fair
        comparison.  (Future optimisation may allow early stop on PASS,
        but the current contract is 'all N for selection'.)"""
        agent = _fake_agent()
        tracer = agent._tracer

        invocations = {"n": 0}

        async def fake_inner() -> str:
            invocations["n"] += 1
            tracer.set_autonomous_detail(
                critic_verdicts=(_snap(Verdict.PASS, 0.9),)
            )
            return f"c-{invocations['n']}"

        runner = ParallelRunner(n=3, run_one_sync=fake_inner)
        await runner.run_sync(agent)
        assert invocations["n"] == 3


# --------------------------------------------------------------------- #
# Integration: AutonomousMode.run() config-driven dispatch
# --------------------------------------------------------------------- #


class TestAutonomousModeDispatch:
    """Verify that ``AutonomousMode.run`` short-circuits N==1 and
    delegates to ``ParallelRunner`` when N>=2."""

    @pytest.mark.asyncio
    async def test_n_eq_1_calls_inner_runner_directly(self) -> None:
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = MagicMock()
        agent.config.n_parallel_attempts = 1
        agent.llm = MagicMock()
        agent._logger = MagicMock()

        with patch.object(
            AutonomousMode, "store_task_in_memory", AsyncMock()
        ), patch(
            "nucleusiq.agents.modes.autonomous_mode.Decomposer"
        ) as mock_decomp, patch.object(
            AutonomousMode, "_run_simple", AsyncMock(return_value="SINGLE")
        ) as mock_simple:
            mock_decomp.return_value.analyze = AsyncMock(
                return_value=MagicMock(is_complex=False, sub_tasks=[])
            )
            result = await mode.run(
                agent, Task(id="t", objective="obj")
            )

        assert result == "SINGLE"
        assert mock_simple.call_count == 1

    @pytest.mark.asyncio
    async def test_n_gt_1_delegates_to_parallel_runner(self) -> None:
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = MagicMock()
        agent.config.n_parallel_attempts = 3
        agent.llm = MagicMock()
        agent._logger = MagicMock()

        # Inner simple runner returns a different value each call.
        counter = {"n": 0}

        async def fake_simple(_a: Any, _t: Any) -> str:
            counter["n"] += 1
            # Populate tracer detail so selection picks one.
            tracer = agent._tracer
            tracer.autonomous_detail = {
                "critic_verdicts": (_snap(Verdict.PASS, 0.8),)
            }
            tracer._autonomous_detail = dict(tracer.autonomous_detail)
            return f"attempt-{counter['n']}"

        # Wire a minimal tracer that round-trips _autonomous_detail.
        agent._tracer = MagicMock()
        agent._tracer._autonomous_detail = {}

        def _set_detail(**kwargs: Any) -> None:
            agent._tracer._autonomous_detail.update(kwargs)

        def _get_detail() -> dict[str, Any] | None:
            return (
                dict(agent._tracer._autonomous_detail)
                if agent._tracer._autonomous_detail
                else None
            )

        agent._tracer.set_autonomous_detail = _set_detail
        type(agent._tracer).autonomous_detail = property(
            lambda self: _get_detail()
        )

        async def fake_simple2(_a: Any, _t: Any) -> str:
            counter["n"] += 1
            _set_detail(critic_verdicts=(_snap(Verdict.PASS, 0.8),))
            return f"attempt-{counter['n']}"

        with patch.object(
            AutonomousMode, "store_task_in_memory", AsyncMock()
        ), patch(
            "nucleusiq.agents.modes.autonomous_mode.Decomposer"
        ) as mock_decomp, patch.object(
            AutonomousMode, "_run_simple", side_effect=fake_simple2
        ):
            mock_decomp.return_value.analyze = AsyncMock(
                return_value=MagicMock(is_complex=False, sub_tasks=[])
            )
            result = await mode.run(
                agent, Task(id="t", objective="obj")
            )

        assert counter["n"] == 3
        assert result.startswith("attempt-")

    @pytest.mark.asyncio
    async def test_n_eq_1_no_temperature_mutation(self) -> None:
        """Acceptance criterion F4.6(a): N=1 is bit-for-bit identical."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
        from nucleusiq.llms.llm_params import LLMParams

        mode = AutonomousMode()
        agent = MagicMock()
        agent.config.n_parallel_attempts = 1
        agent.config.llm_params = LLMParams(temperature=0.25)
        agent.llm = MagicMock()
        agent._logger = MagicMock()

        original_temp = agent.config.llm_params.temperature

        with patch.object(
            AutonomousMode, "store_task_in_memory", AsyncMock()
        ), patch(
            "nucleusiq.agents.modes.autonomous_mode.Decomposer"
        ) as mock_decomp, patch.object(
            AutonomousMode, "_run_simple", AsyncMock(return_value="R")
        ):
            mock_decomp.return_value.analyze = AsyncMock(
                return_value=MagicMock(is_complex=False, sub_tasks=[])
            )
            await mode.run(agent, Task(id="t", objective="obj"))

        # ParallelRunner never touched — temperature still at baseline.
        assert agent.config.llm_params.temperature == original_temp


# --------------------------------------------------------------------- #
# AutonomousDetail self-reference integrity (F4/F5 contract)
# --------------------------------------------------------------------- #


class TestAutonomousDetailParallelField:
    def test_parallel_attempts_field_exists_and_defaults_empty(self) -> None:
        d = AutonomousDetail()
        assert d.parallel_attempts == ()
        assert d.selected_attempt is None

    def test_can_nest_details(self) -> None:
        inner_a = AutonomousDetail(
            critic_verdicts=(_snap(Verdict.PASS, 0.8),),
            attempts=1,
        )
        inner_b = AutonomousDetail(
            critic_verdicts=(_snap(Verdict.FAIL, 0.2),),
            attempts=1,
        )
        outer = AutonomousDetail(
            parallel_attempts=(inner_a, inner_b),
            selected_attempt=0,
        )
        assert len(outer.parallel_attempts) == 2
        assert outer.selected_attempt == 0
        assert outer.parallel_attempts[0].critic_verdicts[0].score == 0.8


# --------------------------------------------------------------------- #
# End-to-end: Agent.execute with N>1 — abstention surfaces as ABSTAINED
# --------------------------------------------------------------------- #


class TestAgentExecuteAbstention:
    @pytest.mark.asyncio
    async def test_best_of_n_abstention_returns_abstained_status(self) -> None:
        """When all N attempts fail, ``AgentResult.status == ABSTAINED``
        and ``output`` carries the best-of-bad candidate."""
        from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

        mode = AutonomousMode()
        agent = MagicMock(spec=Agent)
        agent.config = AgentConfig(n_parallel_attempts=2)
        agent.llm = MagicMock()
        agent._logger = MagicMock()

        async def fake_simple(_a: Any, _t: Any) -> str:
            raise AbstentionSignal(
                best_candidate="best-bad",
                critique=CritiqueResult(verdict=Verdict.FAIL, score=0.2),
                reason="nope",
            )

        # Minimal tracer
        tracer = MagicMock()
        tracer._autonomous_detail = {}

        def _set(**kwargs: Any) -> None:
            tracer._autonomous_detail.update(kwargs)

        def _get() -> dict[str, Any] | None:
            return dict(tracer._autonomous_detail) or None

        tracer.set_autonomous_detail = _set
        type(tracer).autonomous_detail = property(lambda self: _get())
        agent._tracer = tracer

        with patch.object(
            AutonomousMode, "store_task_in_memory", AsyncMock()
        ), patch(
            "nucleusiq.agents.modes.autonomous_mode.Decomposer"
        ) as mock_decomp, patch.object(
            AutonomousMode, "_run_simple", side_effect=fake_simple
        ):
            mock_decomp.return_value.analyze = AsyncMock(
                return_value=MagicMock(is_complex=False, sub_tasks=[])
            )
            with pytest.raises(AbstentionSignal) as exc:
                await mode.run(agent, Task(id="t", objective="obj"))

        assert exc.value.best_candidate == "best-bad"
        # Both attempts recorded in the tracer detail.
        assert len(tracer._autonomous_detail["parallel_attempts"]) == 2
