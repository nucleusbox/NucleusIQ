"""F2 — Critic strict gate + abstention tests.

Covers the contract introduced by the Aletheia alignment work:

    * ``ResultStatus.ABSTAINED`` is a first-class outcome.
    * ``AgentResult.abstention_reason`` surfaces a machine-readable
      reason (pulled from the final Critic feedback).
    * ``AgentResult.is_abstained`` is a convenience boolean.
    * When every retry's Critic pass ends in FAIL / UNCERTAIN-below-
      threshold, ``SimpleRunner`` and ``ComplexRunner`` raise
      ``AbstentionSignal`` carrying the *best candidate* so callers can
      still inspect what the agent produced.
    * ``Agent.execute()`` converts ``AbstentionSignal`` into an
      ``AgentResult`` with ``status == ABSTAINED`` — no exception leaks.
    * ``Agent.execute_stream()`` emits a terminal ``complete_event``
      (best candidate) followed by an ``error_event`` tagged ``ABSTAINED``.
    * The single fixed ``UNCERTAIN`` accept threshold (0.7) is used on
      every attempt — no diminishing strictness, no final-attempt
      bypass.
    * No user-facing knob: abstention is always on.  There is no
      ``AgentConfig.allow_abstention`` attribute.

The tests in this file stub out ``AutonomousMode._run_critic`` directly
because F2 is entirely about how the runner interprets the Critic's
verdict — not about producing that verdict end-to-end.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import (
    AbstentionSignal,
    AgentResult,
    ResultStatus,
)
from nucleusiq.agents.components.critic import CritiqueResult, Verdict
from nucleusiq.agents.config import AgentConfig, AgentState, ExecutionMode
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt


def _make_agent(**overrides) -> Agent:
    defaults = dict(
        name="AbstainAgent",
        role="Analyst",
        objective="Answer questions accurately",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            max_retries=2,
        ),
    )
    defaults.update(overrides)
    return Agent(**defaults)


def _make_task(objective: str = "solve X") -> Task:
    return Task(id="t-abs", objective=objective)


def _pass_critic() -> AsyncMock:
    return AsyncMock(
        return_value=CritiqueResult(verdict=Verdict.PASS, score=1.0, feedback="ok"),
    )


def _fail_critic(feedback: str = "answer is wrong") -> AsyncMock:
    return AsyncMock(
        return_value=CritiqueResult(
            verdict=Verdict.FAIL, score=0.1, feedback=feedback,
        ),
    )


def _uncertain_critic(score: float = 0.5) -> AsyncMock:
    return AsyncMock(
        return_value=CritiqueResult(
            verdict=Verdict.UNCERTAIN, score=score, feedback="not sure",
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ResultStatus / AgentResult contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultStatusAbstainedContract:
    """ResultStatus now exposes ABSTAINED as a first-class outcome."""

    def test_abstained_enum_value(self):
        assert ResultStatus.ABSTAINED.value == "abstained"

    def test_abstained_is_distinct_from_success_and_error(self):
        assert ResultStatus.ABSTAINED is not ResultStatus.SUCCESS
        assert ResultStatus.ABSTAINED is not ResultStatus.ERROR
        assert ResultStatus.ABSTAINED is not ResultStatus.HALTED

    def test_all_four_values(self):
        assert {s.value for s in ResultStatus} == {
            "success",
            "error",
            "halted",
            "abstained",
        }


class TestAgentResultAbstentionFields:
    """``abstention_reason`` + ``is_abstained`` must be plumbed end-to-end."""

    def _make_result(self, **overrides) -> AgentResult:
        defaults = dict(
            agent_id="a-1",
            agent_name="A",
            mode="autonomous",
            task_id="t",
            task="x",
            status=ResultStatus.SUCCESS,
            output="hi",
            started_at=0.0,
            ended_at=1.0,
        )
        defaults.update(overrides)
        return AgentResult(**defaults)

    def test_default_abstention_reason_is_none(self):
        r = self._make_result()
        assert r.abstention_reason is None
        assert r.is_abstained is False

    def test_abstained_result_exposes_reason(self):
        r = self._make_result(
            status=ResultStatus.ABSTAINED,
            output="best-effort",
            abstention_reason="Critic rejected final candidate",
        )
        assert r.is_abstained is True
        assert r.abstention_reason == "Critic rejected final candidate"


class TestAbstentionSignal:
    """``AbstentionSignal`` carries the payload the agent needs."""

    def test_signal_default_reason_uses_critique_feedback(self):
        critique = CritiqueResult(
            verdict=Verdict.FAIL, score=0.2, feedback="missing sources"
        )
        signal = AbstentionSignal(best_candidate="draft", critique=critique)
        assert signal.best_candidate == "draft"
        assert signal.critique is critique
        assert "missing sources" in signal.reason

    def test_signal_explicit_reason_overrides_feedback(self):
        critique = CritiqueResult(verdict=Verdict.FAIL, score=0.0, feedback="nope")
        signal = AbstentionSignal(
            best_candidate=None, critique=critique, reason="custom reason"
        )
        assert signal.reason == "custom reason"

    def test_signal_is_exception(self):
        critique = CritiqueResult(verdict=Verdict.FAIL, score=0.0)
        assert isinstance(AbstentionSignal("x", critique), Exception)


class TestNoAllowAbstentionKnob:
    """Abstention is always on — no user-facing config knob."""

    def test_agent_config_has_no_allow_abstention(self):
        cfg = AgentConfig()
        assert not hasattr(cfg, "allow_abstention")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner-level behaviour
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimpleRunnerAbstention:
    """``SimpleRunner`` raises ``AbstentionSignal`` when every Critic pass fails."""

    @pytest.mark.asyncio
    async def test_raises_when_critic_fails_every_attempt(self):
        agent = _make_agent()
        mode = AutonomousMode()

        with patch.object(AutonomousMode, "_run_critic", new=_fail_critic("bad")):
            with pytest.raises(AbstentionSignal) as excinfo:
                await mode._run_simple(agent, _make_task())

        assert excinfo.value.best_candidate
        assert excinfo.value.critique.verdict == Verdict.FAIL

    @pytest.mark.asyncio
    async def test_no_abstention_when_critic_passes(self):
        agent = _make_agent()
        mode = AutonomousMode()

        with patch.object(AutonomousMode, "_run_critic", new=_pass_critic()):
            result = await mode._run_simple(agent, _make_task())

        assert result is not None
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_uncertain_above_threshold_is_accepted(self):
        """UNCERTAIN with score >= 0.7 counts as accept (single threshold)."""
        agent = _make_agent()
        mode = AutonomousMode()

        critic = AsyncMock(
            return_value=CritiqueResult(
                verdict=Verdict.UNCERTAIN, score=0.75, feedback="close"
            )
        )
        with patch.object(AutonomousMode, "_run_critic", new=critic):
            result = await mode._run_simple(agent, _make_task())

        assert result is not None
        assert agent.state == AgentState.COMPLETED
        critic.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uncertain_below_threshold_abstains(self):
        """UNCERTAIN with score < 0.7 on *every* retry -> abstain."""
        agent = _make_agent(config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS, max_retries=2,
        ))
        mode = AutonomousMode()

        critic = _uncertain_critic(score=0.5)
        with patch.object(AutonomousMode, "_run_critic", new=critic):
            with pytest.raises(AbstentionSignal) as excinfo:
                await mode._run_simple(agent, _make_task())

        assert excinfo.value.critique.verdict == Verdict.UNCERTAIN


class TestNoFinalAttemptBypass:
    """F2 removes the pre-F2 'accept on last attempt' safety valve."""

    @pytest.mark.asyncio
    async def test_critic_runs_on_final_attempt(self):
        """With ``max_retries=1`` the Critic is called exactly once and the
        outcome governs whether we abstain."""
        agent = _make_agent(config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS, max_retries=1,
        ))
        mode = AutonomousMode()

        critic = _fail_critic()
        with patch.object(AutonomousMode, "_run_critic", new=critic):
            with pytest.raises(AbstentionSignal):
                await mode._run_simple(agent, _make_task())

        critic.assert_awaited()


# ═══════════════════════════════════════════════════════════════════════════════
# Agent.execute() — converts AbstentionSignal into AgentResult
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentExecuteAbstention:
    """``Agent.execute()`` must not leak ``AbstentionSignal`` to callers."""

    @pytest.mark.asyncio
    async def test_execute_returns_abstained_result(self):
        agent = _make_agent()

        with patch.object(
            AutonomousMode, "_run_critic", new=_fail_critic("no evidence")
        ):
            result = await agent.execute(_make_task())

        assert isinstance(result, AgentResult)
        assert result.status == ResultStatus.ABSTAINED
        assert result.is_abstained is True
        assert result.abstention_reason  # non-empty
        assert "no evidence" in (result.abstention_reason or "")
        assert result.output  # best candidate preserved

    @pytest.mark.asyncio
    async def test_execute_returns_success_when_critic_passes(self):
        agent = _make_agent()

        with patch.object(AutonomousMode, "_run_critic", new=_pass_critic()):
            result = await agent.execute(_make_task())

        assert result.status == ResultStatus.SUCCESS
        assert result.is_abstained is False
        assert result.abstention_reason is None


class TestAgentExecuteStreamAbstention:
    """Streaming path surfaces abstention as complete + error terminal
    events so UIs can still render the best candidate while tagging the
    run as abstained."""

    @pytest.mark.asyncio
    async def test_stream_emits_complete_and_error_on_abstention(self):
        agent = _make_agent()

        with patch.object(
            AutonomousMode, "_run_critic", new=_fail_critic("bad output")
        ):
            events = []
            async for event in agent.execute_stream(_make_task()):
                events.append(event)

        types = [e.type for e in events]
        assert "complete" in types
        assert "error" in types

        error_evs = [e for e in events if e.type == "error"]
        assert any("ABSTAINED" in (e.message or "") for e in error_evs)
