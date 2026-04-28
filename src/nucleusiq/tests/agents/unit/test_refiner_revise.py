"""
Tests for the F1 Refiner.revise() role surface.

Covers:
    * RevisionCandidate model shape and immutability.
    * Refiner.revise success path — returns a RevisionCandidate with the
      correct content, char_delta, addressed_issues and duration_ms.
    * Refiner.revise tags its LLM call with CallPurpose.REFINER so the
      usage tracker keeps Generator / Verifier / Reviser traffic separated.
    * Refiner.revise raises when agent.llm is missing; caller handles
      graceful degradation.
    * AutonomousMode._summarize_tool_results bounds its output.
    * AutonomousMode._run_refiner returns None on non-fatal Refiner failure
      and records a RevisionRecord on success.
    * AutonomousMode._record_critic_verdict appends a CritiqueSnapshot.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nucleusiq.agents.agent_result import (
    CritiqueSnapshot,
    RevisionRecord,
)
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import CritiqueResult, Verdict
from nucleusiq.agents.components.refiner import Refiner, RevisionCandidate
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode
from nucleusiq.agents.usage.usage_tracker import CallPurpose

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


class _FakeTracer:
    """Minimal tracer that records autonomous_detail updates and tool calls."""

    def __init__(self, tool_calls: tuple = ()) -> None:
        self._autonomous_detail: dict[str, Any] = {}
        self._tool_calls = tuple(tool_calls)

    @property
    def autonomous_detail(self) -> dict[str, Any]:
        return dict(self._autonomous_detail)

    @property
    def tool_calls(self) -> tuple:
        return self._tool_calls

    def set_autonomous_detail(self, **kwargs: Any) -> None:
        self._autonomous_detail.update(kwargs)


def _make_agent(*, has_llm: bool = True) -> MagicMock:
    agent = MagicMock()
    agent.name = "refiner-test-agent"
    agent._logger = MagicMock()
    agent.config = AgentConfig()
    agent.tools = []
    agent._current_llm_overrides = {}
    agent._plugin_manager = None
    agent._tracer = _FakeTracer()
    agent.memory = None
    agent._executor = MagicMock()

    if has_llm:
        llm = MagicMock()
        llm.model_name = "test-model"
        llm.convert_tool_specs = MagicMock(return_value=[])
        llm.call = AsyncMock()
        agent.llm = llm
    else:
        agent.llm = None

    return agent


def _critique(
    verdict: Verdict = Verdict.FAIL,
    score: float = 0.25,
    issues: list[str] | None = None,
    suggestions: list[str] | None = None,
    feedback: str = "Needs work",
) -> CritiqueResult:
    return CritiqueResult(
        verdict=verdict,
        score=score,
        feedback=feedback,
        issues=issues or ["Missing data point"],
        suggestions=suggestions or ["Use the figure from the tool result"],
    )


# ------------------------------------------------------------------ #
# RevisionCandidate model                                              #
# ------------------------------------------------------------------ #


class TestRevisionCandidate:
    def test_defaults(self):
        rc = RevisionCandidate(content="hello")
        assert rc.content == "hello"
        assert rc.addressed_issues == ()
        assert rc.tool_calls_made == 0
        assert rc.char_delta == 0
        assert rc.duration_ms == 0.0

    def test_is_frozen(self):
        rc = RevisionCandidate(content="hello")
        with pytest.raises(Exception):
            rc.content = "changed"  # type: ignore[misc]


# ------------------------------------------------------------------ #
# Refiner.revise                                                       #
# ------------------------------------------------------------------ #


class TestRefinerRevise:
    @pytest.mark.asyncio
    async def test_raises_without_llm(self):
        refiner = Refiner()
        agent = _make_agent(has_llm=False)

        with pytest.raises(RuntimeError):
            await refiner.revise(
                agent=agent,
                task_objective="objective",
                candidate="old",
                critique=_critique(),
            )

    @pytest.mark.asyncio
    async def test_success_returns_revision_candidate(self):
        refiner = Refiner()
        agent = _make_agent()

        with patch(
            "nucleusiq.agents.modes.standard_mode.StandardMode._tool_call_loop",
            new_callable=AsyncMock,
        ) as loop_mock:
            loop_mock.return_value = "revised answer"

            out = await refiner.revise(
                agent=agent,
                task_objective="Summarise Q3 revenue",
                candidate="old short",
                critique=_critique(),
            )

        assert isinstance(out, RevisionCandidate)
        assert out.content == "revised answer"
        assert out.char_delta == len("revised answer") - len("old short")
        assert out.addressed_issues == ("Missing data point",)
        assert out.tool_calls_made == 0
        assert out.duration_ms >= 0.0

    @pytest.mark.asyncio
    async def test_tags_calls_as_refiner_purpose(self):
        """The Reviser must tag its LLM calls with CallPurpose.REFINER so
        telemetry keeps Generator / Verifier / Reviser traffic separate."""
        refiner = Refiner()
        agent = _make_agent()

        observed: dict[str, Any] = {}

        async def fake_loop(
            self_std,
            agent_arg,
            task_arg,
            messages,
            tool_specs,
            *,
            purpose_override=None,
        ) -> str:
            observed["purpose_override"] = purpose_override
            observed["task_id"] = task_arg.id
            observed["first_role"] = messages[0].role if messages else None
            return "rev"

        with patch(
            "nucleusiq.agents.modes.standard_mode.StandardMode._tool_call_loop",
            new=fake_loop,
        ):
            await refiner.revise(
                agent=agent,
                task_objective="obj",
                candidate="prev",
                critique=_critique(),
            )

        assert observed["purpose_override"] is CallPurpose.REFINER
        assert observed["task_id"] == "refiner_pass"
        assert observed["first_role"] == "system"

    @pytest.mark.asyncio
    async def test_counts_tool_calls_via_tracer_delta(self):
        refiner = Refiner()
        agent = _make_agent()
        # Pretend the tracer saw 2 tool calls *before* this revise, and 5 after.
        # Because the tuple is read each time, update it between before/after.
        tracer = agent._tracer

        tracer._tool_calls = (1, 2)  # len == 2 at start

        async def fake_loop(*_, **__):
            # Simulate the tool loop recording 3 more tool calls.
            tracer._tool_calls = (1, 2, 3, 4, 5)
            return "ok"

        with patch(
            "nucleusiq.agents.modes.standard_mode.StandardMode._tool_call_loop",
            new=fake_loop,
        ):
            out = await refiner.revise(
                agent=agent,
                task_objective="obj",
                candidate="prev",
                critique=_critique(),
            )
        assert out.tool_calls_made == 3

    @pytest.mark.asyncio
    async def test_truncates_candidate_and_tool_summary_in_prompt(self):
        """The Reviser must bound oversized inputs to stay within budget."""
        refiner = Refiner()
        agent = _make_agent()

        big_candidate = "x" * 20_000
        big_tool_summary = "t" * 20_000

        seen_prompt: dict[str, str] = {}

        async def fake_loop(
            self_std,
            agent_arg,
            task_arg,
            messages,
            tool_specs,
            *,
            purpose_override=None,
        ) -> str:
            user_msg = next(m for m in messages if m.role == "user")
            seen_prompt["content"] = user_msg.content
            return "short"

        with patch(
            "nucleusiq.agents.modes.standard_mode.StandardMode._tool_call_loop",
            new=fake_loop,
        ):
            await refiner.revise(
                agent=agent,
                task_objective="obj",
                candidate=big_candidate,
                critique=_critique(),
                tool_result_summary=big_tool_summary,
            )

        body = seen_prompt["content"]
        # Candidate and tool summary must both be truncated well below
        # the raw 20 KB inputs. The prompt template contributes a few
        # literal 'x'/'t' characters (e.g. the "Fix" line doesn't but
        # allow a small headroom) so we test against a loose ceiling.
        assert body.count("x") < 9_000, "candidate should be truncated"
        assert body.count("t") < 5_000, "tool summary should be truncated"


# ------------------------------------------------------------------ #
# AutonomousMode helpers                                               #
# ------------------------------------------------------------------ #


class TestSummarizeToolResults:
    def test_returns_none_when_no_tool_messages(self):
        messages = [
            ChatMessage(role="system", content="hi"),
            ChatMessage(role="user", content="hey"),
            ChatMessage(role="assistant", content="yo"),
        ]
        assert AutonomousMode._summarize_tool_results(messages) is None

    def test_total_cap_bounds_aggregate_output(self):
        # v0.7.8: per-tool head-truncation has been removed.  The
        # remaining bound is the aggregate ``total_char_cap``.
        # Use 200-char entries so the cap stops us mid-stream rather
        # than before the first entry.
        tool_body = "z" * 200
        messages = [
            ChatMessage(role="tool", name="fetch", content=tool_body) for _ in range(20)
        ]
        summary = AutonomousMode._summarize_tool_results(messages, total_char_cap=1_000)
        assert summary is not None
        assert len(summary) <= 1_000
        assert "[fetch]" in summary
        # Not all 20 entries fit; we stopped early.
        assert summary.count("[fetch]") < 20

    def test_no_truncation_by_default(self):
        # v0.7.8: without caps the helper passes content through.
        messages = [
            ChatMessage(role="tool", name="fetch", content="x" * 3_000),
            ChatMessage(role="tool", name="fetch", content="y" * 3_000),
        ]
        summary = AutonomousMode._summarize_tool_results(messages)
        assert summary is not None
        assert summary.count("x") == 3_000
        assert summary.count("y") == 3_000


class TestRunRefiner:
    @pytest.mark.asyncio
    async def test_non_fatal_failure_returns_none(self):
        mode = AutonomousMode()
        refiner = Refiner()
        agent = _make_agent()

        with patch.object(
            Refiner, "revise", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            out = await mode._run_refiner(
                agent,
                refiner,
                "obj",
                "candidate",
                _critique(),
                [],
            )
        assert out is None
        agent._logger.warning.assert_called()  # logged, not raised

    @pytest.mark.asyncio
    async def test_success_returns_revision(self):
        mode = AutonomousMode()
        refiner = Refiner()
        agent = _make_agent()

        expected = RevisionCandidate(content="revised")

        with patch.object(Refiner, "revise", new=AsyncMock(return_value=expected)):
            out = await mode._run_refiner(
                agent,
                refiner,
                "obj",
                "candidate",
                _critique(),
                [],
            )
        assert out is expected


# ------------------------------------------------------------------ #
# Telemetry                                                            #
# ------------------------------------------------------------------ #


class TestTelemetryRecording:
    def test_record_revision_appends_to_tracer(self):
        agent = _make_agent()
        critique = _critique()
        rev = RevisionCandidate(
            content="new",
            addressed_issues=("Missing data point",),
            tool_calls_made=1,
            char_delta=5,
            duration_ms=123.4,
        )

        AutonomousMode._record_revision(
            agent, attempt=2, critique=critique, revision=rev
        )

        ad = agent._tracer.autonomous_detail
        revisions = ad["revisions"]
        assert len(revisions) == 1
        record = revisions[0]
        assert isinstance(record, RevisionRecord)
        assert record.attempt == 2
        assert record.triggered_by_verdict == "fail"
        assert record.triggered_by_score == pytest.approx(0.25)
        assert record.char_delta == 5
        assert record.tool_calls_made == 1
        assert record.addressed_issues == ("Missing data point",)
        assert record.duration_ms == pytest.approx(123.4)

    def test_record_revision_is_cumulative(self):
        agent = _make_agent()
        rev = RevisionCandidate(content="x")
        AutonomousMode._record_revision(agent, 1, _critique(), rev)
        AutonomousMode._record_revision(agent, 2, _critique(), rev)
        assert len(agent._tracer.autonomous_detail["revisions"]) == 2

    def test_record_critic_verdict_appends_snapshot(self):
        agent = _make_agent()
        critique = _critique(
            verdict=Verdict.UNCERTAIN,
            score=0.55,
            issues=["I1"],
            suggestions=["S1"],
            feedback="hm",
        )

        AutonomousMode._record_critic_verdict(agent, attempt=1, critique=critique)

        ad = agent._tracer.autonomous_detail
        verdicts = ad["critic_verdicts"]
        assert len(verdicts) == 1
        snap = verdicts[0]
        assert isinstance(snap, CritiqueSnapshot)
        assert snap.attempt == 1
        assert snap.verdict == "uncertain"
        assert snap.score == pytest.approx(0.55)
        assert snap.feedback == "hm"
        assert snap.issues == ("I1",)
        assert snap.suggestions == ("S1",)

    def test_record_critic_verdict_is_cumulative(self):
        agent = _make_agent()
        AutonomousMode._record_critic_verdict(agent, 1, _critique())
        AutonomousMode._record_critic_verdict(agent, 2, _critique(verdict=Verdict.PASS))
        assert len(agent._tracer.autonomous_detail["critic_verdicts"]) == 2
