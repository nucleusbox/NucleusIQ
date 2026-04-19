"""Unit tests for F3: ``ComputeBudget`` and ``decide_next_action``.

These tests are deliberately provider-agnostic: they exercise the pure
budget / decision math and never spin up an LLM.  Runner-level integration
(config mutation, tracer sync, escalation telemetry) is covered in the
autonomous-mode runner tests; this file locks down the core invariants
that those runners rely on.
"""

from __future__ import annotations

import pytest

from nucleusiq.agents.components.compute_budget import (
    DEFAULT_RUN_TOKEN_CEILING,
    HARD_TOKEN_CEILING,
    IMPROVEMENT_DELTA,
    MAX_ESCALATIONS_PER_RUN,
    TOKEN_ESCALATION_FACTOR,
    TOOL_CALL_ESCALATION_FACTOR,
    UNCERTAIN_CLOSE_DELTA,
    Action,
    ComputeBudget,
    EscalationDecision,
    decide_next_action,
)
from nucleusiq.agents.components.critic import CritiqueResult, Verdict


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _budget(
    *,
    max_retries: int = 3,
    max_output_tokens: int = 2048,
    max_tool_calls: int = 20,
    initial_token_budget: int | None = None,
    total_run_token_budget: int = DEFAULT_RUN_TOKEN_CEILING,
    escalations_used: int = 0,
    cumulative_tokens_spent: int = 0,
) -> ComputeBudget:
    return ComputeBudget(
        max_retries=max_retries,
        max_output_tokens=max_output_tokens,
        max_tool_calls=max_tool_calls,
        initial_token_budget=initial_token_budget
        if initial_token_budget is not None
        else max_output_tokens,
        total_run_token_budget=total_run_token_budget,
        escalations_used=escalations_used,
        cumulative_tokens_spent=cumulative_tokens_spent,
    )


def _critique(verdict: Verdict, score: float, feedback: str = "") -> CritiqueResult:
    return CritiqueResult(verdict=verdict, score=score, feedback=feedback)


# --------------------------------------------------------------------- #
# Framework constants — sanity lock-down
# --------------------------------------------------------------------- #


class TestFrameworkConstants:
    """Lock in the research-harness numbers so accidental tuning is loud."""

    def test_max_escalations_is_positive_and_small(self) -> None:
        assert MAX_ESCALATIONS_PER_RUN >= 1
        assert MAX_ESCALATIONS_PER_RUN <= 5, (
            "Large escalation counts defeat the 'bounded compute' story"
        )

    def test_token_ceiling_reasonable(self) -> None:
        assert HARD_TOKEN_CEILING >= 4_096
        assert HARD_TOKEN_CEILING <= 65_536

    def test_default_run_token_ceiling_covers_realistic_run(self) -> None:
        # At least enough for a tool-heavy autonomous loop (primary agent +
        # Critic + Refiner + 2 escalations on a 128K-context model with
        # large tool results).  Upper bound keeps us from forgetting that
        # this is a *safety* cap, not a blank cheque.
        assert DEFAULT_RUN_TOKEN_CEILING >= 100_000
        assert DEFAULT_RUN_TOKEN_CEILING <= 5_000_000

    def test_escalation_factors_grow_budget(self) -> None:
        assert TOKEN_ESCALATION_FACTOR > 1.0
        assert TOOL_CALL_ESCALATION_FACTOR > 1.0

    def test_uncertain_close_delta_small(self) -> None:
        assert 0.0 < UNCERTAIN_CLOSE_DELTA < 0.5

    def test_improvement_delta_small(self) -> None:
        assert 0.0 < IMPROVEMENT_DELTA < 0.5


# --------------------------------------------------------------------- #
# ComputeBudget basics
# --------------------------------------------------------------------- #


class TestComputeBudgetConstruction:
    def test_requires_positive_fields(self) -> None:
        with pytest.raises(Exception):
            ComputeBudget(
                max_retries=0,
                max_output_tokens=100,
                max_tool_calls=10,
                initial_token_budget=100,
            )
        with pytest.raises(Exception):
            ComputeBudget(
                max_retries=1,
                max_output_tokens=0,
                max_tool_calls=10,
                initial_token_budget=100,
            )

    def test_is_frozen(self) -> None:
        b = _budget()
        with pytest.raises(Exception):
            b.max_retries = 99  # type: ignore[misc]

    def test_from_config_reads_agent_config(self) -> None:
        class FakeCfg:
            llm_max_output_tokens = 1024
            max_retries = 5
            max_tool_calls = 40

            def get_effective_max_tool_calls(self) -> int:
                return self.max_tool_calls

        b = ComputeBudget.from_config(FakeCfg())
        assert b.max_retries == 5
        assert b.max_output_tokens == 1024
        assert b.max_tool_calls == 40
        assert b.initial_token_budget == 1024
        assert b.escalations_used == 0
        assert b.cumulative_tokens_spent == 0

    def test_from_config_falls_back_when_missing_effective_helper(self) -> None:
        class FakeCfg:
            llm_max_output_tokens = 512
            max_retries = 2
            max_tool_calls = 17

        b = ComputeBudget.from_config(FakeCfg())
        assert b.max_tool_calls == 17

    def test_from_config_coerces_nones_to_defaults(self) -> None:
        class FakeCfg:
            llm_max_output_tokens = None
            max_retries = None
            max_tool_calls = None

        b = ComputeBudget.from_config(FakeCfg())
        assert b.max_retries >= 1
        assert b.max_output_tokens > 0
        assert b.max_tool_calls > 0


# --------------------------------------------------------------------- #
# ComputeBudget transitions
# --------------------------------------------------------------------- #


class TestCanEscalate:
    def test_true_when_under_both_caps(self) -> None:
        b = _budget(max_output_tokens=1_000, cumulative_tokens_spent=100)
        assert b.can_escalate() is True

    def test_false_when_max_escalations_reached(self) -> None:
        b = _budget(escalations_used=MAX_ESCALATIONS_PER_RUN)
        assert b.can_escalate() is False

    def test_false_when_cumulative_cap_hit(self) -> None:
        b = _budget(
            max_output_tokens=1_000,
            initial_token_budget=1_000,
            total_run_token_budget=10_000,
            cumulative_tokens_spent=10_001,
        )
        assert b.can_escalate() is False

    def test_cumulative_cap_matches_total_run_token_budget(self) -> None:
        b = _budget(
            max_output_tokens=2_048,
            initial_token_budget=2_048,
            total_run_token_budget=250_000,
        )
        assert b.cumulative_cap() == 250_000

    def test_true_when_tokens_under_total_run_budget(self) -> None:
        # Regression for the pre-fix bug: the old cap was
        # ``initial_token_budget * 4`` which exhausted after ~8K tokens
        # on a default config, killing escalation on any real task.
        # The new cap is a flat ``total_run_token_budget`` so runs that
        # consume far more than a single call's output cap can still
        # escalate.
        b = _budget(
            max_output_tokens=2_048,
            initial_token_budget=2_048,
            total_run_token_budget=DEFAULT_RUN_TOKEN_CEILING,
            cumulative_tokens_spent=150_000,
        )
        assert b.can_escalate() is True


class TestEscalate:
    def test_grows_retries_tokens_and_tool_calls(self) -> None:
        b = _budget(max_retries=3, max_output_tokens=2_000, max_tool_calls=20)
        after = b.escalate("uncertain_close")
        assert after.max_retries == 4
        assert after.max_output_tokens == int(2_000 * TOKEN_ESCALATION_FACTOR)
        assert after.max_tool_calls == int(20 * TOOL_CALL_ESCALATION_FACTOR)
        assert after.escalations_used == 1

    def test_never_exceeds_hard_token_ceiling(self) -> None:
        b = _budget(
            max_output_tokens=HARD_TOKEN_CEILING,
            initial_token_budget=HARD_TOKEN_CEILING,
        )
        after = b.escalate("stuck")
        assert after.max_output_tokens == HARD_TOKEN_CEILING

    def test_never_reduces_tool_calls(self) -> None:
        # Integer truncation of 1 * 1.25 = 1, so floor stays the same.
        b = _budget(max_tool_calls=1)
        after = b.escalate("stuck")
        assert after.max_tool_calls >= 1

    def test_raises_when_cannot_escalate(self) -> None:
        b = _budget(escalations_used=MAX_ESCALATIONS_PER_RUN)
        with pytest.raises(RuntimeError):
            b.escalate("stuck")

    def test_returns_new_instance(self) -> None:
        b = _budget()
        after = b.escalate("stuck")
        assert after is not b
        assert b.escalations_used == 0

    def test_reason_only_affects_telemetry(self) -> None:
        """Both reasons produce arithmetically identical budgets."""
        a = _budget().escalate("uncertain_close")
        b = _budget().escalate("stuck")
        assert a.max_retries == b.max_retries
        assert a.max_output_tokens == b.max_output_tokens
        assert a.max_tool_calls == b.max_tool_calls


class TestRecordTokens:
    def test_positive_tokens_accumulate(self) -> None:
        b = _budget(cumulative_tokens_spent=10)
        after = b.record_tokens(25)
        assert after.cumulative_tokens_spent == 35

    def test_zero_or_negative_is_noop(self) -> None:
        b = _budget(cumulative_tokens_spent=10)
        assert b.record_tokens(0) is b
        assert b.record_tokens(-5) is b

    def test_returns_new_instance_when_nonzero(self) -> None:
        b = _budget(cumulative_tokens_spent=0)
        after = b.record_tokens(1)
        assert after is not b


# --------------------------------------------------------------------- #
# decide_next_action — per-case matrix
# --------------------------------------------------------------------- #


class TestDecisionAccept:
    def test_pass_always_accepts_regardless_of_score(self) -> None:
        c = _critique(Verdict.PASS, 0.0)
        d = decide_next_action(c, [c], _budget(), attempt=0, uncertain_accept_threshold=0.7)
        assert d.action == Action.STOP_ACCEPT
        assert d.escalation_reason is None

    def test_uncertain_above_threshold_accepts(self) -> None:
        c = _critique(Verdict.UNCERTAIN, 0.75)
        d = decide_next_action(c, [c], _budget(), attempt=0, uncertain_accept_threshold=0.7)
        assert d.action == Action.STOP_ACCEPT

    def test_uncertain_exactly_at_threshold_accepts(self) -> None:
        c = _critique(Verdict.UNCERTAIN, 0.7)
        d = decide_next_action(c, [c], _budget(), attempt=0, uncertain_accept_threshold=0.7)
        assert d.action == Action.STOP_ACCEPT


class TestDecisionUncertainClose:
    def test_close_to_threshold_escalates_when_allowed(self) -> None:
        c = _critique(Verdict.UNCERTAIN, 0.65)  # within delta 0.10 of 0.7
        b = _budget(max_retries=3)
        d = decide_next_action(c, [c], b, attempt=0, uncertain_accept_threshold=0.7)
        assert d.action == Action.RETRY
        assert d.escalation_reason == "uncertain_close"
        assert d.budget.escalations_used == 1
        assert d.budget.max_retries == b.max_retries + 1

    def test_close_to_threshold_last_attempt_escalates_not_abstain(self) -> None:
        c = _critique(Verdict.UNCERTAIN, 0.65)
        b = _budget(max_retries=3)
        d = decide_next_action(c, [c], b, attempt=2, uncertain_accept_threshold=0.7)
        assert d.action == Action.RETRY
        assert d.escalation_reason == "uncertain_close"

    def test_close_to_threshold_abstains_when_exhausted_no_escalations_left(
        self,
    ) -> None:
        c = _critique(Verdict.UNCERTAIN, 0.65)
        b = _budget(
            max_retries=3,
            escalations_used=MAX_ESCALATIONS_PER_RUN,  # cannot escalate
        )
        d = decide_next_action(c, [c], b, attempt=2, uncertain_accept_threshold=0.7)
        assert d.action == Action.STOP_ABSTAIN


class TestDecisionFail:
    def test_fail_with_progress_retries_no_escalation(self) -> None:
        prev = _critique(Verdict.FAIL, 0.2)
        cur = _critique(Verdict.FAIL, 0.2 + IMPROVEMENT_DELTA + 0.01)
        d = decide_next_action(
            cur, [prev, cur], _budget(), attempt=1, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.RETRY
        assert d.escalation_reason is None

    def test_fail_stuck_escalates_when_allowed(self) -> None:
        prev = _critique(Verdict.FAIL, 0.3)
        cur = _critique(Verdict.FAIL, 0.3)  # no delta
        b = _budget(max_retries=3)
        d = decide_next_action(
            cur, [prev, cur], b, attempt=1, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.RETRY
        assert d.escalation_reason == "stuck"
        assert d.budget.escalations_used == 1

    def test_fail_stuck_abstains_when_no_escalations_left(self) -> None:
        prev = _critique(Verdict.FAIL, 0.3)
        cur = _critique(Verdict.FAIL, 0.3)
        b = _budget(escalations_used=MAX_ESCALATIONS_PER_RUN)
        d = decide_next_action(
            cur, [prev, cur], b, attempt=1, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.STOP_ABSTAIN

    def test_fail_first_attempt_retries_baseline(self) -> None:
        cur = _critique(Verdict.FAIL, 0.2)
        d = decide_next_action(
            cur, [cur], _budget(), attempt=0, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.RETRY
        assert d.escalation_reason is None

    def test_fail_budget_exhausted_abstains(self) -> None:
        cur = _critique(Verdict.FAIL, 0.2)
        b = _budget(max_retries=3)
        d = decide_next_action(cur, [cur], b, attempt=2, uncertain_accept_threshold=0.7)
        assert d.action == Action.STOP_ABSTAIN


class TestDecisionFailPostEscalationGuard:
    """F3.1 — 'Futile escalation' guard.

    Regression tests for the production bug where the loop escalated a
    second time after the first escalation produced no Critic-score
    improvement.  The observed symptom was:

        Attempt 3 [ESCALATE]: stuck → tokens 12000→16384  (real growth)
        Attempt 4 [CRITIC]:   score unchanged (delta=+0.00)
        Attempt 4 [ESCALATE]: stuck → tokens 16384→16384  (NO growth)
        Attempt 5 [REFINE]:   answer got shorter, same score
        Attempt 5 [ABSTAIN]:  (~10 extra minutes of wasted compute)

    After the fix, ``decide_next_action`` must abstain at Attempt 4
    instead of escalating a second time.
    """

    def test_post_escalation_no_progress_abstains(self) -> None:
        """escalations_used=1 + stuck delta → ABSTAIN, not RETRY."""
        prev = _critique(Verdict.FAIL, 0.22)
        cur = _critique(Verdict.FAIL, 0.22)  # delta = 0.0
        b = _budget(
            max_retries=5,  # nominally 2 retries remaining
            escalations_used=1,  # but we already spent one escalation
        )
        d = decide_next_action(
            cur, [prev, cur], b, attempt=3, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.STOP_ABSTAIN, (
            "A second escalation after no progress is wasted compute; "
            "Aletheia's philosophy says abstain."
        )
        assert d.escalation_reason is None
        assert d.budget.escalations_used == 1  # unchanged

    def test_post_escalation_tiny_progress_still_abstains(self) -> None:
        """delta below IMPROVEMENT_DELTA after escalation = still stuck."""
        prev = _critique(Verdict.FAIL, 0.30)
        cur = _critique(Verdict.FAIL, 0.30 + IMPROVEMENT_DELTA - 0.01)
        b = _budget(max_retries=5, escalations_used=1)
        d = decide_next_action(
            cur, [prev, cur], b, attempt=3, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.STOP_ABSTAIN

    def test_post_escalation_real_progress_retries_without_escalating(
        self,
    ) -> None:
        """delta >= IMPROVEMENT_DELTA after escalation: NOT stuck -> keep going."""
        prev = _critique(Verdict.FAIL, 0.30)
        cur = _critique(Verdict.FAIL, 0.30 + IMPROVEMENT_DELTA + 0.01)
        b = _budget(max_retries=5, escalations_used=1)
        d = decide_next_action(
            cur, [prev, cur], b, attempt=3, uncertain_accept_threshold=0.7
        )
        # Real progress -> regular retry, NOT the stuck branch.
        assert d.action == Action.RETRY
        assert d.escalation_reason is None  # no second escalation needed
        assert d.budget.escalations_used == 1  # unchanged

    def test_first_stuck_still_escalates(self) -> None:
        """Guardrail only fires after the FIRST escalation.
        A first 'stuck' signal must still be able to escalate once."""
        prev = _critique(Verdict.FAIL, 0.30)
        cur = _critique(Verdict.FAIL, 0.30)
        b = _budget(max_retries=3, escalations_used=0)
        d = decide_next_action(
            cur, [prev, cur], b, attempt=1, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.RETRY
        assert d.escalation_reason == "stuck"
        assert d.budget.escalations_used == 1

    def test_post_escalation_guard_triggers_even_on_final_attempt(self) -> None:
        """If we somehow reached the final attempt with escalations_used=1
        and delta=0, the guard should still fire cleanly (no RuntimeError
        from trying to escalate into an exhausted budget)."""
        prev = _critique(Verdict.FAIL, 0.22)
        cur = _critique(Verdict.FAIL, 0.22)
        b = _budget(max_retries=4, escalations_used=1)
        d = decide_next_action(
            cur, [prev, cur], b, attempt=3, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.STOP_ABSTAIN
        assert d.budget is b  # no mutation


class TestDecisionUncertainFar:
    """UNCERTAIN verdicts well below threshold behave like FAIL for control
    flow but do not trigger the 'uncertain_close' escalation reason."""

    def test_retries_when_budget_left(self) -> None:
        cur = _critique(Verdict.UNCERTAIN, 0.3)  # far below threshold
        b = _budget(max_retries=3)
        d = decide_next_action(
            cur, [cur], b, attempt=0, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.RETRY
        assert d.escalation_reason is None

    def test_abstains_when_budget_exhausted(self) -> None:
        cur = _critique(Verdict.UNCERTAIN, 0.3)
        b = _budget(max_retries=3)
        d = decide_next_action(
            cur, [cur], b, attempt=2, uncertain_accept_threshold=0.7
        )
        assert d.action == Action.STOP_ABSTAIN


# --------------------------------------------------------------------- #
# EscalationDecision model
# --------------------------------------------------------------------- #


class TestEscalationDecisionModel:
    def test_is_frozen(self) -> None:
        b = _budget()
        d = EscalationDecision(action=Action.RETRY, budget=b)
        with pytest.raises(Exception):
            d.action = Action.STOP_ACCEPT  # type: ignore[misc]

    def test_escalation_reason_optional(self) -> None:
        b = _budget()
        d = EscalationDecision(action=Action.RETRY, budget=b)
        assert d.escalation_reason is None
