"""Unit tests for F5 (best-candidate tracking) and F6 (accept-with-warning).

These tests exercise the pure helpers on ``SimpleRunner`` without spinning
up a full autonomous run.  Integration of the helpers into the retry loop
is covered by the broader autonomous-mode suite.
"""

from __future__ import annotations

import pytest

from nucleusiq.agents.components.critic import CritiqueResult, Verdict
from nucleusiq.agents.modes.autonomous.simple_runner import (
    _ACCEPT_WITH_WARNING_FLOOR,
    _ACCEPT_WITH_WARNING_IMPROVEMENT,
    SimpleRunner,
)


def _c(verdict: Verdict, score: float, feedback: str = "") -> CritiqueResult:
    return CritiqueResult(verdict=verdict, score=score, feedback=feedback)


# --------------------------------------------------------------------- #
# F5 — _is_better_critique
# --------------------------------------------------------------------- #


class TestIsBetterCritiqueBaseline:
    def test_none_current_best_always_wins(self) -> None:
        new = _c(Verdict.FAIL, 0.0)
        assert SimpleRunner._is_better_critique(new, None) is True


class TestIsBetterCritiquePassDominance:
    """PASS should always beat non-PASS, regardless of score."""

    def test_pass_beats_uncertain_at_higher_score(self) -> None:
        new = _c(Verdict.PASS, 0.72)
        cur = _c(Verdict.UNCERTAIN, 0.99)
        assert SimpleRunner._is_better_critique(new, cur) is True

    def test_pass_beats_fail(self) -> None:
        new = _c(Verdict.PASS, 0.7)
        cur = _c(Verdict.FAIL, 0.99)
        assert SimpleRunner._is_better_critique(new, cur) is True

    def test_uncertain_never_beats_pass(self) -> None:
        new = _c(Verdict.UNCERTAIN, 0.99)
        cur = _c(Verdict.PASS, 0.7)
        assert SimpleRunner._is_better_critique(new, cur) is False

    def test_fail_never_beats_pass(self) -> None:
        new = _c(Verdict.FAIL, 0.99)
        cur = _c(Verdict.PASS, 0.7)
        assert SimpleRunner._is_better_critique(new, cur) is False


class TestIsBetterCritiqueUncertainOverFail:
    """UNCERTAIN beats FAIL when score is equal-or-higher."""

    def test_uncertain_beats_fail_at_equal_score(self) -> None:
        new = _c(Verdict.UNCERTAIN, 0.4)
        cur = _c(Verdict.FAIL, 0.4)
        assert SimpleRunner._is_better_critique(new, cur) is True

    def test_uncertain_beats_fail_at_higher_score(self) -> None:
        new = _c(Verdict.UNCERTAIN, 0.5)
        cur = _c(Verdict.FAIL, 0.3)
        assert SimpleRunner._is_better_critique(new, cur) is True

    def test_fail_with_higher_score_does_not_beat_uncertain(self) -> None:
        # Even a FAIL at higher raw score should not dethrone an
        # UNCERTAIN at equal-or-higher score.
        new = _c(Verdict.FAIL, 0.45)
        cur = _c(Verdict.UNCERTAIN, 0.45)
        assert SimpleRunner._is_better_critique(new, cur) is False


class TestIsBetterCritiqueScoreTiebreaker:
    def test_higher_score_wins_at_same_verdict(self) -> None:
        new = _c(Verdict.FAIL, 0.5)
        cur = _c(Verdict.FAIL, 0.3)
        assert SimpleRunner._is_better_critique(new, cur) is True

    def test_equal_score_same_verdict_does_not_swap(self) -> None:
        # Strictly ">" on score keeps the first best_critique stable so
        # tied attempts don't keep overwriting each other.
        new = _c(Verdict.FAIL, 0.3)
        cur = _c(Verdict.FAIL, 0.3)
        assert SimpleRunner._is_better_critique(new, cur) is False


# --------------------------------------------------------------------- #
# F6 — _should_accept_with_warning
# --------------------------------------------------------------------- #


class TestAcceptWithWarningPolicy:
    def test_fail_best_verdict_never_accepts(self) -> None:
        # FAIL means Critic cited concrete errors; do not paper over.
        best = _c(Verdict.FAIL, 0.65)
        first = _c(Verdict.FAIL, 0.2)
        assert SimpleRunner._should_accept_with_warning(best, first) is False

    def test_uncertain_below_floor_never_accepts(self) -> None:
        best = _c(Verdict.UNCERTAIN, _ACCEPT_WITH_WARNING_FLOOR - 0.01)
        first = _c(Verdict.FAIL, 0.1)
        assert SimpleRunner._should_accept_with_warning(best, first) is False

    def test_uncertain_above_floor_without_improvement_rejects(self) -> None:
        best = _c(Verdict.UNCERTAIN, 0.6)
        first = _c(Verdict.UNCERTAIN, 0.58)  # delta < required improvement
        assert SimpleRunner._should_accept_with_warning(best, first) is False

    def test_uncertain_above_floor_with_improvement_accepts(self) -> None:
        # Floor met AND improvement >= threshold → accept with warning.
        best = _c(Verdict.UNCERTAIN, 0.65)
        first = _c(Verdict.FAIL, 0.2)
        assert (
            best.score - first.score >= _ACCEPT_WITH_WARNING_IMPROVEMENT
        )
        assert SimpleRunner._should_accept_with_warning(best, first) is True

    def test_regression_below_first_rejects(self) -> None:
        # Refiner made it worse, not better.
        best = _c(Verdict.UNCERTAIN, 0.6)
        first = _c(Verdict.UNCERTAIN, 0.7)
        assert SimpleRunner._should_accept_with_warning(best, first) is False

    def test_exact_floor_with_exact_improvement_accepts(self) -> None:
        # Add a tiny safety margin to avoid fighting float subtraction.
        best = _c(Verdict.UNCERTAIN, _ACCEPT_WITH_WARNING_FLOOR)
        first_score = (
            _ACCEPT_WITH_WARNING_FLOOR - _ACCEPT_WITH_WARNING_IMPROVEMENT - 1e-9
        )
        first = _c(Verdict.FAIL, max(0.0, first_score))
        assert SimpleRunner._should_accept_with_warning(best, first) is True


class TestBuildWarningMessage:
    def test_warning_contains_scores_and_attempt_count(self) -> None:
        best = _c(Verdict.UNCERTAIN, 0.65, feedback="needs more detail")
        first = _c(Verdict.FAIL, 0.3)
        msg = SimpleRunner._build_warning(best, first, attempts=3)
        assert "score=0.65" in msg
        assert "uncertain" in msg.lower()
        assert "3 attempt" in msg
        # Improvement should render as a signed delta for clarity.
        assert "+0.35" in msg


# --------------------------------------------------------------------- #
# Framework constants — sanity lockdown.
# --------------------------------------------------------------------- #


class TestAcceptWithWarningConstants:
    def test_floor_within_uncertain_band(self) -> None:
        # UNCERTAIN band per the Critic prompt is 0.4–0.69 — the F6
        # floor should live in the upper half so we only soften on
        # "close to PASS" cases.
        assert 0.4 < _ACCEPT_WITH_WARNING_FLOOR <= 0.7

    def test_improvement_threshold_is_meaningful(self) -> None:
        # Must be bigger than typical Critic-score noise (~0.02) so we
        # don't soften on sampling jitter alone.
        assert _ACCEPT_WITH_WARNING_IMPROVEMENT >= 0.05
        assert _ACCEPT_WITH_WARNING_IMPROVEMENT <= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
