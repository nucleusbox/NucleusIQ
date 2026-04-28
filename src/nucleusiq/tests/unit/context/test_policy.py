"""Tests for Context Mgmt v2 — Step 2 (policy classifier).

The :class:`PolicyClassifier` decides how each tool result is treated
when the context comes under pressure: ``EVIDENCE`` results are
offloaded into the :class:`ContentStore`, ``EPHEMERAL`` results may be
dropped silently, ``AUTO`` results are run through the same heuristics
as undeclared tool results.

Test groups
-----------
1. TestDeclaredPolicy — explicit declarations win unconditionally
2. TestNamePatterns   — name-pattern hits drive the AUTO decision
3. TestSizeHeuristic  — small results → EPHEMERAL, large → EVIDENCE
4. TestResolvedPolicy — confidence + source bookkeeping is correct
5. TestEdgeCases      — empty / whitespace / None inputs are stable
"""

from __future__ import annotations

import pytest
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.policy import (
    ContextPolicy,
    PolicyClassifier,
    ResolvedPolicy,
)


def _make_classifier(
    *,
    evidence_patterns: tuple[str, ...] | None = None,
    ephemeral_patterns: tuple[str, ...] | None = None,
    ephemeral_size_threshold: int = 500,
) -> PolicyClassifier:
    """Build a classifier with overridable heuristic knobs.

    Test isolation: every test creates its own classifier so default
    config drift on the project-wide ``ContextConfig`` cannot
    silently break the assertions.
    """
    cfg = ContextConfig(
        evidence_name_patterns=evidence_patterns
        if evidence_patterns is not None
        else ("read_pdf", "search_web"),
        ephemeral_name_patterns=ephemeral_patterns
        if ephemeral_patterns is not None
        else ("get_time", "ping"),
        ephemeral_size_threshold=ephemeral_size_threshold,
    )
    return PolicyClassifier(cfg)


# ---------------------------------------------------------------------- #
# 1. Declared policy short-circuits the heuristic                         #
# ---------------------------------------------------------------------- #


class TestDeclaredPolicy:
    """``declared_policy`` always wins — no heuristic is consulted."""

    def test_declared_evidence_overrides_size_heuristic(self):
        """A 5-token EVIDENCE declaration is honoured even though size says EPHEMERAL."""
        c = _make_classifier(ephemeral_size_threshold=10_000)
        r = c.classify(
            tool_name="ping",
            content_tokens=5,
            declared_policy=ContextPolicy.EVIDENCE,
        )
        assert r.policy is ContextPolicy.EVIDENCE
        assert r.source == "tool_decoration"

    def test_declared_ephemeral_overrides_name_pattern(self):
        c = _make_classifier()
        r = c.classify(
            tool_name="search_web",
            content_tokens=5000,
            declared_policy=ContextPolicy.EPHEMERAL,
        )
        assert r.policy is ContextPolicy.EPHEMERAL
        assert r.source == "tool_decoration"

    def test_auto_does_not_short_circuit(self):
        """``AUTO`` is not a "declared" decision — heuristic still runs."""
        c = _make_classifier()
        r = c.classify(
            tool_name="search_web",
            content_tokens=5000,
            declared_policy=ContextPolicy.AUTO,
        )
        assert r.source != "tool_decoration"

    def test_none_declared_policy_treated_as_auto(self):
        """Passing ``declared_policy=None`` mirrors the AUTO path."""
        c = _make_classifier()
        r = c.classify(
            tool_name="search_web", content_tokens=5000, declared_policy=None
        )
        assert r.source != "tool_decoration"


# ---------------------------------------------------------------------- #
# 2. Name-pattern hits drive AUTO decisions                               #
# ---------------------------------------------------------------------- #


class TestNamePatterns:
    def test_evidence_pattern_match(self):
        """Tool whose name contains an evidence fragment classifies as EVIDENCE."""
        c = _make_classifier(
            evidence_patterns=("read_pdf",),
            ephemeral_size_threshold=10_000,  # would say EPHEMERAL by size alone
        )
        r = c.classify(tool_name="read_pdf_page", content_tokens=10)
        assert r.policy is ContextPolicy.EVIDENCE
        assert r.source == "name_pattern"

    def test_ephemeral_pattern_match(self):
        c = _make_classifier(
            ephemeral_patterns=("get_time",),
            ephemeral_size_threshold=1,  # would say EVIDENCE by size alone
        )
        r = c.classify(tool_name="get_time_iso", content_tokens=1000)
        assert r.policy is ContextPolicy.EPHEMERAL
        assert r.source == "name_pattern"

    def test_evidence_pattern_takes_priority_over_ephemeral(self):
        """Same tool matching both lists → EVIDENCE wins (preserve more info)."""
        c = _make_classifier(
            evidence_patterns=("search",),
            ephemeral_patterns=("search",),
        )
        r = c.classify(tool_name="search_web", content_tokens=10)
        assert r.policy is ContextPolicy.EVIDENCE

    def test_pattern_match_is_case_insensitive(self):
        c = _make_classifier(evidence_patterns=("read_pdf",))
        r = c.classify(tool_name="Read_PDF_Tool", content_tokens=10)
        assert r.policy is ContextPolicy.EVIDENCE


# ---------------------------------------------------------------------- #
# 3. Size heuristic when no name pattern matches                          #
# ---------------------------------------------------------------------- #


class TestSizeHeuristic:
    def test_small_result_is_ephemeral(self):
        """Below threshold + no pattern hit → EPHEMERAL."""
        c = _make_classifier(
            evidence_patterns=(),
            ephemeral_patterns=(),
            ephemeral_size_threshold=500,
        )
        r = c.classify(tool_name="unknown_tool", content_tokens=100)
        assert r.policy is ContextPolicy.EPHEMERAL
        assert r.source == "size"

    def test_large_result_falls_through_to_default(self):
        """At/above threshold + no pattern → conservative EVIDENCE default."""
        c = _make_classifier(
            evidence_patterns=(),
            ephemeral_patterns=(),
            ephemeral_size_threshold=500,
        )
        r = c.classify(tool_name="unknown_tool", content_tokens=4000)
        assert r.policy is ContextPolicy.EVIDENCE
        assert r.source == "default"

    def test_threshold_strict_less_than(self):
        """Tokens *equal to* threshold are NOT below it → EVIDENCE default."""
        c = _make_classifier(
            evidence_patterns=(),
            ephemeral_patterns=(),
            ephemeral_size_threshold=100,
        )
        r = c.classify(tool_name="x", content_tokens=100)
        assert r.policy is ContextPolicy.EVIDENCE


# ---------------------------------------------------------------------- #
# 4. ResolvedPolicy bookkeeping                                           #
# ---------------------------------------------------------------------- #


class TestResolvedPolicy:
    def test_resolved_policy_is_frozen_value(self):
        c = _make_classifier()
        r = c.classify(
            tool_name="search_web",
            content_tokens=5000,
            declared_policy=ContextPolicy.EVIDENCE,
        )
        assert isinstance(r, ResolvedPolicy)

    def test_declared_has_max_confidence(self):
        c = _make_classifier()
        r = c.classify(
            tool_name="x",
            content_tokens=10,
            declared_policy=ContextPolicy.EVIDENCE,
        )
        assert r.confidence == pytest.approx(1.0)

    def test_heuristic_confidence_below_one(self):
        """Heuristic decisions are intentionally less confident than declarations."""
        c = _make_classifier()
        r = c.classify(tool_name="search_web", content_tokens=10)
        assert r.confidence < 1.0

    def test_default_confidence_is_lowest(self):
        """The conservative ``default`` branch has the weakest confidence."""
        c = _make_classifier(evidence_patterns=(), ephemeral_patterns=())
        r = c.classify(tool_name="unfamiliar", content_tokens=4000)
        # Default branch confidence is intentionally low (~0.5) so
        # operators can spot it in telemetry and tune thresholds.
        assert r.source == "default"
        assert r.confidence <= 0.5

    def test_resolved_policy_rejects_auto(self):
        """``ResolvedPolicy(policy=AUTO)`` is a programming error."""
        with pytest.raises(ValueError):
            ResolvedPolicy(policy=ContextPolicy.AUTO, source="x", confidence=1.0)

    def test_resolved_policy_rejects_out_of_range_confidence(self):
        with pytest.raises(ValueError):
            ResolvedPolicy(
                policy=ContextPolicy.EVIDENCE,
                source="x",
                confidence=1.5,
            )


# ---------------------------------------------------------------------- #
# 5. Edge cases                                                           #
# ---------------------------------------------------------------------- #


class TestEdgeCases:
    def test_zero_tokens_is_ephemeral_below_threshold(self):
        """Empty / zero-token content is the cheapest → EPHEMERAL."""
        c = _make_classifier(evidence_patterns=(), ephemeral_patterns=())
        r = c.classify(tool_name="x", content_tokens=0)
        assert r.policy is ContextPolicy.EPHEMERAL

    def test_empty_tool_name_falls_back_to_size(self):
        """Defensive: empty name should not raise; size dominates."""
        c = _make_classifier(
            evidence_patterns=(),
            ephemeral_patterns=(),
            ephemeral_size_threshold=500,
        )
        r = c.classify(tool_name="", content_tokens=4000)
        assert r.policy is ContextPolicy.EVIDENCE
        assert r.source == "default"
