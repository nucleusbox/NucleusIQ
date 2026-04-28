"""Tests for Context Mgmt v2 — Step 2 (engine wiring).

These tests cover the integration seams between the new components
and :class:`ContextEngine`:

* :meth:`ContextEngine.ingest_tool_result` runs the
  :class:`PolicyClassifier` on every result, persists the
  :class:`ResolvedPolicy` keyed by ``tool_call_id``, and bumps
  telemetry counters.
* :attr:`ContextEngine.recall_tracker` advances its turn counter
  exactly once per :meth:`ContextEngine.post_response` call.
* :attr:`ContextEngine.telemetry` surfaces ``recall_count``,
  ``recall_tokens``, ``policy_breakdown`` and
  ``policy_source_breakdown`` — the four observability signals the
  v2 redesign requires (§9).
"""

from __future__ import annotations

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.policy import ContextPolicy


def _engine(**overrides) -> ContextEngine:
    """Build an engine with sensible Step-2 defaults."""
    cfg = ContextConfig(
        max_context_tokens=overrides.pop("max_context_tokens", 100_000),
        tool_result_threshold=overrides.pop("tool_result_threshold", 100),
        evidence_name_patterns=overrides.pop(
            "evidence_name_patterns", ("read_pdf", "search")
        ),
        ephemeral_name_patterns=overrides.pop(
            "ephemeral_name_patterns", ("get_time", "ping")
        ),
        ephemeral_size_threshold=overrides.pop("ephemeral_size_threshold", 50),
        **overrides,
    )
    return ContextEngine(cfg)


# ---------------------------------------------------------------------- #
# 1. Classifier wired into ingest_tool_result                              #
# ---------------------------------------------------------------------- #


class TestIngestRunsClassifier:
    def test_declared_evidence_is_persisted(self):
        engine = _engine()
        engine.ingest_tool_result(
            "small content",
            "ping",
            tool_call_id="call-1",
            declared_policy=ContextPolicy.EVIDENCE,
        )
        resolved = engine.get_policy_for("call-1")
        assert resolved is not None
        assert resolved.policy is ContextPolicy.EVIDENCE
        assert resolved.source == "tool_decoration"

    def test_declared_ephemeral_is_persisted(self):
        engine = _engine()
        engine.ingest_tool_result(
            "x" * 5_000,
            "search_web",
            tool_call_id="call-2",
            declared_policy=ContextPolicy.EPHEMERAL,
        )
        resolved = engine.get_policy_for("call-2")
        assert resolved is not None
        assert resolved.policy is ContextPolicy.EPHEMERAL
        assert resolved.source == "tool_decoration"

    def test_auto_falls_through_to_heuristic(self):
        """AUTO is not terminal — heuristic must finish the decision."""
        engine = _engine()
        engine.ingest_tool_result(
            "small ephemeral fact",
            "get_time_iso",
            tool_call_id="call-3",
            declared_policy=ContextPolicy.AUTO,
        )
        resolved = engine.get_policy_for("call-3")
        assert resolved is not None
        assert resolved.policy is not ContextPolicy.AUTO
        assert resolved.source != "tool_decoration"

    def test_undeclared_uses_name_pattern(self):
        engine = _engine()
        engine.ingest_tool_result("x" * 4_000, "read_pdf_page", tool_call_id="call-4")
        resolved = engine.get_policy_for("call-4")
        assert resolved is not None
        assert resolved.policy is ContextPolicy.EVIDENCE
        assert resolved.source == "name_pattern"

    def test_undeclared_falls_through_to_size_for_small_unknown_tool(self):
        engine = _engine(ephemeral_size_threshold=500)
        engine.ingest_tool_result("tiny", "totally_unknown_tool", tool_call_id="call-5")
        resolved = engine.get_policy_for("call-5")
        assert resolved is not None
        assert resolved.policy is ContextPolicy.EPHEMERAL
        assert resolved.source == "size"

    def test_undeclared_large_unknown_tool_defaults_to_evidence(self):
        engine = _engine(ephemeral_size_threshold=10)
        engine.ingest_tool_result(
            "x" * 4_000, "totally_unknown_tool", tool_call_id="call-6"
        )
        resolved = engine.get_policy_for("call-6")
        assert resolved is not None
        assert resolved.policy is ContextPolicy.EVIDENCE
        assert resolved.source == "default"


class TestIngestKeyingFallback:
    """When no ``tool_call_id`` is supplied (legacy callers), the
    engine still classifies — but the resolved policy is keyed under
    a generated id and not directly addressable by the caller.
    Telemetry still increments.
    """

    def test_missing_tool_call_id_does_not_raise(self):
        engine = _engine()
        engine.ingest_tool_result("x" * 5_000, "read_pdf_page", tool_call_id=None)
        # Telemetry must still record the classification.
        assert (
            engine.telemetry.policy_breakdown.get(ContextPolicy.EVIDENCE.value, 0) == 1
        )

    def test_each_call_gets_independent_key(self):
        engine = _engine()
        engine.ingest_tool_result("a" * 5_000, "read_pdf_page")
        engine.ingest_tool_result("b" * 5_000, "read_pdf_page")
        assert (
            engine.telemetry.policy_breakdown.get(ContextPolicy.EVIDENCE.value, 0) == 2
        )


# ---------------------------------------------------------------------- #
# 2. ingest_tool_result returns content unchanged                          #
# ---------------------------------------------------------------------- #


class TestIngestReturnsContentUnchanged:
    """v2 design: classify is a side effect — content is *always*
    returned verbatim so the LLM sees real data on the first pass."""

    def test_small_content_round_trips(self):
        engine = _engine()
        out = engine.ingest_tool_result("hello world", "search_web", tool_call_id="x")
        assert out == "hello world"

    def test_large_content_round_trips(self):
        engine = _engine()
        big = "y" * 10_000
        out = engine.ingest_tool_result(big, "read_pdf_page", tool_call_id="x")
        assert out == big

    def test_strategy_none_skips_classification(self):
        """strategy='none' → no classifier work, no telemetry."""
        cfg = ContextConfig(strategy="none", max_context_tokens=100_000)
        engine = ContextEngine(cfg)

        out = engine.ingest_tool_result(
            "anything",
            "read_pdf_page",
            tool_call_id="x",
            declared_policy=ContextPolicy.EVIDENCE,
        )
        assert out == "anything"
        # Telemetry is empty because the classifier never ran.
        assert engine.telemetry.policy_breakdown == {}
        assert engine.get_policy_for("x") is None


# ---------------------------------------------------------------------- #
# 3. RecallTracker advances on post_response                              #
# ---------------------------------------------------------------------- #


class TestRecallTrackerTurnAdvance:
    def test_post_response_bumps_turn_counter(self):
        engine = _engine()
        before = engine.recall_tracker.turn

        engine.post_response(
            [
                ChatMessage(role="user", content="q"),
                ChatMessage(role="assistant", content="a"),
            ]
        )

        assert engine.recall_tracker.turn == before + 1

    def test_repeated_post_response_calls_advance_repeatedly(self):
        engine = _engine()
        msgs = [
            ChatMessage(role="user", content="q"),
            ChatMessage(role="assistant", content="a"),
        ]
        for _ in range(5):
            engine.post_response(msgs)
        assert engine.recall_tracker.turn == 5

    def test_recorded_recall_visible_in_hot_set_with_lookback(self):
        engine = _engine()
        # Simulate the recall path: tracker.record_recall + a turn boundary.
        engine.recall_tracker.record_recall("ref-x", tokens=10)
        engine.post_response(
            [
                ChatMessage(role="user", content="q"),
                ChatMessage(role="assistant", content="a"),
            ]
        )
        # One turn after recall; lookback=2 should still see it.
        assert "ref-x" in engine.recall_tracker.hot_set(lookback_turns=2)
        # Lookback=0 (only current turn) should NOT — recall was last turn.
        assert "ref-x" not in engine.recall_tracker.hot_set(lookback_turns=0)


# ---------------------------------------------------------------------- #
# 4. Telemetry surface                                                     #
# ---------------------------------------------------------------------- #


class TestTelemetrySurface:
    def test_policy_breakdown_separates_evidence_and_ephemeral(self):
        engine = _engine()
        engine.ingest_tool_result(
            "x" * 5_000,
            "read_pdf_page",
            tool_call_id="a",
            declared_policy=ContextPolicy.EVIDENCE,
        )
        engine.ingest_tool_result(
            "tiny",
            "ping",
            tool_call_id="b",
            declared_policy=ContextPolicy.EPHEMERAL,
        )
        engine.ingest_tool_result(
            "x" * 5_000, "read_pdf_page", tool_call_id="c"
        )  # name pattern → EVIDENCE

        bd = engine.telemetry.policy_breakdown
        assert bd.get(ContextPolicy.EVIDENCE.value) == 2
        assert bd.get(ContextPolicy.EPHEMERAL.value) == 1

    def test_policy_source_breakdown_records_provenance(self):
        engine = _engine()
        # Two declarations, one heuristic.
        engine.ingest_tool_result(
            "x" * 1_000,
            "search_web",
            tool_call_id="a",
            declared_policy=ContextPolicy.EVIDENCE,
        )
        engine.ingest_tool_result(
            "x" * 1_000,
            "ping",
            tool_call_id="b",
            declared_policy=ContextPolicy.EPHEMERAL,
        )
        engine.ingest_tool_result("x" * 5_000, "read_pdf_page", tool_call_id="c")

        sb = engine.telemetry.policy_source_breakdown
        assert sb.get("tool_decoration") == 2
        assert sb.get("name_pattern") == 1

    def test_telemetry_recall_counters_pull_from_tracker(self):
        engine = _engine()
        engine.recall_tracker.record_recall("ref-1", tokens=42)
        engine.recall_tracker.record_recall("ref-2", tokens=100)

        t = engine.telemetry
        assert t.recall_count == 2
        assert t.recall_tokens == 142

    def test_telemetry_breakdowns_are_snapshots(self):
        """The dicts on telemetry are copies — mutating them must not
        leak back into the engine's internal state."""
        engine = _engine()
        engine.ingest_tool_result("x" * 5_000, "read_pdf_page", tool_call_id="a")

        snap = engine.telemetry.policy_breakdown
        snap[ContextPolicy.EVIDENCE.value] = 999  # mutate caller-side

        again = engine.telemetry.policy_breakdown
        assert again[ContextPolicy.EVIDENCE.value] == 1, (
            "Telemetry must return defensive copies"
        )
