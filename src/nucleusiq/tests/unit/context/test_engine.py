"""Tests for ContextEngine facade."""

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.engine import ContextEngine


def _make_engine(
    max_tokens: int = 10_000,
    reserve: int = 1000,
    strategy: str = "progressive",
    threshold: int = 100,
    **kw,
) -> ContextEngine:
    config = ContextConfig(
        max_context_tokens=max_tokens,
        response_reserve=reserve,
        strategy=strategy,
        tool_result_threshold=threshold,
        **kw,
    )
    return ContextEngine(config, DefaultTokenCounter(), max_tokens=max_tokens)


class TestContextEnginePrepare:
    @pytest.mark.asyncio
    async def test_prepare_passthrough_low_utilization(self):
        engine = _make_engine(max_tokens=100_000)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hello"),
        ]
        result = engine.budget
        assert result.utilization == 0.0

        prepared = await engine.prepare(msgs)
        assert len(prepared) == 2

    @pytest.mark.asyncio
    async def test_prepare_triggers_compaction_on_high_util(self):
        engine = _make_engine(max_tokens=500, reserve=50, threshold=50)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="tool", name="search", content="x" * 2000),
            ChatMessage(role="user", content="question"),
        ]
        prepared = await engine.prepare(msgs)
        tel = engine.telemetry
        assert tel.compaction_count > 0 or tel.peak_utilization > 0

    @pytest.mark.asyncio
    async def test_prepare_no_op_when_strategy_none(self):
        engine = _make_engine(strategy="none")
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="tool", name="search", content="x" * 5000),
        ]
        prepared = await engine.prepare(msgs)
        assert len(prepared) == 2
        assert engine.telemetry.compaction_count == 0


class TestContextEngineIngest:
    def test_ingest_small_unchanged(self):
        engine = _make_engine(threshold=1000)
        result = engine.ingest_tool_result("small result", "calc")
        assert result == "small result"

    def test_ingest_large_stored_but_full_content_returned(self):
        engine = _make_engine(threshold=10)
        large = "x" * 500
        result = engine.ingest_tool_result(large, "search")
        assert result == large
        assert engine.store.size == 1

    def test_ingest_no_op_when_strategy_none(self):
        engine = _make_engine(strategy="none", threshold=10)
        large = "x" * 500
        result = engine.ingest_tool_result(large, "search")
        assert result == large


class TestContextEngineCheckpoint:
    def test_checkpoint_records(self):
        engine = _make_engine()
        engine.checkpoint("phase1")
        engine.checkpoint("phase2")
        assert engine.budget is not None


class TestContextEngineTelemetry:
    @pytest.mark.asyncio
    async def test_telemetry_structure(self):
        engine = _make_engine()
        msgs = [ChatMessage(role="user", content="hello")]
        await engine.prepare(msgs)
        tel = engine.telemetry
        assert tel.context_limit > 0
        assert tel.response_reserve > 0
        assert isinstance(tel.region_breakdown, dict)
        assert isinstance(tel.compaction_events, tuple)

    def test_telemetry_after_offload(self):
        engine = _make_engine(threshold=5)
        engine.ingest_tool_result("x" * 200, "tool1")
        tel = engine.telemetry
        assert tel.artifacts_offloaded == 1


class TestContextEngineConfig:
    def test_mode_aware_defaults(self):
        auto_cfg = ContextConfig.for_mode("autonomous")
        std_cfg = ContextConfig.for_mode("standard")
        direct_cfg = ContextConfig.for_mode("direct")
        assert auto_cfg.tool_compaction_trigger < std_cfg.tool_compaction_trigger
        assert direct_cfg.tool_compaction_trigger > std_cfg.tool_compaction_trigger

    def test_summary_schema_defined(self):
        from nucleusiq.agents.context.config import SummarySchema

        schema = SummarySchema(
            goals=("deliver v0.7.6",), decisions=("use progressive",)
        )
        assert len(schema.goals) == 1


class TestAdaptiveOptimalBudget:
    """v0.7.9 — engine uses ``resolve_optimal_budget`` at construction time.

    Root cause regression tests: before v0.7.9 the engine hard-coded
    ``min(config.optimal_budget, resolved_max)`` using a 50K default.
    On a 128K model this made EmergencyCompactor fire at ~45K —
    ~35% of real capacity — and nuked context catastrophically for
    weaker models on tool-heavy tasks.
    """

    def test_engine_auto_resolves_for_128k_model(self):
        """Zero-config against a 128K model yields 89_600 token budget."""
        config = ContextConfig(response_reserve=1_000)
        engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=128_000)
        # The ledger's ``max_tokens`` *is* the resolved optimal budget
        # (it's what utilization is computed against).
        assert engine.budget.max_tokens == 89_600

    def test_engine_auto_resolves_for_small_window(self):
        """8K Llama: resolved budget stays proportionally small."""
        config = ContextConfig(response_reserve=1_000)
        engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=8_000)
        assert engine.budget.max_tokens == 5_600

    def test_engine_caps_at_quality_plateau(self):
        """2M Gemini: capped at the 120K quality plateau."""
        config = ContextConfig(response_reserve=1_000)
        engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=2_000_000)
        assert engine.budget.max_tokens == 120_000

    def test_engine_respects_explicit_override(self):
        """User-set ``optimal_budget`` wins over auto-resolution."""
        config = ContextConfig(
            optimal_budget=30_000,
            response_reserve=1_000,
        )
        engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=128_000)
        assert engine.budget.max_tokens == 30_000

    def test_telemetry_reports_resolved_budget_not_config(self):
        """Telemetry must report the effective budget, not the raw config.

        This is critical for experiment reproducibility — research
        notebooks were mis-reporting ``optimal_budget=50000`` even
        though the ledger was using a different number.
        """
        config = ContextConfig(response_reserve=1_000)
        engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=128_000)
        tel = engine.telemetry
        # 128K × 0.7 = 89_600 — the actual budget used by the ledger.
        assert tel.optimal_budget == 89_600
