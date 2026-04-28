"""Tests for ContextConfig and ObservabilityConfig."""

import pytest
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.config.observability_config import ObservabilityConfig
from nucleusiq.agents.context.config import ContextConfig


class TestContextConfig:
    def test_defaults(self):
        cfg = ContextConfig()
        # v0.7.9: optimal_budget is None (auto-resolve at engine time)
        assert cfg.optimal_budget is None
        assert cfg.optimal_budget_fraction == 0.70
        assert cfg.optimal_budget_ceiling == 120_000
        assert cfg.max_context_tokens is None
        assert cfg.response_reserve == 8192
        assert cfg.strategy == "progressive"
        assert cfg.tool_compaction_trigger == 0.60
        assert cfg.compaction_trigger == 0.75
        assert cfg.emergency_trigger == 0.90
        assert cfg.enable_offloading is True
        assert cfg.enable_observation_masking is True
        assert cfg.enable_summarization is False
        assert cfg.cost_per_million_input is None

    def test_frozen(self):
        cfg = ContextConfig()
        with pytest.raises(Exception):
            cfg.strategy = "none"  # type: ignore[misc]

    def test_for_mode_standard(self):
        cfg = ContextConfig.for_mode("standard")
        assert cfg.compaction_trigger == 0.75
        # v0.7.9: no hard-coded optimal_budget any more
        assert cfg.optimal_budget is None

    def test_for_mode_autonomous(self):
        cfg = ContextConfig.for_mode("autonomous")
        assert cfg.tool_compaction_trigger == 0.55
        assert cfg.preserve_recent_turns == 6
        # v0.7.9: no hard-coded optimal_budget any more
        assert cfg.optimal_budget is None

    def test_for_mode_direct(self):
        cfg = ContextConfig.for_mode("direct")
        assert cfg.tool_compaction_trigger == 0.80
        assert cfg.preserve_recent_turns == 2
        assert cfg.optimal_budget is None


class TestResolveOptimalBudget:
    """v0.7.9 — optimal_budget adapts to the model's real context window.

    The old fixed 50K default caused compaction to fire at ~35% of
    real capacity on a 128K model.  These tests pin down the new
    adaptive behaviour across the full provider spectrum.
    """

    def test_explicit_override_wins(self):
        """User-supplied ``optimal_budget`` always wins — respect intent."""
        cfg = ContextConfig(optimal_budget=30_000)
        # Even with a 2M context window, explicit 30K is honoured.
        assert ContextConfig.resolve_optimal_budget(cfg, 2_000_000) == 30_000
        # And with a tiny window too.
        assert ContextConfig.resolve_optimal_budget(cfg, 8_000) == 30_000

    def test_auto_resolve_128k_model(self):
        """gpt-5.* class: 128K × 0.7 = 89_600, below 120K ceiling."""
        cfg = ContextConfig()
        assert ContextConfig.resolve_optimal_budget(cfg, 128_000) == 89_600

    def test_auto_resolve_caps_at_ceiling_for_large_window(self):
        """Gemini 2.5 Pro @ 2M: capped at the 120K quality plateau."""
        cfg = ContextConfig()
        assert ContextConfig.resolve_optimal_budget(cfg, 2_000_000) == 120_000

    def test_auto_resolve_scales_down_for_small_window(self):
        """Llama 3.1 8K: 8K × 0.7 = 5_600 — far below the old fixed 50K."""
        cfg = ContextConfig()
        assert ContextConfig.resolve_optimal_budget(cfg, 8_000) == 5_600

    def test_auto_resolve_never_returns_zero(self):
        """Safety rail: even with pathological inputs we never return 0."""
        cfg = ContextConfig(optimal_budget_fraction=0.0)
        # Fraction 0.0 × any window = 0, but the helper floors at 1 so
        # the ledger never gets a zero denominator.
        assert ContextConfig.resolve_optimal_budget(cfg, 128_000) == 1

    def test_custom_fraction_and_ceiling_are_respected(self):
        """Power users can tighten the fraction or raise the ceiling."""
        cfg = ContextConfig(
            optimal_budget_fraction=0.5,
            optimal_budget_ceiling=200_000,
        )
        # 128K × 0.5 = 64K, below 200K ceiling
        assert ContextConfig.resolve_optimal_budget(cfg, 128_000) == 64_000
        # 1M × 0.5 = 500K, capped at 200K ceiling
        assert ContextConfig.resolve_optimal_budget(cfg, 1_000_000) == 200_000


class TestObservabilityConfig:
    def test_defaults(self):
        obs = ObservabilityConfig()
        assert obs.tracing is False
        assert obs.verbose is False
        assert obs.log_level == "INFO"

    def test_effective_log_level_normal(self):
        obs = ObservabilityConfig(log_level="WARNING")
        assert obs.effective_log_level == "WARNING"

    def test_effective_log_level_verbose_overrides(self):
        obs = ObservabilityConfig(verbose=True, log_level="WARNING")
        assert obs.effective_log_level == "DEBUG"


class TestAgentConfigIntegration:
    def test_default_context_none(self):
        ac = AgentConfig()
        assert ac.context is None
        assert ac.observability is None

    def test_with_context(self):
        ac = AgentConfig(context=ContextConfig(max_context_tokens=50_000))
        assert ac.context is not None
        assert ac.context.max_context_tokens == 50_000

    def test_effective_tracing_legacy(self):
        ac = AgentConfig(enable_tracing=True)
        assert ac.effective_tracing is True

    def test_effective_tracing_observability_overrides(self):
        ac = AgentConfig(
            enable_tracing=False,
            observability=ObservabilityConfig(tracing=True),
        )
        assert ac.effective_tracing is True

    def test_effective_verbose_legacy(self):
        ac = AgentConfig(verbose=True)
        assert ac.effective_verbose is True

    def test_effective_verbose_observability_overrides(self):
        ac = AgentConfig(
            verbose=False,
            observability=ObservabilityConfig(verbose=True),
        )
        assert ac.effective_verbose is True
