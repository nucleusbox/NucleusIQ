"""Tests for ContextConfig and ObservabilityConfig."""

import pytest
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.config.observability_config import ObservabilityConfig
from nucleusiq.agents.context.config import ContextConfig


class TestContextConfig:
    def test_defaults(self):
        cfg = ContextConfig()
        assert cfg.optimal_budget == 50_000
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
        assert cfg.optimal_budget == 50_000

    def test_for_mode_autonomous(self):
        cfg = ContextConfig.for_mode("autonomous")
        assert cfg.tool_compaction_trigger == 0.55
        assert cfg.preserve_recent_turns == 6
        assert cfg.optimal_budget == 40_000

    def test_for_mode_direct(self):
        cfg = ContextConfig.for_mode("direct")
        assert cfg.tool_compaction_trigger == 0.80
        assert cfg.preserve_recent_turns == 2


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
