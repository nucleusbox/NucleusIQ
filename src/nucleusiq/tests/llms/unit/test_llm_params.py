"""
Tests for nucleusiq.llms.llm_params.LLMParams — the type-safe base LLM parameter class.

Covers:
 - Construction with valid / invalid values
 - Typo detection (extra="forbid")
 - Range validation (temperature, max_tokens, etc.)
 - to_call_kwargs() — only non-None values
 - merge() — priority chain (base < override)
 - Integration with AgentConfig.llm_params
"""

import pytest
from nucleusiq.llms.llm_params import LLMParams
from pydantic import ValidationError

# ===================================================================== #
# Construction & Validation                                              #
# ===================================================================== #


class TestLLMParamsConstruction:
    """Basic construction and Pydantic validation."""

    def test_default_all_none(self):
        """All fields default to None."""
        p = LLMParams()
        assert p.temperature is None
        assert p.max_tokens is None
        assert p.top_p is None
        assert p.frequency_penalty is None
        assert p.presence_penalty is None
        assert p.seed is None
        assert p.stop is None
        assert p.n is None
        assert p.stream is None

    def test_valid_construction(self):
        """Construct with several valid params."""
        p = LLMParams(temperature=0.7, max_tokens=256, seed=42)
        assert p.temperature == 0.7
        assert p.max_tokens == 256
        assert p.seed == 42

    def test_temperature_lower_bound(self):
        """temperature >= 0.0."""
        LLMParams(temperature=0.0)  # OK
        with pytest.raises(ValidationError):
            LLMParams(temperature=-0.1)

    def test_temperature_upper_bound(self):
        """temperature <= 2.0."""
        LLMParams(temperature=2.0)  # OK
        with pytest.raises(ValidationError):
            LLMParams(temperature=2.1)

    def test_max_tokens_must_be_positive(self):
        """max_tokens >= 1."""
        LLMParams(max_tokens=1)
        with pytest.raises(ValidationError):
            LLMParams(max_tokens=0)

    def test_top_p_range(self):
        """top_p in [0.0, 1.0]."""
        LLMParams(top_p=0.0)
        LLMParams(top_p=1.0)
        with pytest.raises(ValidationError):
            LLMParams(top_p=1.1)

    def test_frequency_penalty_range(self):
        """frequency_penalty in [-2.0, 2.0]."""
        LLMParams(frequency_penalty=-2.0)
        LLMParams(frequency_penalty=2.0)
        with pytest.raises(ValidationError):
            LLMParams(frequency_penalty=2.1)

    def test_presence_penalty_range(self):
        """presence_penalty in [-2.0, 2.0]."""
        LLMParams(presence_penalty=-2.0)
        LLMParams(presence_penalty=2.0)
        with pytest.raises(ValidationError):
            LLMParams(presence_penalty=-2.1)

    def test_n_range(self):
        """n in [1, 128]."""
        LLMParams(n=1)
        LLMParams(n=128)
        with pytest.raises(ValidationError):
            LLMParams(n=0)
        with pytest.raises(ValidationError):
            LLMParams(n=129)


# ===================================================================== #
# Typo Detection (extra="forbid")                                       #
# ===================================================================== #


class TestTypoDetection:
    """extra='forbid' catches unknown fields immediately."""

    def test_typo_max_token(self):
        """Rejects max_token (should be max_tokens)."""
        with pytest.raises(ValidationError, match="max_token"):
            LLMParams(max_token=100)  # type: ignore

    def test_typo_temprature(self):
        """Rejects temprature (should be temperature)."""
        with pytest.raises(ValidationError, match="temprature"):
            LLMParams(temprature=0.5)  # type: ignore

    def test_typo_stops(self):
        """Rejects stops (should be stop)."""
        with pytest.raises(ValidationError, match="stops"):
            LLMParams(stops=["END"])  # type: ignore

    def test_completely_unknown_field(self):
        """Rejects any unknown field."""
        with pytest.raises(ValidationError):
            LLMParams(foo_bar="baz")  # type: ignore


# ===================================================================== #
# to_call_kwargs()                                                       #
# ===================================================================== #


class TestToCallKwargs:
    """to_call_kwargs() should only return non-None values."""

    def test_empty_when_all_none(self):
        """No defaults leak through."""
        assert LLMParams().to_call_kwargs() == {}

    def test_returns_set_fields_only(self):
        p = LLMParams(temperature=0.3, seed=42)
        kwargs = p.to_call_kwargs()
        assert kwargs == {"temperature": 0.3, "seed": 42}
        assert "max_tokens" not in kwargs

    def test_full_params(self):
        p = LLMParams(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            seed=7,
            stop=["END"],
            n=2,
            stream=True,
        )
        kwargs = p.to_call_kwargs()
        assert len(kwargs) == 9
        assert kwargs["stop"] == ["END"]
        assert kwargs["stream"] is True


# ===================================================================== #
# merge()                                                                #
# ===================================================================== #


class TestMerge:
    """merge() should combine two LLMParams with override priority."""

    def test_merge_none_returns_self(self):
        base = LLMParams(temperature=0.5)
        result = base.merge(None)
        assert result.temperature == 0.5

    def test_override_takes_precedence(self):
        base = LLMParams(temperature=0.5, max_tokens=100)
        override = LLMParams(temperature=0.9)
        result = base.merge(override)
        assert result.temperature == 0.9
        assert result.max_tokens == 100  # kept from base

    def test_new_fields_added(self):
        base = LLMParams(temperature=0.5)
        override = LLMParams(seed=42, n=3)
        result = base.merge(override)
        assert result.temperature == 0.5
        assert result.seed == 42
        assert result.n == 3

    def test_merge_does_not_mutate(self):
        base = LLMParams(temperature=0.5)
        override = LLMParams(temperature=1.0)
        base.merge(override)
        assert base.temperature == 0.5  # unchanged


# ===================================================================== #
# Integration: AgentConfig                                               #
# ===================================================================== #


class TestAgentConfigIntegration:
    """LLMParams integrates with AgentConfig.llm_params."""

    def test_agent_config_accepts_llm_params(self):
        from nucleusiq.agents.config.agent_config import AgentConfig

        params = LLMParams(temperature=0.3, seed=42)
        config = AgentConfig(llm_params=params)
        assert config.llm_params is not None
        assert config.llm_params.temperature == 0.3
        assert config.llm_params.seed == 42

    def test_agent_config_default_is_none(self):
        from nucleusiq.agents.config.agent_config import AgentConfig

        config = AgentConfig()
        assert config.llm_params is None

    def test_agent_config_coerces_dict_to_llm_params(self):
        """Pydantic coerces a valid dict into LLMParams automatically."""
        from nucleusiq.agents.config.agent_config import AgentConfig

        config = AgentConfig(llm_params={"temperature": 0.5})
        assert isinstance(config.llm_params, LLMParams)
        assert config.llm_params.temperature == 0.5

    def test_agent_config_rejects_bad_dict(self):
        """Dict with unknown keys is rejected."""
        from nucleusiq.agents.config.agent_config import AgentConfig

        with pytest.raises(ValidationError):
            AgentConfig(llm_params={"typo_field": 0.5})  # type: ignore


# ===================================================================== #
# Integration: Agent._resolve_llm_params                                #
# ===================================================================== #


class TestResolveIntegration:
    """Agent._resolve_llm_params merges config + per-execute."""

    def _make_agent(self, config_params=None):
        """Helper to build an Agent with optional llm_params in config."""
        from nucleusiq.agents.agent import Agent
        from nucleusiq.agents.config.agent_config import AgentConfig
        from nucleusiq.llms.mock_llm import MockLLM
        from nucleusiq.prompts.zero_shot import ZeroShotPrompt

        config = AgentConfig(llm_params=config_params)
        prompt = ZeroShotPrompt(template="Hello {task}")
        return Agent(
            name="test-agent",
            role="tester",
            objective="test the merge chain",
            llm=MockLLM(),
            prompt=prompt,
            config=config,
        )

    def test_no_overrides(self):
        agent = self._make_agent()
        assert agent._resolve_llm_params() == {}

    def test_config_only(self):
        params = LLMParams(temperature=0.3)
        agent = self._make_agent(config_params=params)
        assert agent._resolve_llm_params() == {"temperature": 0.3}

    def test_per_execute_only(self):
        agent = self._make_agent()
        per = LLMParams(seed=42)
        assert agent._resolve_llm_params(per_execute=per) == {"seed": 42}

    def test_per_execute_overrides_config(self):
        config_p = LLMParams(temperature=0.3, max_tokens=100)
        agent = self._make_agent(config_params=config_p)
        per = LLMParams(temperature=0.9)
        result = agent._resolve_llm_params(per_execute=per)
        assert result["temperature"] == 0.9
        assert result["max_tokens"] == 100
