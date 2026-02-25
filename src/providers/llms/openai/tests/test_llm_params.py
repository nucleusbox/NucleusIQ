"""
Tests for nucleusiq_openai.llm_params.OpenAILLMParams — OpenAI-specific typed params.

Covers:
 - All OpenAI-specific fields: validation, Literal constraints
 - Inheritance from base LLMParams
 - Typo rejection (extra="forbid")
 - to_call_kwargs()
 - merge() with base LLMParams and OpenAILLMParams
 - AudioOutputConfig nested model
 - Integration with AgentConfig
"""

import pytest
from nucleusiq.llms.llm_params import LLMParams
from pydantic import ValidationError

from nucleusiq_openai.llm_params import AudioOutputConfig, OpenAILLMParams

# ===================================================================== #
# Construction & Validation                                              #
# ===================================================================== #


class TestOpenAILLMParamsConstruction:
    """OpenAI-specific fields and validation."""

    def test_inherits_base_fields(self):
        """OpenAILLMParams has all base LLMParams fields."""
        p = OpenAILLMParams(temperature=0.5, max_tokens=200, seed=42)
        assert p.temperature == 0.5
        assert p.max_tokens == 200
        assert p.seed == 42

    def test_openai_specific_defaults(self):
        """All OpenAI-specific fields default to None."""
        p = OpenAILLMParams()
        assert p.reasoning_effort is None
        assert p.service_tier is None
        assert p.modalities is None
        assert p.audio is None
        assert p.parallel_tool_calls is None
        assert p.logprobs is None
        assert p.top_logprobs is None
        assert p.metadata is None
        assert p.store is None
        assert p.truncation is None
        assert p.max_tool_calls is None
        assert p.safety_identifier is None

    def test_reasoning_effort_valid(self):
        """Valid reasoning_effort values."""
        for val in ["none", "minimal", "low", "medium", "high", "xhigh"]:
            p = OpenAILLMParams(reasoning_effort=val)
            assert p.reasoning_effort == val

    def test_reasoning_effort_invalid(self):
        """Invalid reasoning_effort value rejected."""
        with pytest.raises(ValidationError, match="reasoning_effort"):
            OpenAILLMParams(reasoning_effort="ultra")

    def test_service_tier_valid(self):
        for val in ["auto", "default", "flex", "priority"]:
            p = OpenAILLMParams(service_tier=val)
            assert p.service_tier == val

    def test_service_tier_invalid(self):
        with pytest.raises(ValidationError, match="service_tier"):
            OpenAILLMParams(service_tier="turbo")

    def test_modalities_valid(self):
        p = OpenAILLMParams(modalities=["text", "audio"])
        assert p.modalities == ["text", "audio"]

    def test_modalities_invalid_value(self):
        with pytest.raises(ValidationError):
            OpenAILLMParams(modalities=["text", "video"])

    def test_truncation_valid(self):
        for val in ["auto", "disabled"]:
            p = OpenAILLMParams(truncation=val)
            assert p.truncation == val

    def test_truncation_invalid(self):
        with pytest.raises(ValidationError, match="truncation"):
            OpenAILLMParams(truncation="always")

    def test_top_logprobs_range(self):
        OpenAILLMParams(top_logprobs=0)
        OpenAILLMParams(top_logprobs=20)
        with pytest.raises(ValidationError):
            OpenAILLMParams(top_logprobs=21)

    def test_max_tool_calls_positive(self):
        OpenAILLMParams(max_tool_calls=1)
        with pytest.raises(ValidationError):
            OpenAILLMParams(max_tool_calls=0)


# ===================================================================== #
# AudioOutputConfig                                                      #
# ===================================================================== #


class TestAudioOutputConfig:
    """Nested AudioOutputConfig validation."""

    def test_valid_audio_config(self):
        audio = AudioOutputConfig(voice="nova", format="mp3")
        assert audio.voice == "nova"
        assert audio.format == "mp3"

    def test_default_format(self):
        audio = AudioOutputConfig(voice="alloy")
        assert audio.format == "mp3"

    def test_invalid_format(self):
        with pytest.raises(ValidationError, match="format"):
            AudioOutputConfig(voice="nova", format="wma")

    def test_voice_required(self):
        with pytest.raises(ValidationError, match="voice"):
            AudioOutputConfig()  # type: ignore

    def test_audio_in_openai_params(self):
        p = OpenAILLMParams(
            modalities=["text", "audio"],
            audio=AudioOutputConfig(voice="coral", format="wav"),
        )
        assert p.audio.voice == "coral"
        assert p.audio.format == "wav"


# ===================================================================== #
# Typo Detection                                                         #
# ===================================================================== #


class TestOpenAITypoDetection:
    """extra='forbid' rejects unknown OpenAI-specific fields."""

    def test_typo_reasoning(self):
        with pytest.raises(ValidationError, match="reasoning"):
            OpenAILLMParams(reasoning="high")  # type: ignore

    def test_typo_service_teir(self):
        with pytest.raises(ValidationError, match="service_teir"):
            OpenAILLMParams(service_teir="flex")  # type: ignore

    def test_completely_unknown(self):
        with pytest.raises(ValidationError):
            OpenAILLMParams(xyz_option=True)  # type: ignore


# ===================================================================== #
# to_call_kwargs()                                                       #
# ===================================================================== #


class TestOpenAIToCallKwargs:
    """to_call_kwargs() includes both base and provider-specific fields."""

    def test_only_set_fields(self):
        p = OpenAILLMParams(temperature=0.3, reasoning_effort="high")
        kwargs = p.to_call_kwargs()
        assert kwargs == {"temperature": 0.3, "reasoning_effort": "high"}
        assert "service_tier" not in kwargs

    def test_audio_config_serialized(self):
        p = OpenAILLMParams(
            audio=AudioOutputConfig(voice="nova", format="mp3"),
        )
        kwargs = p.to_call_kwargs()
        assert "audio" in kwargs
        assert kwargs["audio"]["voice"] == "nova"

    def test_empty_when_all_none(self):
        assert OpenAILLMParams().to_call_kwargs() == {}


# ===================================================================== #
# merge()                                                                #
# ===================================================================== #


class TestOpenAIMerge:
    """merge() works across base and provider-specific LLMParams."""

    def test_openai_merge_with_base(self):
        """OpenAILLMParams override merges on top of base LLMParams."""
        base = LLMParams(temperature=0.5, max_tokens=100)
        override = OpenAILLMParams(temperature=0.9, reasoning_effort="low")
        result = base.merge(override)
        assert result.temperature == 0.9
        assert result.max_tokens == 100  # kept from base

    def test_openai_merge_openai(self):
        """Two OpenAILLMParams merge correctly."""
        base = OpenAILLMParams(reasoning_effort="high", service_tier="auto")
        override = OpenAILLMParams(reasoning_effort="low", seed=7)
        result = base.merge(override)
        assert result.reasoning_effort == "low"
        assert result.service_tier == "auto"
        assert result.seed == 7

    def test_merge_none_preserves_all(self):
        p = OpenAILLMParams(reasoning_effort="high", temperature=0.3)
        result = p.merge(None)
        assert result.reasoning_effort == "high"
        assert result.temperature == 0.3


# ===================================================================== #
# Integration: AgentConfig + OpenAILLMParams                            #
# ===================================================================== #


class TestAgentConfigWithOpenAI:
    """AgentConfig accepts OpenAILLMParams via polymorphism."""

    def test_accepts_openai_params(self):
        from nucleusiq.agents.config.agent_config import AgentConfig

        params = OpenAILLMParams(temperature=0.3, reasoning_effort="medium")
        config = AgentConfig(llm_params=params)
        assert config.llm_params.temperature == 0.3
        # Access provider-specific field via the actual instance
        actual = config.llm_params
        assert isinstance(actual, OpenAILLMParams)
        assert actual.reasoning_effort == "medium"

    def test_serialization_roundtrip(self):
        """model_dump preserves base fields; provider fields via to_call_kwargs."""
        from nucleusiq.agents.config.agent_config import AgentConfig

        params = OpenAILLMParams(temperature=0.5, reasoning_effort="high")
        config = AgentConfig(llm_params=params)

        # to_call_kwargs preserves all set fields (base + provider)
        kwargs = config.llm_params.to_call_kwargs()
        assert kwargs["temperature"] == 0.5
        assert kwargs["reasoning_effort"] == "high"

        # The runtime object is still OpenAILLMParams
        assert isinstance(config.llm_params, OpenAILLMParams)
        assert config.llm_params.reasoning_effort == "high"
