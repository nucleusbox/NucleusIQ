"""Tests for nucleusiq_gemini.llm_params.GeminiLLMParams."""

import pytest
from nucleusiq.llms.llm_params import LLMParams
from nucleusiq_gemini.llm_params import (
    GeminiLLMParams,
    GeminiSafetySettings,
    GeminiThinkingConfig,
)
from pydantic import ValidationError


class TestGeminiLLMParamsConstruction:
    def test_inherits_base_fields(self):
        p = GeminiLLMParams(temperature=0.5, max_output_tokens=200, seed=42)
        assert p.temperature == 0.5
        assert p.max_output_tokens == 200
        assert p.seed == 42

    def test_gemini_specific_defaults(self):
        p = GeminiLLMParams()
        assert p.top_k is None
        assert p.thinking_config is None
        assert p.safety_settings is None
        assert p.response_mime_type is None
        assert p.response_json_schema is None
        assert p.candidate_count is None

    def test_top_k_valid(self):
        p = GeminiLLMParams(top_k=40)
        assert p.top_k == 40

    def test_top_k_minimum(self):
        with pytest.raises(ValidationError):
            GeminiLLMParams(top_k=0)

    def test_thinking_config(self):
        tc = GeminiThinkingConfig(thinking_budget=2048)
        p = GeminiLLMParams(thinking_config=tc)
        assert p.thinking_config.thinking_budget == 2048

    def test_safety_settings(self):
        ss = GeminiSafetySettings(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_ONLY_HIGH",
        )
        p = GeminiLLMParams(safety_settings=[ss])
        assert len(p.safety_settings) == 1

    def test_response_mime_type_valid(self):
        p = GeminiLLMParams(response_mime_type="application/json")
        assert p.response_mime_type == "application/json"

    def test_response_mime_type_invalid(self):
        with pytest.raises(ValidationError):
            GeminiLLMParams(response_mime_type="invalid/type")

    def test_candidate_count_valid(self):
        p = GeminiLLMParams(candidate_count=4)
        assert p.candidate_count == 4

    def test_candidate_count_too_high(self):
        with pytest.raises(ValidationError):
            GeminiLLMParams(candidate_count=9)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            GeminiLLMParams(bogus_field="nope")

    def test_isinstance_base(self):
        p = GeminiLLMParams()
        assert isinstance(p, LLMParams)


class TestGeminiSafetySettings:
    def test_valid(self):
        s = GeminiSafetySettings(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE",
        )
        assert s.category == "HARM_CATEGORY_HATE_SPEECH"

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            GeminiSafetySettings(category="INVALID", threshold="BLOCK_NONE")

    def test_invalid_threshold(self):
        with pytest.raises(ValidationError):
            GeminiSafetySettings(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="INVALID",
            )

    def test_all_categories(self):
        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]
        for cat in categories:
            s = GeminiSafetySettings(category=cat, threshold="BLOCK_NONE")
            assert s.category == cat

    def test_all_thresholds(self):
        thresholds = [
            "BLOCK_NONE",
            "BLOCK_LOW_AND_ABOVE",
            "BLOCK_MEDIUM_AND_ABOVE",
            "BLOCK_ONLY_HIGH",
            "OFF",
        ]
        for th in thresholds:
            s = GeminiSafetySettings(
                category="HARM_CATEGORY_HARASSMENT",
                threshold=th,
            )
            assert s.threshold == th


class TestGeminiThinkingConfig:
    def test_valid(self):
        tc = GeminiThinkingConfig(thinking_budget=1024)
        assert tc.thinking_budget == 1024

    def test_zero(self):
        tc = GeminiThinkingConfig(thinking_budget=0)
        assert tc.thinking_budget == 0

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            GeminiThinkingConfig(thinking_budget=-1)

    def test_required(self):
        with pytest.raises(ValidationError):
            GeminiThinkingConfig()

    def test_extra_rejected(self):
        with pytest.raises(ValidationError):
            GeminiThinkingConfig(thinking_budget=100, extra="nope")


class TestToCallKwargs:
    def test_empty(self):
        p = GeminiLLMParams()
        kwargs = p.to_call_kwargs()
        assert kwargs == {}

    def test_with_values(self):
        p = GeminiLLMParams(temperature=0.3, top_k=40, seed=42)
        kwargs = p.to_call_kwargs()
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_k"] == 40
        assert kwargs["seed"] == 42

    def test_none_excluded(self):
        p = GeminiLLMParams(temperature=0.5)
        kwargs = p.to_call_kwargs()
        assert "top_k" not in kwargs
        assert "safety_settings" not in kwargs


class TestMerge:
    def test_merge_with_base(self):
        base = LLMParams(temperature=0.5)
        override = GeminiLLMParams(top_k=40)
        merged = base.merge(override)
        assert isinstance(merged, GeminiLLMParams)
        assert merged.temperature == 0.5
        assert merged.top_k == 40

    def test_override_takes_precedence(self):
        base = GeminiLLMParams(temperature=0.5, top_k=20)
        override = GeminiLLMParams(temperature=0.9)
        merged = base.merge(override)
        assert merged.temperature == 0.9

    def test_merge_with_none(self):
        p = GeminiLLMParams(temperature=0.5)
        merged = p.merge(None)
        assert merged.temperature == 0.5
