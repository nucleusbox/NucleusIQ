"""Typed call parameters."""

from __future__ import annotations

import pytest
from nucleusiq.llms.llm_params import LLMParams
from nucleusiq_openai_compatible.llm_params import OpenAICompatibleLLMParams
from pydantic import ValidationError


class TestContract:
    def test_extends_the_shared_base(self) -> None:
        assert issubclass(OpenAICompatibleLLMParams, LLMParams)

    def test_all_provider_fields_optional(self) -> None:
        params = OpenAICompatibleLLMParams()
        assert params.parallel_tool_calls is None
        assert params.user is None
        assert params.extra_body is None
        assert params.strict_capabilities is False

    def test_unknown_field_rejected(self) -> None:
        # A silently ignored typo would look like the server ignoring the knob.
        with pytest.raises(ValidationError):
            OpenAICompatibleLLMParams(paralel_tool_calls=True)

    @pytest.mark.parametrize(
        "field", ["service_tier", "store", "logit_bias", "modalities", "prediction"]
    )
    def test_openai_cloud_only_fields_absent(self, field: str) -> None:
        assert field not in OpenAICompatibleLLMParams.model_fields, (
            f"{field} is OpenAI-cloud-only; typing it here would invite a "
            "400 from a strict self-hosted server"
        )

    def test_engine_knobs_go_through_extra_body(self) -> None:
        params = OpenAICompatibleLLMParams(
            extra_body={"top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1}
        )
        assert params.extra_body["top_k"] == 40


class TestToCallKwargs:
    def test_empty_when_unset(self) -> None:
        assert OpenAICompatibleLLMParams().to_call_kwargs() == {}

    def test_includes_set_fields(self) -> None:
        kwargs = OpenAICompatibleLLMParams(user="svc", seed=7).to_call_kwargs()
        assert kwargs["user"] == "svc"
        assert kwargs["seed"] == 7

    def test_drops_none(self) -> None:
        kwargs = OpenAICompatibleLLMParams(user="svc").to_call_kwargs()
        assert "parallel_tool_calls" not in kwargs
        assert "extra_body" not in kwargs

    def test_strict_flag_is_config_not_payload(self) -> None:
        kwargs = OpenAICompatibleLLMParams(strict_capabilities=True).to_call_kwargs()
        assert "strict_capabilities" not in kwargs, (
            "it controls local gating and would be rejected on the wire"
        )

    def test_false_is_kept(self) -> None:
        kwargs = OpenAICompatibleLLMParams(parallel_tool_calls=False).to_call_kwargs()
        assert kwargs["parallel_tool_calls"] is False

    def test_extra_body_forwarded(self) -> None:
        kwargs = OpenAICompatibleLLMParams(extra_body={"top_k": 40}).to_call_kwargs()
        assert kwargs["extra_body"] == {"top_k": 40}
