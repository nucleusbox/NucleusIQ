"""GroqLLMParams tests."""

from __future__ import annotations

from nucleusiq_groq.llm_params import GroqLLMParams


def test_groq_llm_params_to_call_kwargs_only_non_none() -> None:
    p = GroqLLMParams(parallel_tool_calls=True, user=None)
    assert p.to_call_kwargs() == {"parallel_tool_calls": True}


def test_groq_llm_params_strict_not_in_call_kwargs() -> None:
    p = GroqLLMParams(
        strict_model_capabilities=True,
        parallel_tool_calls=True,
    )
    d = p.to_call_kwargs()
    assert "strict_model_capabilities" not in d
    assert d.get("parallel_tool_calls") is True
