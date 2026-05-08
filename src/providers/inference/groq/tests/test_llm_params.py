"""GroqLLMParams tests."""

from __future__ import annotations

from nucleusiq_groq.llm_params import GroqLLMParams


def test_groq_llm_params_to_call_kwargs_only_non_none() -> None:
    p = GroqLLMParams(parallel_tool_calls=True, user=None)
    assert p.to_call_kwargs() == {"parallel_tool_calls": True}


def test_groq_llm_params_includes_base_fields() -> None:
    p = GroqLLMParams(temperature=0.2, seed=3)
    d = p.to_call_kwargs()
    assert d["temperature"] == 0.2
    assert d["seed"] == 3
