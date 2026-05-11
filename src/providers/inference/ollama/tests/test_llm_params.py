"""LLM params dump."""

from __future__ import annotations

from nucleusiq_ollama.llm_params import OllamaLLMParams


def test_ollama_llm_params_to_kwargs() -> None:
    p = OllamaLLMParams(think="low", keep_alive="10m", temperature=0.2)
    d = p.to_call_kwargs()
    assert d["think"] == "low"
    assert d["keep_alive"] == "10m"
    assert d["temperature"] == 0.2
