"""Tests for :class:`AnthropicLLMParams`."""

from __future__ import annotations

from nucleusiq_anthropic.llm_params import AnthropicLLMParams


def test_top_k_added_only_when_set() -> None:
    merged = AnthropicLLMParams(top_k=10).to_call_kwargs()
    assert merged["top_k"] == 10


def test_empty_optional_fields_omitted_from_merge() -> None:
    merged = AnthropicLLMParams().to_call_kwargs()
    assert merged == {}
