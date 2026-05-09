"""Groq-specific LLM call parameters (OpenAI-compatible chat surface).

Per Groq OpenAI compatibility docs, several Chat Completions fields are
unsupported; those are stripped in :mod:`nucleusiq_groq._shared.wire`, not
listed here. This model carries supported, typed knobs only.
"""

from __future__ import annotations

from typing import Any

from nucleusiq.llms.llm_params import LLMParams
from pydantic import ConfigDict, Field


class GroqLLMParams(LLMParams):
    """Parameters forwarded to Groq's OpenAI-compatible Chat Completions API."""

    model_config = ConfigDict(extra="forbid")

    parallel_tool_calls: bool | None = Field(
        None,
        description="Allow the model to request multiple tool calls in parallel.",
    )
    strict_model_capabilities: bool = Field(
        False,
        description=(
            "When true, reject ``parallel_tool_calls=True`` if the model is not "
            "in the built-in Groq capability allowlist (see capabilities module)."
        ),
    )
    user: str | None = Field(
        None,
        description="End-user identifier for abuse monitoring (OpenAI-compatible).",
    )

    def to_call_kwargs(self) -> dict[str, Any]:
        """Non-None fields suitable for merging into ``BaseGroq.call`` kwargs."""
        data = self.model_dump(exclude={"strict_model_capabilities"})
        return {k: v for k, v in data.items() if v is not None}
