"""Call parameters for OpenAI-compatible Chat Completions endpoints.

Only parameters that a *generic* OpenAI-compatible server can be expected to
accept are typed here.  OpenAI-cloud-only fields (``service_tier``,
``store``, ``reasoning_effort``, ``logit_bias``, …) are deliberately absent
and are stripped on the wire — strict servers answer ``400 unknown
parameter`` for anything they do not recognize.

Engine-specific knobs travel through :attr:`OpenAICompatibleLLMParams.extra_body`
rather than becoming typed fields, so this class does not grow a new
attribute every time an inference engine invents a sampler.
"""

from __future__ import annotations

from typing import Any

from nucleusiq.llms.llm_params import LLMParams
from pydantic import ConfigDict, Field

__all__ = ["OpenAICompatibleLLMParams"]


class OpenAICompatibleLLMParams(LLMParams):
    """Parameters forwarded to a generic OpenAI-compatible Chat Completions API."""

    model_config = ConfigDict(extra="forbid")

    parallel_tool_calls: bool | None = Field(
        None,
        description=(
            "Allow the model to request multiple tool calls in one turn. "
            "Gated by the engine profile; warns (or raises under "
            "strict_capabilities) when the server does not declare support."
        ),
    )
    user: str | None = Field(
        None,
        description="End-user identifier for abuse monitoring (OpenAI-compatible).",
    )
    extra_body: dict[str, Any] | None = Field(
        None,
        description=(
            "Engine-specific body keys that are not part of the OpenAI schema, "
            "e.g. vLLM's top_k, min_p, repetition_penalty, guided_json, "
            "guided_regex, guided_choice, chat_template_kwargs. Forwarded only "
            "when the engine profile allows a non-standard body."
        ),
    )
    strict_capabilities: bool = Field(
        False,
        description=(
            "Raise instead of warn when a requested parameter is not declared "
            "as supported by the resolved engine profile."
        ),
    )

    def to_call_kwargs(self) -> dict[str, Any]:
        """Non-None fields suitable for merging into ``call`` kwargs."""
        data = self.model_dump(exclude={"strict_capabilities"})
        return {k: v for k, v in data.items() if v is not None}
