"""
Type-safe LLM call parameters.

Provides a base ``LLMParams`` class with **universally supported**
parameters that every LLM provider recognises (temperature, seed,
stop, etc.).  Provider-specific subclasses (``OpenAILLMParams``,
``GeminiLLMParams``) extend this with provider-only fields.

**Design principles:**

- Only parameters supported by *all* major providers live here.
- Parameters whose *name* or *semantics* differ across providers
  (``n`` vs ``candidate_count``, ``stream``) are provider-level.
- ``max_output_tokens`` is the provider-neutral name for "max tokens
  to generate".  Each provider translates it to the API-specific
  parameter (``max_tokens``, ``max_completion_tokens``,
  ``max_output_tokens``) in its payload builder.
- ``extra = "forbid"`` catches typos immediately.
- ``to_call_kwargs()`` returns only non-None values as a dict.
- The merge chain is:
  LLM defaults < AgentConfig.llm_params < per-execute llm_params.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class LLMParams(BaseModel):
    """
    Base type-safe LLM call parameters.

    Contains only parameters that are **universally supported** across
    all LLM providers.  Provider-specific fields live in subclasses
    (e.g. ``OpenAILLMParams``, ``GeminiLLMParams``).

    Usage::

        from nucleusiq.llms.llm_params import LLMParams

        params = LLMParams(temperature=0.3, seed=42)
        kwargs = params.to_call_kwargs()
        # → {"temperature": 0.3, "seed": 42}
    """

    model_config = ConfigDict(extra="forbid")

    # --- Sampling ---
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = deterministic, 2 = max randomness.",
    )
    max_output_tokens: int | None = Field(
        None,
        ge=1,
        description=(
            "Maximum tokens to generate in the response. "
            "Each provider translates this to the appropriate API "
            "parameter (e.g. max_tokens, max_completion_tokens, "
            "max_output_tokens)."
        ),
    )
    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling. 0.1 = only top 10% probability mass.",
    )
    frequency_penalty: float | None = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Penalize repeated tokens based on frequency.",
    )
    presence_penalty: float | None = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Penalize tokens that appear at all in text so far.",
    )

    # --- Control ---
    seed: int | None = Field(
        None,
        description="Fixed seed for deterministic/reproducible outputs.",
    )
    stop: List[str] | None = Field(
        None,
        description="Stop sequences — generation halts when any of these appear.",
    )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def to_call_kwargs(self) -> Dict[str, Any]:
        """
        Convert to a kwargs dict for ``llm.call()``, excluding ``None`` values.

        Returns:
            Dict of parameter name → value (only non-None entries).
        """
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def merge(self, override: LLMParams | None) -> LLMParams:
        """
        Return a **new** instance with *override* values taking precedence.

        Only non-None fields from *override* replace fields in ``self``.

        Args:
            override: Another LLMParams (or subclass) to merge on top of self.

        Returns:
            New LLMParams with merged values.
        """
        if override is None:
            return self
        base = self.to_call_kwargs()
        top = override.to_call_kwargs()
        base.update(top)
        cls = type(override) if type(override) is not LLMParams else type(self)
        return cls(**base)
