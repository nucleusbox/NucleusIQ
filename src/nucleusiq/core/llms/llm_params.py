"""
Type-safe LLM call parameters.

Provides a base ``LLMParams`` class with universally supported parameters
(temperature, max_tokens, seed, etc.) and validation.  Provider-specific
subclasses (e.g. ``OpenAILLMParams``) extend this with provider-only
fields such as ``reasoning_effort`` or ``service_tier``.

Design:
    - ``extra = "forbid"`` catches typos immediately.
    - ``to_call_kwargs()`` returns only non-None values as a dict.
    - The merge chain is: LLM defaults < AgentConfig.llm_params < per-execute llm_params.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class LLMParams(BaseModel):
    """
    Base type-safe LLM call parameters.

    All fields are optional — only set what you need to override.
    Provider-specific subclasses add extra typed fields.

    Usage::

        from nucleusiq.llms.llm_params import LLMParams

        params = LLMParams(temperature=0.3, seed=42)
        kwargs = params.to_call_kwargs()
        # → {"temperature": 0.3, "seed": 42}
    """

    model_config = ConfigDict(extra="forbid")

    # --- Sampling ---
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0,
        description="Sampling temperature. 0 = deterministic, 2 = max randomness.",
    )
    max_tokens: Optional[int] = Field(
        None, ge=1,
        description="Maximum tokens to generate.",
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Nucleus sampling. 0.1 = only top 10% probability mass.",
    )
    frequency_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0,
        description="Penalize repeated tokens based on frequency.",
    )
    presence_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0,
        description="Penalize tokens that appear at all in text so far.",
    )

    # --- Control ---
    seed: Optional[int] = Field(
        None,
        description="Fixed seed for deterministic/reproducible outputs.",
    )
    stop: Optional[List[str]] = Field(
        None,
        description="Stop sequences — generation halts when any of these appear.",
    )
    n: Optional[int] = Field(
        None, ge=1, le=128,
        description="Number of completions to generate.",
    )
    stream: Optional[bool] = Field(
        None,
        description="Enable streaming (async generator of chunks).",
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

    def merge(self, override: "LLMParams | None") -> "LLMParams":
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
        # Use the most-derived class for the result
        cls = type(override) if type(override) is not LLMParams else type(self)
        return cls(**base)
