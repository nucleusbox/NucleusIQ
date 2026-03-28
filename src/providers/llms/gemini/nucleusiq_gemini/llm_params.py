"""Gemini-specific type-safe LLM call parameters.

Extends the base :class:`~nucleusiq.llms.llm_params.LLMParams` with every
Gemini-specific parameter — ``top_k``, ``safety_settings``,
``thinking_config``, etc.

All constrained fields use ``Literal`` types so IDEs show valid values and
Pydantic rejects typos immediately.

Usage::

    from nucleusiq_gemini import GeminiLLMParams

    params = GeminiLLMParams(
        temperature=0.3,
        top_k=40,
        thinking_config=GeminiThinkingConfig(thinking_budget=1024),
    )
"""

from __future__ import annotations

from typing import Any, Literal

from nucleusiq.llms.llm_params import LLMParams
from pydantic import BaseModel, ConfigDict, Field

# ======================================================================== #
# Nested config models                                                     #
# ======================================================================== #


class GeminiSafetySettings(BaseModel):
    """Safety settings for Gemini content generation.

    Each setting maps a harm category to a blocking threshold.
    """

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    threshold: Literal[
        "BLOCK_NONE",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "OFF",
    ]


class GeminiThinkingConfig(BaseModel):
    """Configuration for model thinking/reasoning (Gemini 2.5+ models)."""

    model_config = ConfigDict(extra="forbid")

    thinking_budget: int = Field(
        ...,
        ge=0,
        description=(
            "Token budget for model thinking. "
            "0 disables thinking. Higher values allow deeper reasoning."
        ),
    )


# ======================================================================== #
# GeminiLLMParams                                                          #
# ======================================================================== #


class GeminiLLMParams(LLMParams):
    """Gemini-specific LLM call parameters with full type safety.

    Extends :class:`LLMParams` with parameters unique to Google's
    Gemini API.  IDE autocomplete shows all valid fields **and**
    constrained values.

    Examples::

        # Basic tuning
        params = GeminiLLMParams(temperature=0.3, top_k=40, seed=42)

        # With thinking budget for Gemini 2.5+ models
        params = GeminiLLMParams(
            thinking_config=GeminiThinkingConfig(thinking_budget=2048),
        )

        # With safety settings
        params = GeminiLLMParams(
            safety_settings=[
                GeminiSafetySettings(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ],
        )
    """

    # --- Gemini-specific sampling ---
    top_k: int | None = Field(
        None,
        ge=1,
        description=(
            "Top-K sampling. Considers the top K most probable tokens. "
            "Lower values = more focused output."
        ),
    )

    # --- Thinking / Reasoning ---
    thinking_config: GeminiThinkingConfig | None = Field(
        None,
        description="Thinking budget config for Gemini 2.5+ models.",
    )

    # --- Safety ---
    safety_settings: list[GeminiSafetySettings] | None = Field(
        None,
        description="Per-category safety thresholds for content filtering.",
    )

    # --- Output format ---
    response_mime_type: Literal["text/plain", "application/json"] | None = Field(
        None,
        description=(
            "MIME type of the generated response. "
            "Use 'application/json' for structured output."
        ),
    )
    response_json_schema: dict[str, Any] | None = Field(
        None,
        description="JSON schema for structured output (requires response_mime_type='application/json').",
    )

    # --- Candidate count ---
    candidate_count: int | None = Field(
        None,
        ge=1,
        le=8,
        description="Number of response candidates to generate.",
    )
