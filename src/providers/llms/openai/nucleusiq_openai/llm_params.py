"""
OpenAI-specific type-safe LLM call parameters.

Extends the base :class:`~nucleusiq.llms.llm_params.LLMParams` with every
OpenAI-specific parameter — ``reasoning_effort``, ``service_tier``,
``modalities``, ``audio``, etc.

All constrained fields use ``Literal`` types so IDEs show valid values and
Pydantic rejects typos immediately.

Usage::

    from nucleusiq_openai import OpenAILLMParams

    params = OpenAILLMParams(
        temperature=0.3,
        reasoning_effort="high",
        service_tier="flex",
    )
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

from nucleusiq.llms.llm_params import LLMParams


# ======================================================================== #
# Nested config models                                                     #
# ======================================================================== #


class AudioOutputConfig(BaseModel):
    """Audio output configuration for multimodal responses."""

    model_config = ConfigDict(extra="forbid")

    voice: str = Field(
        ...,
        description=(
            "Voice for audio output. Built-in voices: alloy, ash, ballad, "
            "coral, echo, fable, nova, onyx, sage, shimmer, marin, cedar."
        ),
    )
    format: Literal["wav", "mp3", "aac", "flac", "opus", "pcm16"] = Field(
        "mp3",
        description="Audio output format.",
    )


# ======================================================================== #
# OpenAILLMParams                                                          #
# ======================================================================== #


class OpenAILLMParams(LLMParams):
    """
    OpenAI-specific LLM call parameters with full type safety.

    Extends :class:`LLMParams` with parameters unique to OpenAI's
    Chat Completions and Responses APIs.  IDE autocomplete shows all
    valid fields **and** constrained values.

    Examples::

        # Basic tuning
        params = OpenAILLMParams(temperature=0.3, seed=42)

        # Reasoning model cost control
        params = OpenAILLMParams(reasoning_effort="low", service_tier="flex")

        # Audio output
        params = OpenAILLMParams(
            modalities=["text", "audio"],
            audio=AudioOutputConfig(voice="nova", format="mp3"),
        )
    """

    # --- Reasoning (o-series / gpt-5 models) ---
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    ] = Field(
        None,
        description=(
            "Reasoning depth for o-series and gpt-5 models. "
            "Lower values = faster + cheaper."
        ),
    )

    # --- Cost & Performance ---
    service_tier: Optional[
        Literal["auto", "default", "flex", "priority"]
    ] = Field(
        None,
        description=(
            "Processing tier. 'flex' = ~50%% cheaper (async processing). "
            "'priority' = fastest."
        ),
    )

    # --- Multimodal Output ---
    modalities: Optional[List[Literal["text", "audio"]]] = Field(
        None,
        description='Output modalities. ["text"] or ["text", "audio"].',
    )
    audio: Optional[AudioOutputConfig] = Field(
        None,
        description="Audio output config (required when modalities includes 'audio').",
    )

    # --- Tool Behaviour ---
    parallel_tool_calls: Optional[bool] = Field(
        None,
        description="Allow model to call multiple tools in parallel.",
    )

    # --- Logging & Debugging ---
    logprobs: Optional[bool] = Field(
        None,
        description="Return log probabilities of output tokens.",
    )
    top_logprobs: Optional[int] = Field(
        None, ge=0, le=20,
        description="Number of most likely tokens to return at each position.",
    )

    # --- Storage & Tracking ---
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Up to 16 key-value pairs for tracking/querying.",
    )
    store: Optional[bool] = Field(
        None,
        description="Whether to store response for later retrieval via API.",
    )

    # --- Caching ---
    prompt_cache_key: Optional[str] = Field(
        None,
        description="Stable identifier for prompt cache optimisation.",
    )
    prompt_cache_retention: Optional[Literal["in-memory", "24h"]] = Field(
        None,
        description="Cache retention policy.",
    )

    # --- Responses API Specific ---
    truncation: Optional[Literal["auto", "disabled"]] = Field(
        None,
        description=(
            "Context overflow handling. 'auto' = truncate from the "
            "beginning of the conversation. 'disabled' = return 400 error."
        ),
    )
    max_tool_calls: Optional[int] = Field(
        None, ge=1,
        description="Maximum total built-in tool calls allowed per response.",
    )

    # --- Safety ---
    safety_identifier: Optional[str] = Field(
        None,
        description="Hashed end-user identifier for abuse detection.",
    )
