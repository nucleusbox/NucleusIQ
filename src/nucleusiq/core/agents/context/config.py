"""ContextConfig — immutable configuration for context window management.

All fields have sensible defaults so users can enable context management
with zero configuration::

    agent = Agent(config=AgentConfig(context=ContextConfig()))

Or override selectively::

    agent = Agent(
        config=AgentConfig(
            context=ContextConfig(
                optimal_budget=30_000, strategy=ContextStrategy.PROGRESSIVE
            )
        )
    )

Design:
    - ``optimal_budget: 50_000`` → quality-optimized default. Research shows
      effective utilization drops to 60% beyond 100K tokens.
    - ``max_context_tokens: None`` → auto-detect from provider via
      ``BaseLLM.get_context_window()``.  Used as hard ceiling.
    - ``strategy: ContextStrategy.NONE`` → complete opt-out with zero overhead.
    - ``enable_observation_masking: True`` → auto-strip consumed tool results.
    - ``enable_summarization: False`` → no surprise LLM costs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ContextStrategy(str, Enum):
    """Context window management strategies.

    - ``PROGRESSIVE``: Full 4-tier compaction pipeline
      (ObservationMasker -> ToolResultCompactor -> ConversationCompactor -> EmergencyCompactor).
    - ``TRUNCATE_ONLY``: Only truncation, no offloading or summarization.
    - ``NONE``: Disabled entirely — zero overhead.
    """

    PROGRESSIVE = "progressive"
    TRUNCATE_ONLY = "truncate_only"
    NONE = "none"


class SummarySchema(BaseModel):
    """Contract for structured summarization (v0.8.x).

    Defined now so the schema is stable when summarization is enabled.
    A good summary preserves intent, not just words.
    """

    model_config = ConfigDict(frozen=True)

    goals: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    open_items: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    tool_findings: tuple[str, ...] = ()
    safety_conditions: tuple[str, ...] = ()
    confidence: float = 1.0


class ContextConfig(BaseModel):
    """Configuration for context window management.

    Designed to be part of ``AgentConfig``. Immutable after creation.
    """

    model_config = ConfigDict(frozen=True)

    optimal_budget: int = Field(
        default=50_000,
        description=(
            "Quality-optimized context budget in tokens. Compaction triggers "
            "fire against this budget, NOT max_context_tokens. Research shows "
            "effective utilization drops to 60% beyond 100K tokens."
        ),
    )

    max_context_tokens: int | None = Field(
        default=None,
        description=(
            "Hard ceiling — the model's actual context window. "
            "None = auto-detect via BaseLLM.get_context_window(). "
            "Used for emergency compaction only."
        ),
    )

    response_reserve: int = Field(
        default=8192,
        description="Tokens reserved for the model's response. Never allocated to context.",
    )

    tool_result_threshold: int = Field(
        default=20_000,
        description=(
            "Tool results exceeding this token count are candidates for "
            "offloading to ContentStore."
        ),
    )

    compaction_trigger: float = Field(
        default=0.75,
        description=(
            "Start conversation compaction when utilization of optimal_budget "
            "exceeds this. Lowered from 0.85 (Phase 1) to trigger earlier "
            "for quality."
        ),
    )

    tool_compaction_trigger: float = Field(
        default=0.60,
        description=(
            "Start tool-result compaction when utilization of optimal_budget "
            "exceeds this. Lowered from 0.70 (Phase 1) for quality focus."
        ),
    )

    emergency_trigger: float = Field(
        default=0.90,
        description=(
            "Hard emergency compaction when utilization of optimal_budget "
            "exceeds this. Lowered from 0.95 (Phase 1)."
        ),
    )

    strategy: ContextStrategy = Field(
        default=ContextStrategy.PROGRESSIVE,
        description=(
            "PROGRESSIVE: full pipeline (tool -> conversation -> emergency). "
            "TRUNCATE_ONLY: only truncation, no offloading or summarization. "
            "NONE: disabled entirely (zero overhead)."
        ),
    )

    preserve_recent_turns: int = Field(
        default=4,
        description="Always keep last N turn-pairs (user+assistant). Never evicted.",
    )

    enable_offloading: bool = Field(
        default=True,
        description="Store large tool results in ContentStore and replace with preview+reference.",
    )

    enable_observation_masking: bool = Field(
        default=True,
        description=(
            "After each LLM response, replace consumed tool results with slim "
            "markers. Full content stays in ContentStore. This alone solves "
            "~80%% of context rot."
        ),
    )

    enable_summarization: bool = Field(
        default=False,
        description=(
            "LLM-based summarization (costs an extra API call). Off by default "
            "to prevent surprise costs. When enabled, uses SummarySchema contract."
        ),
    )

    cost_per_million_input: float | None = Field(
        default=None,
        description=(
            "Cost per million input tokens in USD. When set, ContextTelemetry "
            "reports estimated dollar savings. Example: 3.0 for Anthropic Sonnet."
        ),
    )

    @staticmethod
    def for_mode(mode: str) -> ContextConfig:
        """Factory with mode-aware defaults.

        Direct mode rarely needs compaction; autonomous mode needs
        aggressive early compaction because tool loops are long.
        """
        if mode == "direct":
            return ContextConfig(
                optimal_budget=50_000,
                tool_compaction_trigger=0.80,
                compaction_trigger=0.90,
                emergency_trigger=0.97,
                preserve_recent_turns=2,
            )
        if mode == "autonomous":
            return ContextConfig(
                optimal_budget=40_000,
                tool_compaction_trigger=0.55,
                compaction_trigger=0.70,
                emergency_trigger=0.90,
                preserve_recent_turns=6,
            )
        # standard (default)
        return ContextConfig()
