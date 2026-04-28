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
    - ``optimal_budget: None`` (v0.7.9) → auto-resolved at engine
      construction as ``min(optimal_budget_fraction × ctx_window,
      optimal_budget_ceiling)`` where ``ctx_window`` comes from
      ``BaseLLM.get_context_window()``.  This scales naturally from 8K
      open-source models (5.6K effective budget) to 2M Gemini (capped
      at the 120K quality plateau).  Set an explicit integer to
      override.
    - ``max_context_tokens: None`` → auto-detect from provider via
      ``BaseLLM.get_context_window()``.  Used as hard ceiling.
    - ``strategy: ContextStrategy.NONE`` → complete opt-out with zero overhead.
    - ``enable_observation_masking: True`` → auto-strip consumed tool results.
    - ``enable_summarization: False`` → no surprise LLM costs.

v0.7.9 — Adaptive optimal_budget (provider-agnostic fix)
--------------------------------------------------------
Before v0.7.9, ``optimal_budget`` was hard-coded to 50_000 tokens, so
compaction triggers fired against 50K regardless of whether the model
had an 8K or 200K context window.  On a 128K model the Emergency
compactor fired at 45K — ~35% of real capacity — causing unnecessary
and catastrophic context reduction for tool-heavy tasks.

Evidence: gpt-5.2 × Task E × full_progressive consistently hit
``peak_utilization=1.0`` (saturating the 50K budget) and triggered
EmergencyCompactor events that nuked context from 84K → 75 tokens,
even though the model's real window had 44K headroom remaining.

The auto-resolved default fixes this without any per-model tuning.
"""

from __future__ import annotations

from enum import Enum

from nucleusiq.agents.context.policy import (
    DEFAULT_EPHEMERAL_NAME_PATTERNS,
    DEFAULT_EPHEMERAL_SIZE_THRESHOLD,
    DEFAULT_EVIDENCE_NAME_PATTERNS,
)
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

    optimal_budget: int | None = Field(
        default=None,
        description=(
            "Quality-optimized context budget in tokens.  Compaction "
            "triggers fire against this budget, NOT max_context_tokens.\n\n"
            "When ``None`` (the default), the budget is auto-resolved at "
            "``ContextEngine`` construction time from the model's actual "
            "context window::\n\n"
            "    resolved = min(\n"
            "        optimal_budget_fraction * ctx_window,\n"
            "        optimal_budget_ceiling,\n"
            "    )\n\n"
            "This scales naturally across providers: 8K Llama → 5.6K, "
            "128K gpt-5.* → 89.6K, 2M Gemini → 120K (capped).  Without "
            "this, a fixed 50K default caused EmergencyCompactor to "
            "fire at 35% of a 128K model's real capacity — catastrophic "
            "for tool-heavy autonomous tasks.\n\n"
            "Set an explicit integer to override (e.g. 30_000 for a "
            "very tight research budget, or 100_000 for maximum recall "
            "on a 128K model).  Research backing: effective attention "
            "utilization drops to ~60% beyond 100K tokens (Kuratov et "
            "al., 2024; Liu et al., 2024), which informs the 120K "
            "ceiling."
        ),
    )

    optimal_budget_fraction: float = Field(
        default=0.70,
        description=(
            "When ``optimal_budget`` is ``None``, use this fraction of the "
            "model's context window as the working budget.  Leaves a "
            "30% headroom for incremental tool results, retries, and "
            "mid-turn overhead before the emergency trigger would fire."
        ),
    )

    optimal_budget_ceiling: int = Field(
        default=120_000,
        description=(
            "Upper bound on the auto-resolved ``optimal_budget``.  "
            "Attention utilization research (Kuratov et al., 2024; Liu "
            "et al., 2024) shows that effective recall drops sharply "
            "beyond ~100K tokens regardless of advertised context "
            "window.  We cap at 120K as a conservative plateau.  Set "
            "``optimal_budget`` explicitly to bypass."
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
            "Per-message size gate for the ``ToolResultCompactor`` tier — "
            "a MID-TURN EMERGENCY BRAKE, not the primary mechanism.\n\n"
            "A single tool result whose content exceeds this many tokens "
            "will be preemptively offloaded (or truncated) before the "
            "assistant replies, so one pathological tool call cannot "
            "single-handedly blow the context window in one step.\n\n"
            "Under normal operation this fires rarely.  The PRIMARY "
            "mechanism for reclaiming tool-result tokens is the "
            "``ObservationMasker`` (F1), which runs after every "
            "assistant response, losslessly offloads full content to "
            "``ContentStore``, and replaces consumed tool messages with "
            "a structured marker that Critic/Refiner can rehydrate "
            "(F2).  Empirically (gpt-5.2 × Task E, N=6 runs) the "
            "Masker reclaims 150K-700K tokens/run while the Compactor "
            "correctly frees 0 — no pathological result appeared.\n\n"
            "``compactor_tokens_freed == 0`` in ``ContextTelemetry`` is "
            "therefore the HEALTHY steady state.  A non-zero value "
            "means a truly oversized tool result was caught mid-turn "
            "before the Masker got a chance to handle it losslessly.\n\n"
            "Rationale for 20K default: at ``optimal_budget=40K`` "
            "(autonomous mode) the ``tool_compaction_trigger`` of 0.60 "
            "opens a 24K compaction window; 20K says 'any single result "
            "that alone fills nearly the entire trigger window is too "
            "big to let through unchecked'.  Lowering this without care "
            "replaces the Masker's lossless offloading with the "
            "Compactor's lossy truncation for the same content — a net "
            "quality regression."
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

    squeeze_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Budget gate for the ObservationMasker (Context Mgmt v2 — Step 1).\n\n"
            "Masking only runs when the current utilization of "
            "``optimal_budget`` is at or above this threshold.  Below the "
            "threshold the masker is skipped entirely and tool results are "
            "left untouched in the conversation history.\n\n"
            "Why a gate?  Empirically (gpt-5.2 × Task E, N=4 full_progressive "
            "runs) the always-on masker degraded the Generator's view of the "
            "task on short and medium runs that never approached the budget "
            "ceiling.  Running it unconditionally cost output quality "
            "(words 1556 vs 3289 baseline, quality 15.6 vs 72.3) without "
            "providing a token-saving benefit, because the runs in question "
            "had plenty of headroom.\n\n"
            "v0.7.10 history — bumped 0.70 → 0.85 then reverted:\n"
            "  - The 0.85 default was tried to 'mask only under genuine\n"
            "    pressure', but on tool-heavy tasks (Task E with PDF\n"
            "    excerpts ~50K tokens each) it caused context to inflate\n"
            "    past the model's hard limit (272K) before masking ever\n"
            "    fired — the model never reached synthesis (output 42\n"
            "    words vs 3038 at 0.70).\n"
            "  - Reverted to 0.70 because bounded marker previews (Step 4)\n"
            "    + idempotent-tool dedup already reduce the re-fetch loops\n"
            "    that motivated the bump.  Previews are capped at ~300 chars\n"
            "    and only attached to large results, so markers do not become\n"
            "    a second prompt-sized evidence store.\n\n"
            "Defaults:\n"
            "  - 0.70 — current default; mask early enough to avoid hard\n"
            "    context overflow, relying on bounded markers and recall for\n"
            "    full evidence recovery.\n"
            "  - 0.85 — preserved as an option for callers on large\n"
            "    contexts (≥1M tokens) where late masking is acceptable.\n"
            "  - 0.0 — disable the gate; behaves like v1 (always on).  Useful\n"
            "    for unit tests that isolate the masker's mechanics.\n"
            "  - 1.0 — disable masking entirely (gate never opens).\n\n"
            "This field is honoured by ``ContextEngine.post_response()`` "
            "and is independent of ``enable_observation_masking``: setting "
            "the latter to ``False`` disables masking outright, while this "
            "threshold tunes *when* masking activates given that it is "
            "enabled."
        ),
    )

    enable_summarization: bool = Field(
        default=False,
        description=(
            "LLM-based summarization (costs an extra API call). Off by default "
            "to prevent surprise costs. When enabled, uses SummarySchema contract."
        ),
    )

    # ------------------------------------------------------------------ #
    # Context Mgmt v2 — Step 2: tool-result policy classifier knobs       #
    # ------------------------------------------------------------------ #
    # The classifier (see ``context.policy.PolicyClassifier``) decides
    # whether each tool result is EVIDENCE or EPHEMERAL when no
    # explicit declaration is given (``@tool(context_policy=...)``).
    # The defaults below cover the common cases — override per-app if
    # you have a domain-specific tool inventory.

    evidence_name_patterns: tuple[str, ...] = Field(
        default=DEFAULT_EVIDENCE_NAME_PATTERNS,
        description=(
            "Case-insensitive substrings that mark a tool's output as "
            "EVIDENCE (preserve under pressure, recallable via "
            "``recall_tool_result``).  Match wins over the size gate.\n\n"
            "Defaults cover read/search/fetch/query/retrieve plus a few "
            "domain hints (e.g. ``annual_report``, ``_excerpt``, "
            "``pdf_``).  Add fragments specific to your tool inventory "
            "if the heuristic misclassifies.\n\n"
            "Examples:\n"
            '  evidence_name_patterns=("read_", "sql_", "vector_")'
        ),
    )

    ephemeral_name_patterns: tuple[str, ...] = Field(
        default=DEFAULT_EPHEMERAL_NAME_PATTERNS,
        description=(
            "Case-insensitive substrings that mark a tool's output as "
            "EPHEMERAL (drop under pressure, not recallable).  "
            "Defaults cover time/date utilities, formatters, "
            "validators, and predicates (``is_*``, ``has_*``, "
            "``check_*``).  Add fragments for any tool whose output "
            "is small and trivially recomputable."
        ),
    )

    ephemeral_size_threshold: int = Field(
        default=DEFAULT_EPHEMERAL_SIZE_THRESHOLD,
        ge=0,
        description=(
            "Token-count gate for the EPHEMERAL classification (rule "
            "3 of the heuristic).  An undecorated, un-name-matched "
            "tool result with fewer than this many tokens is treated "
            "as EPHEMERAL.  Default 500 ≈ 2000 chars of English — "
            "small enough that re-running the tool is cheaper than "
            "persisting the bytes through a recall round-trip."
        ),
    )

    hot_set_lookback_turns: int = Field(
        default=3,
        ge=0,
        description=(
            "How many recent turns to consider 'hot' for the recall "
            "tracker.  Any evidence ref recalled within the last N "
            "turns is pinned by the Compactor — re-evicting content "
            "the model just asked for would be a waste.  Step 3 "
            "consumes this; Step 2 only records the data."
        ),
    )

    cost_per_million_input: float | None = Field(
        default=None,
        description=(
            "Cost per million input tokens in USD. When set, ContextTelemetry "
            "reports estimated dollar savings. Example: 3.0 for Anthropic Sonnet."
        ),
    )

    # ------------------------------------------------------------------ #
    # v0.7.8 — Adaptive (budget-based) rehydration caps                   #
    # ------------------------------------------------------------------ #
    # Replaces the legacy fixed ``CriticLimits.tool_result`` cap (3K/5K
    # chars).  At Critic/Refiner call time the framework computes a
    # **per-call** char cap for rehydrated tool results based on:
    #
    #     per_tool_chars = clamp(
    #         (ctx_window - prompt_overhead - response_reserve)
    #         / num_tool_results_in_trace,
    #         min = tool_result_per_call_min_chars,
    #         max = tool_result_per_call_max_chars,
    #     ) * 4   # tokens → chars (≈ 4 chars / token for English)
    #
    # This scales naturally across providers (8K Llama → 2M Gemini)
    # without any per-model tuning.  For small windows it degrades
    # gracefully (hits the min floor and emits a log warning); for
    # large windows it provides full evidence (hits the max ceiling).
    # The ceiling exists purely as a safety rail — a single 400K-char
    # PDF excerpt in a 400K-ctx model would otherwise be unbounded.

    critic_prompt_overhead_tokens: int = Field(
        default=5_000,
        description=(
            "Tokens reserved for the Critic's prompt framing (system "
            "instructions + task + verification rubric) when computing "
            "the per-tool rehydration cap.  Increase if you add a "
            "larger Critic rubric; decrease for short prompts."
        ),
    )

    critic_response_reserve_tokens: int = Field(
        default=8_000,
        description=(
            "Tokens reserved for the Critic's response (JSON verdict + "
            "feedback) when computing the per-tool rehydration cap."
        ),
    )

    refiner_prompt_overhead_tokens: int = Field(
        default=8_000,
        description=(
            "Tokens reserved for the Refiner's prompt framing (system + "
            "task + prior candidate + critique) when computing the "
            "per-tool rehydration cap.  Larger than the Critic's "
            "overhead because the Refiner prompt includes the full "
            "prior candidate."
        ),
    )

    refiner_response_reserve_tokens: int = Field(
        default=16_000,
        description=(
            "Tokens reserved for the Refiner's response (the full "
            "revised answer) when computing the per-tool rehydration "
            "cap.  Larger than the Critic's reserve because the "
            "Refiner writes the complete corrected output, not just a "
            "verdict."
        ),
    )

    tool_result_per_call_max_chars: int = Field(
        default=50_000,
        description=(
            "Absolute ceiling on rehydrated tool-result content per "
            "call, in characters.  Safety rail — a single pathological "
            "tool result (a 400K-char PDF excerpt in a 400K-ctx model) "
            "cannot alone consume an entire Critic/Refiner prompt."
        ),
    )

    tool_result_per_call_min_chars: int = Field(
        default=500,
        description=(
            "Absolute floor on rehydrated tool-result content per "
            "call, in characters.  Below this the content is too small "
            "to be useful; the framework emits a warning that the "
            "model's context window is saturated.  Most open-source "
            "32K models evaluating 60+ tool calls land on this floor."
        ),
    )

    @staticmethod
    def for_mode(mode: str) -> ContextConfig:
        """Factory with mode-aware defaults.

        Direct mode rarely needs compaction; autonomous mode needs
        aggressive early compaction because tool loops are long.

        v0.7.9: no longer hard-codes ``optimal_budget`` — the value
        auto-resolves from each model's real context window at engine
        construction time.  Only the *trigger fractions* and
        ``preserve_recent_turns`` differ between modes.
        """
        if mode == "direct":
            return ContextConfig(
                tool_compaction_trigger=0.80,
                compaction_trigger=0.90,
                emergency_trigger=0.97,
                preserve_recent_turns=2,
            )
        if mode == "autonomous":
            return ContextConfig(
                tool_compaction_trigger=0.55,
                compaction_trigger=0.70,
                emergency_trigger=0.90,
                preserve_recent_turns=6,
            )
        # standard (default)
        return ContextConfig()

    @staticmethod
    def resolve_optimal_budget(
        config: ContextConfig,
        context_window: int,
    ) -> int:
        """Resolve the effective ``optimal_budget`` for a given model.

        If the user has set ``optimal_budget`` explicitly, return it
        unchanged — respect user intent even if it's too small for the
        model (the ledger's ``min(optimal, resolved_max)`` clamp will
        keep it sane).

        Otherwise, auto-resolve as ``min(fraction × ctx_window,
        ceiling)``.  For a 128K model with default settings this
        yields 89_600; for an 8K Llama it yields 5_600; for a 2M
        Gemini it caps at the 120_000 quality plateau.

        This helper is the single source of truth for the resolution
        rule, called by ``ContextEngine.__init__`` and by tests.  It
        has no side effects and is cheap (pure integer math).
        """
        if config.optimal_budget is not None:
            return config.optimal_budget
        fraction_budget = int(config.optimal_budget_fraction * context_window)
        return max(1, min(fraction_budget, config.optimal_budget_ceiling))
