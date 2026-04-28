"""ContextTelemetry — observability for context window management.

Exposed in ``AgentResult.context_telemetry`` so users and dashboards
can see exactly what happened to their context during execution.

Phase 2 additions: observation masking counts, optimal budget tracking,
and cost estimation fields for dollar-savings reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompactionEvent(BaseModel):
    """Record of a single compaction operation."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    trigger_utilization: float
    tokens_before: int
    tokens_after: int
    tokens_freed: int
    artifacts_offloaded: int = 0
    duration_ms: float = 0.0


class ContextTelemetry(BaseModel):
    """Context management observability — exposed in AgentResult.

    F3 — separated masker vs compactor accounting
    -----------------------------------------------
    ``tokens_freed_total`` sums both the ``ObservationMasker`` (Tier 0
    post-response) and the ``CompactionPipeline`` (ToolResult /
    Conversation / Emergency).  The combined number hid which mechanism
    was actually doing the work — a run could show
    ``compaction_count = 5`` with ``tokens_freed_total = 800`` while the
    compactor itself freed 0 tokens (the masker did all 800).  The two
    fields below make that separation explicit::

        masker_tokens_freed    = tokens saved by ObservationMasker
        compactor_tokens_freed = tokens saved by CompactionPipeline events
        tokens_freed_total     = masker + compactor (backward-compatible)
    """

    model_config = ConfigDict(frozen=True)

    peak_utilization: float = 0.0
    final_utilization: float = 0.0
    compaction_count: int = 0
    compaction_events: tuple[CompactionEvent, ...] = ()
    tokens_freed_total: int = 0
    compactor_tokens_freed: int = 0  # F3
    masker_tokens_freed: int = 0  # F3
    artifacts_offloaded: int = 0
    region_breakdown: dict[str, int] = Field(default_factory=dict)
    context_limit: int = 0
    response_reserve: int = 0
    warnings: tuple[str, ...] = ()

    # --- Phase 2: observation masking ---
    observations_masked: int = 0
    tokens_masked: int = 0

    # --- Context Mgmt v2 — Step 1: masker gating ---
    # masker_triggered_count = number of post_response calls where masking ran
    # masker_skipped_count   = number of post_response calls gated off because
    #                          util < squeeze_threshold
    # The ratio is the cleanest single signal for "is the gate doing useful
    # work?": skipped >> triggered means short run with the gate saving the
    # quality regression v1 used to cause.
    masker_triggered_count: int = 0
    masker_skipped_count: int = 0

    # --- Phase 2: quality-optimized budget ---
    optimal_budget: int = 0

    # --- Phase 2: cost estimation ---
    estimated_cost_without_mgmt: float = 0.0
    estimated_cost_with_mgmt: float = 0.0
    estimated_savings_pct: float = 0.0

    # --- Context Mgmt v2 — Step 2: recall + policy observability ---
    # ``recall_count``   = how many times ``recall_tool_result`` ran in this
    #                      execution.  High counts on a model + task pair
    #                      mean the masker is too aggressive (or the budget
    #                      too small) — telemetry is the way operators
    #                      notice.
    # ``recall_tokens``  = total tokens *returned* by recall calls (not the
    #                      tokens stored — see ``artifacts_offloaded`` for
    #                      that).  Useful for budgeting: if the model recalls
    #                      30K tokens to answer one question, the masker
    #                      probably should have left them in place.
    # ``policy_breakdown`` and ``policy_source_breakdown`` show how the
    # ``PolicyClassifier`` is partitioning tool results.  An imbalance
    # ("AUTO classifies 95% as EVIDENCE") usually means the size threshold
    # needs tuning, not that the classifier is broken.
    recall_count: int = 0
    recall_tokens: int = 0
    policy_breakdown: dict[str, int] = Field(default_factory=dict)
    policy_source_breakdown: dict[str, int] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Merge support (autonomous sub-agent rollup)                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def merge(
        cls,
        parent: ContextTelemetry | None,
        children: list[ContextTelemetry],
    ) -> ContextTelemetry:
        """Merge parent telemetry with sub-agent telemetries.

        Used by autonomous mode to aggregate metrics from parallel
        sub-agents into the parent agent's result.  Additive for
        counts/costs, max for peak utilization.
        """
        if not children and parent is not None:
            return parent
        if not children:
            return cls()

        p = parent or cls()
        peak = p.peak_utilization
        comp_count = p.compaction_count
        comp_events = list(p.compaction_events)
        freed = p.tokens_freed_total
        compactor_freed = p.compactor_tokens_freed
        masker_freed = p.masker_tokens_freed
        offloaded = p.artifacts_offloaded
        regions: dict[str, int] = dict(p.region_breakdown)
        obs_masked = p.observations_masked
        tok_masked = p.tokens_masked
        masker_trig = p.masker_triggered_count
        masker_skip = p.masker_skipped_count
        cost_without = p.estimated_cost_without_mgmt
        cost_with = p.estimated_cost_with_mgmt
        # v2 Step 2 — additive recall + policy stats.
        recall_count = p.recall_count
        recall_tokens = p.recall_tokens
        policy_breakdown: dict[str, int] = dict(p.policy_breakdown)
        policy_source_breakdown: dict[str, int] = dict(p.policy_source_breakdown)
        warns = list(p.warnings)

        for child in children:
            peak = max(peak, child.peak_utilization)
            comp_count += child.compaction_count
            comp_events.extend(child.compaction_events)
            freed += child.tokens_freed_total
            compactor_freed += child.compactor_tokens_freed
            masker_freed += child.masker_tokens_freed
            offloaded += child.artifacts_offloaded
            obs_masked += child.observations_masked
            tok_masked += child.tokens_masked
            masker_trig += child.masker_triggered_count
            masker_skip += child.masker_skipped_count
            cost_without += child.estimated_cost_without_mgmt
            cost_with += child.estimated_cost_with_mgmt
            recall_count += child.recall_count
            recall_tokens += child.recall_tokens
            for k, v in child.policy_breakdown.items():
                policy_breakdown[k] = policy_breakdown.get(k, 0) + v
            for k, v in child.policy_source_breakdown.items():
                policy_source_breakdown[k] = policy_source_breakdown.get(k, 0) + v
            warns.extend(child.warnings)
            for rgn, tokens in child.region_breakdown.items():
                regions[rgn] = regions.get(rgn, 0) + tokens

        savings_pct = 0.0
        if cost_without > 0:
            savings_pct = (cost_without - cost_with) / cost_without * 100

        return cls(
            peak_utilization=peak,
            final_utilization=p.final_utilization,
            compaction_count=comp_count,
            compaction_events=tuple(comp_events),
            tokens_freed_total=freed,
            compactor_tokens_freed=compactor_freed,
            masker_tokens_freed=masker_freed,
            artifacts_offloaded=offloaded,
            region_breakdown=regions,
            context_limit=p.context_limit,
            response_reserve=p.response_reserve,
            warnings=tuple(warns),
            observations_masked=obs_masked,
            tokens_masked=tok_masked,
            masker_triggered_count=masker_trig,
            masker_skipped_count=masker_skip,
            optimal_budget=p.optimal_budget,
            estimated_cost_without_mgmt=cost_without,
            estimated_cost_with_mgmt=cost_with,
            estimated_savings_pct=savings_pct,
            recall_count=recall_count,
            recall_tokens=recall_tokens,
            policy_breakdown=policy_breakdown,
            policy_source_breakdown=policy_source_breakdown,
        )
