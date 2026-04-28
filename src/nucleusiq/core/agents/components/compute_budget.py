"""F3 — Inference-time compute scaling (``ComputeBudget``).

Translates Aletheia's claim *"compute spent = difficulty signal"* into a
bounded, signal-driven controller.  Default on, not a user-facing knob.

The controller makes three decisions per attempt:

    1. Should the loop stop (accept / abstain) or continue?
    2. Should the next attempt get more budget than the previous one?
    3. What escalation reason (if any) should be recorded for telemetry?

All numbers are framework-level constants living on this module so tests
and future tuning work touches exactly one file.  No per-user knobs, no
provider-specific logic.

Design principles applied (see ``docs/design/AUTONOMOUS_MODE_ALETHEIA_ALIGNMENT.md``):

* **SRP** — this module owns *only* budget + escalation math.  It does
  not know how to call an LLM, format a prompt, or parse a verdict.
* **DIP** — ``SimpleRunner`` / ``ComplexRunner`` depend on
  ``ComputeBudget.from_config`` (a plain factory) and on
  ``decide_next_action`` (a pure function).  Tests substitute fakes by
  constructing ``ComputeBudget`` instances directly.
* **OCP** — new escalation reasons can be added to ``EscalationReason``
  and handled in ``decide_next_action`` without touching the runners.
* **Immutability** — ``ComputeBudget`` is a frozen Pydantic model.  The
  ``escalate`` and ``record_tokens`` methods return *new* copies so
  callers cannot accidentally race on shared state.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from nucleusiq.agents.components.critic import CritiqueResult
    from nucleusiq.agents.config.agent_config import AgentConfig

# ------------------------------------------------------------------ #
# Framework constants — tunable in one place, tested for invariants.  #
# ------------------------------------------------------------------ #

#: Maximum number of budget escalations allowed in a single run.
#: Aletheia's own data (Fig. 5) shows diminishing returns beyond ~2.5×
#: baseline on their hardest problems; two escalations at 1.5× reach
#: 2.25× which matches that empirical knee.
MAX_ESCALATIONS_PER_RUN: int = 2

#: Hard ceiling on ``max_output_tokens`` for *any* single LLM call,
#: enforced by ``escalate`` so even a pathological escalation chain
#: cannot run away with context.
HARD_TOKEN_CEILING: int = 16_384

#: Default ceiling on **cumulative tokens** (input + output + tool
#: results) across *all* LLM calls in a single run.  Sized so that a
#: realistic autonomous loop — primary agent with tools, Critic,
#: Refiner, and possibly two escalations — stays well under this
#: while a pathological runaway still trips it.
#:
#: Why a flat number instead of a multiple of ``llm_max_output_tokens``?
#: ``llm_max_output_tokens`` is a *per-call output* cap (default 2048).
#: Cumulative tokens include input prompts, tool results, and every
#: call's usage.  For any tool-using task the input dwarfs the output,
#: so multiplying the per-call output cap is a category error — it
#: produced an 8K ceiling in the old design, which was exhausted
#: before the first attempt finished.  A flat total-run budget is the
#: right mental model: "this run may consume at most N tokens".
#:
#: Override per-run via ``ComputeBudget.from_config(cfg,
#: total_run_token_budget=...)``.
DEFAULT_RUN_TOKEN_CEILING: int = 500_000

#: Per-escalation multipliers.  Conservative on tokens (1.5× rather
#: than 2×) so two escalations land at 2.25× of baseline per call
#: instead of 4×.  Tool-call cap grows more gently since most tasks
#: don't need more tool calls, just bigger ones.
TOKEN_ESCALATION_FACTOR: float = 1.5
TOOL_CALL_ESCALATION_FACTOR: float = 1.25

#: How close an UNCERTAIN verdict must be to the accept threshold before
#: we consider it a "one-more-nudge" candidate rather than a stall.
#: Empirically matches the old ``_UNCERTAIN_ACCEPT_THRESHOLD`` band.
UNCERTAIN_CLOSE_DELTA: float = 0.10

#: Minimum absolute score delta considered real progress between two
#: successive Critic verdicts.  Larger than typical Critic-score noise
#: (~0.02 across re-runs on the same candidate), so we trust it.
IMPROVEMENT_DELTA: float = 0.05


# ------------------------------------------------------------------ #
# Types                                                                #
# ------------------------------------------------------------------ #

#: Machine-readable reason for a budget escalation.  Surfaced via
#: telemetry (``EscalationRecord.reason``) so the research harness can
#: correlate compute spend with problem difficulty.
EscalationReason = Literal["uncertain_close", "stuck"]


#: Machine-readable reason for the loop refusing to continue despite
#: having nominal escalation slots remaining.  Surfaced via
#: ``AbstentionSignal.reason`` prefix so the harness can distinguish
#: "ran out of retries" from "refused to spend more compute because it
#: would be futile".  Purely informational today; kept as a literal so
#: the set of reasons is closed and typed.
AbstainReason = Literal[
    "budget_exhausted",  # classical path: ran out of retries
    "stuck_after_escalation",  # F3.1: delta < IMPROVEMENT_DELTA on the
    # attempt that *followed* an escalation.
    # Spending a second escalation would be
    # futile (Aletheia: "compute spent ==
    # difficulty signal").
]


class Action(str, Enum):
    """What the autonomous runner should do after a Critic verdict."""

    #: Continue with another Generator/Refiner pass.  The caller must
    #: consult ``EscalationDecision.budget`` for the (possibly updated)
    #: budget to use for the next attempt.
    RETRY = "retry"

    #: Current candidate passed.  The runner returns the candidate.
    STOP_ACCEPT = "stop_accept"

    #: No acceptable candidate under the budget.  The runner raises
    #: ``AbstentionSignal`` (F2) carrying the best candidate so far.
    STOP_ABSTAIN = "stop_abstain"


class ComputeBudget(BaseModel):
    """Immutable snapshot of the compute budget for a run.

    Grown via ``escalate`` — never mutated in place — so concurrent
    parallel attempts (F4) can hold independent budgets without aliasing.
    """

    model_config = ConfigDict(frozen=True)

    #: Maximum number of attempts (= Generator / Refiner passes).
    #: Escalation grows this value so the outer loop will iterate again.
    max_retries: int = Field(gt=0)

    #: Per-LLM-call output-token cap handed to the provider.
    max_output_tokens: int = Field(gt=0)

    #: Per-run tool-call ceiling used by the primary tool loop.
    max_tool_calls: int = Field(gt=0)

    #: Initial ``max_output_tokens`` at the start of the run.  Kept for
    #: telemetry (so we can report "started at X, escalated to Y").
    initial_token_budget: int = Field(gt=0)

    #: Hard cap on **cumulative tokens** (input + output + tool results)
    #: across every LLM call in this run.  Independent of
    #: ``max_output_tokens`` — see ``DEFAULT_RUN_TOKEN_CEILING`` for
    #: the design rationale.
    total_run_token_budget: int = Field(gt=0)

    #: How many times ``escalate`` has been called so far.
    escalations_used: int = Field(default=0, ge=0)

    #: Running sum of tokens consumed by LLM calls in this run (primary
    #: agent + Critic + Refiner).  Fed in by the runner from the
    #: ``UsageTracker`` — see ``record_tokens`` below.
    cumulative_tokens_spent: int = Field(default=0, ge=0)

    # --------------------------------------------------------------- #
    # Constructors                                                     #
    # --------------------------------------------------------------- #

    @classmethod
    def from_config(
        cls,
        cfg: AgentConfig,
        *,
        total_run_token_budget: int | None = None,
    ) -> ComputeBudget:
        """Build the initial budget from ``AgentConfig``.

        Reads ``max_retries``, ``llm_max_output_tokens``, and the
        effective ``max_tool_calls`` (respecting execution-mode caps).

        Parameters
        ----------
        cfg:
            The agent's config.
        total_run_token_budget:
            Optional override for the per-run cumulative token ceiling.
            When ``None`` we use ``DEFAULT_RUN_TOKEN_CEILING``.  Runners
            that know the LLM's context window can pass
            ``context_window * N`` for a model-aware ceiling; the
            default is a single flat number that covers realistic
            autonomous loops (up to ~5 attempts with tool-heavy tasks
            on 128K-context models).
        """
        tokens = int(getattr(cfg, "llm_max_output_tokens", 2048) or 2048)
        retries = max(1, int(getattr(cfg, "max_retries", 3) or 3))
        if hasattr(cfg, "get_effective_max_tool_calls"):
            tool_calls = int(cfg.get_effective_max_tool_calls())
        else:
            tool_calls = int(getattr(cfg, "max_tool_calls", 80) or 80)
        total_budget = int(
            total_run_token_budget
            if total_run_token_budget is not None
            else DEFAULT_RUN_TOKEN_CEILING
        )
        return cls(
            max_retries=retries,
            max_output_tokens=tokens,
            max_tool_calls=tool_calls,
            initial_token_budget=tokens,
            total_run_token_budget=total_budget,
        )

    # --------------------------------------------------------------- #
    # Queries                                                          #
    # --------------------------------------------------------------- #

    def can_escalate(self) -> bool:
        """True iff another escalation is allowed under both caps.

        Two independent guards:

        * ``escalations_used < MAX_ESCALATIONS_PER_RUN`` — hard cap on
          how many times the budget can grow in a single run.
        * ``cumulative_tokens_spent < total_run_token_budget`` — hard
          cap on total tokens the run may consume.  See
          ``DEFAULT_RUN_TOKEN_CEILING`` for why this is a flat number
          rather than a multiple of ``initial_token_budget``.
        """
        if self.escalations_used >= MAX_ESCALATIONS_PER_RUN:
            return False
        return self.cumulative_tokens_spent < self.total_run_token_budget

    def cumulative_cap(self) -> int:
        """Hard ceiling on cumulative tokens across all LLM calls."""
        return self.total_run_token_budget

    # --------------------------------------------------------------- #
    # Transitions (all return new budgets)                             #
    # --------------------------------------------------------------- #

    def escalate(self, _reason: EscalationReason) -> ComputeBudget:
        """Return a grown budget.

        The ``reason`` argument is purely for telemetry traceability —
        the math is the same for every reason.  We scale retries +1,
        tokens by ``TOKEN_ESCALATION_FACTOR`` (capped at
        ``HARD_TOKEN_CEILING``), and tool calls by
        ``TOOL_CALL_ESCALATION_FACTOR``.

        Raises ``RuntimeError`` if called when ``can_escalate()`` is
        False — callers are expected to guard with ``can_escalate`` so
        that escalation failures are loud, not silent.
        """
        if not self.can_escalate():
            raise RuntimeError(
                "ComputeBudget.escalate called with no escalations remaining "
                f"(used={self.escalations_used}, "
                f"cumulative={self.cumulative_tokens_spent}, "
                f"cap={self.cumulative_cap()})"
            )
        new_tokens = min(
            int(self.max_output_tokens * TOKEN_ESCALATION_FACTOR),
            HARD_TOKEN_CEILING,
        )
        new_tool_calls = max(
            self.max_tool_calls,
            int(self.max_tool_calls * TOOL_CALL_ESCALATION_FACTOR),
        )
        return self.model_copy(
            update={
                "max_retries": self.max_retries + 1,
                "max_output_tokens": new_tokens,
                "max_tool_calls": new_tool_calls,
                "escalations_used": self.escalations_used + 1,
            }
        )

    def record_tokens(self, tokens: int) -> ComputeBudget:
        """Return a new budget with ``cumulative_tokens_spent`` bumped.

        Tokens may be ``0`` (the LLM call had no usage reported); in
        that case the method is a no-op but still returns a copy so
        callers can use a single assignment idiom.
        """
        if tokens <= 0:
            return self
        return self.model_copy(
            update={
                "cumulative_tokens_spent": self.cumulative_tokens_spent + tokens,
            }
        )


# ------------------------------------------------------------------ #
# Escalation decision — pure function, trivially testable              #
# ------------------------------------------------------------------ #


class EscalationDecision(BaseModel):
    """Outcome of a single ``decide_next_action`` call."""

    model_config = ConfigDict(frozen=True)

    action: Action

    #: The budget to use for the *next* attempt.  When ``action ==
    #: RETRY`` this may be the same instance (no escalation) or a grown
    #: one (``escalate`` was called).  When action is terminal, the
    #: budget is unchanged.
    budget: ComputeBudget

    #: Populated iff an escalation actually happened.  Surfaces to
    #: telemetry as ``EscalationRecord``.
    escalation_reason: EscalationReason | None = None

    #: F5 — Populated iff ``action == STOP_ABSTAIN``.  Distinguishes
    #: "ran out of retries" (``"budget_exhausted"``) from "refused to
    #: spend more compute because it would be futile"
    #: (``"stuck_after_escalation"``).  Surfaced through
    #: ``AbstentionSignal.abstain_reason`` so downstream callers can
    #: react programmatically (e.g. a smarter model vs. hand off to a
    #: human) instead of string-matching free-form feedback.
    abstain_reason: AbstainReason | None = None


def decide_next_action(
    critique: CritiqueResult,
    critique_history: list[CritiqueResult],
    budget: ComputeBudget,
    attempt: int,
    uncertain_accept_threshold: float,
) -> EscalationDecision:
    """Decide what to do after a Critic verdict.

    Pure function — no I/O, no state mutation.  The runner is expected
    to:

        * append ``critique`` to ``critique_history`` **before** calling
          this (so ``critique_history[-1] is critique``);
        * read ``decision.budget`` for the next attempt's limits;
        * emit an ``EscalationRecord`` when ``escalation_reason`` is
          non-None.

    Parameters
    ----------
    critique:
        The verdict just produced for the latest candidate.
    critique_history:
        All verdicts produced in this run *so far* (including
        ``critique``).  Used to compute the ``delta`` improvement signal.
    budget:
        Current budget.  Never mutated.
    attempt:
        Zero-based index of the attempt that just finished.  Used to
        detect "budget exhausted" (``attempt >= budget.max_retries - 1``).
    uncertain_accept_threshold:
        The threshold at which an UNCERTAIN verdict is accepted.  Kept
        as a parameter so runners with different thresholds (currently
        only 0.7) need not tell this module about their constants.

    Returns
    -------
    EscalationDecision
    """
    from nucleusiq.agents.components.critic import Verdict

    verdict = critique.verdict
    score = critique.score

    # Terminal: PASS always accepts.
    if verdict == Verdict.PASS:
        return EscalationDecision(action=Action.STOP_ACCEPT, budget=budget)

    # Terminal: UNCERTAIN above threshold accepts.
    if verdict == Verdict.UNCERTAIN and score >= uncertain_accept_threshold:
        return EscalationDecision(action=Action.STOP_ACCEPT, budget=budget)

    # Compute improvement delta vs previous verdict (if any).
    prev_score = critique_history[-2].score if len(critique_history) >= 2 else 0.0
    delta = score - prev_score

    budget_exhausted = attempt >= budget.max_retries - 1

    # Case 1: UNCERTAIN, close to the threshold — "one more nudge"
    # with a larger budget often pushes us over. Eligible even on what
    # *would* be the last attempt, because escalating grows retries.
    if verdict == Verdict.UNCERTAIN and score >= (
        uncertain_accept_threshold - UNCERTAIN_CLOSE_DELTA
    ):
        if budget.can_escalate():
            return EscalationDecision(
                action=Action.RETRY,
                budget=budget.escalate("uncertain_close"),
                escalation_reason="uncertain_close",
            )
        if budget_exhausted:
            return EscalationDecision(
                action=Action.STOP_ABSTAIN,
                budget=budget,
                abstain_reason="budget_exhausted",
            )
        return EscalationDecision(action=Action.RETRY, budget=budget)

    # Case 2: FAIL.  Two sub-cases.
    if verdict == Verdict.FAIL:
        # (a) Revision is making progress — keep going at the current
        # budget.  No escalation needed.
        if delta >= IMPROVEMENT_DELTA:
            if budget_exhausted:
                # Progress is real but the budget is gone; abstain
                # rather than silently accepting a still-failing answer.
                return EscalationDecision(
                    action=Action.STOP_ABSTAIN,
                    budget=budget,
                    abstain_reason="budget_exhausted",
                )
            return EscalationDecision(action=Action.RETRY, budget=budget)

        # (b) Stuck — no meaningful delta over >=1 prior verdict.
        # Escalate once if we can; otherwise abstain.
        if len(critique_history) >= 2:
            # F3.1 — "futile escalation" guard.
            #
            # If we already spent an escalation AND the very next
            # attempt's Critic score is still not moving (delta <
            # IMPROVEMENT_DELTA), granting a second escalation is
            # almost always wasted compute.  The bug this guards
            # against was observed in production:
            #
            #     Attempt N   [ESCALATE]: stuck → tokens X→HARD_CAP
            #     Attempt N+1 [CRITIC]:   score unchanged (delta=+0.00)
            #     Attempt N+1 [ESCALATE]: stuck → tokens HARD_CAP→HARD_CAP
            #     Attempt N+2 [REFINE]:   answer got *shorter*, same score
            #     Attempt N+2 [ABSTAIN]:  (50 min elapsed, zero progress)
            #
            # Aletheia's own formulation — "compute spent correlates
            # with task difficulty" — says that when you grant extra
            # compute and still see no improvement, the task is beyond
            # the model's reach; spending *more* compute is actively
            # harmful (wastes tokens, sometimes degrades the answer).
            # Abstain now and surface the best candidate we have.
            if budget.escalations_used >= 1:
                return EscalationDecision(
                    action=Action.STOP_ABSTAIN,
                    budget=budget,
                    abstain_reason="stuck_after_escalation",
                )
            if budget.can_escalate():
                return EscalationDecision(
                    action=Action.RETRY,
                    budget=budget.escalate("stuck"),
                    escalation_reason="stuck",
                )
            return EscalationDecision(
                action=Action.STOP_ABSTAIN,
                budget=budget,
                abstain_reason="budget_exhausted",
            )

        # First FAIL of the run — retry at baseline budget.
        if budget_exhausted:
            return EscalationDecision(
                action=Action.STOP_ABSTAIN,
                budget=budget,
                abstain_reason="budget_exhausted",
            )
        return EscalationDecision(action=Action.RETRY, budget=budget)

    # Case 3: UNCERTAIN well below threshold (not "close") — treat like
    # FAIL for control-flow purposes.
    if budget_exhausted:
        return EscalationDecision(
            action=Action.STOP_ABSTAIN,
            budget=budget,
            abstain_reason="budget_exhausted",
        )
    return EscalationDecision(action=Action.RETRY, budget=budget)


__all__ = [
    "AbstainReason",
    "Action",
    "ComputeBudget",
    "DEFAULT_RUN_TOKEN_CEILING",
    "EscalationDecision",
    "EscalationReason",
    "HARD_TOKEN_CEILING",
    "IMPROVEMENT_DELTA",
    "MAX_ESCALATIONS_PER_RUN",
    "TOKEN_ESCALATION_FACTOR",
    "TOOL_CALL_ESCALATION_FACTOR",
    "UNCERTAIN_CLOSE_DELTA",
    "decide_next_action",
]
