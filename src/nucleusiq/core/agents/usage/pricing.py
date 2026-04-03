"""
Cost estimation — per-model pricing tables and ``estimated_cost`` computation.

Extends the v0.5.0 token-origin split with dollar-cost attribution.
Users register model pricing and get cost estimates after ``agent.execute()``.

Usage::

    from nucleusiq.agents.usage.pricing import CostTracker, ModelPricing

    # Built-in pricing is auto-loaded for OpenAI and Gemini models
    tracker = CostTracker()
    cost = tracker.estimate(usage_summary, model="gpt-4o")
    print(cost.display())

    # Custom model pricing
    tracker.register(
        "my-fine-tune",
        ModelPricing(
            prompt_price_per_1k=0.01,
            completion_price_per_1k=0.03,
        ),
    )
"""

from __future__ import annotations

from datetime import date
from typing import Any

from nucleusiq.agents.usage.usage_tracker import (
    UsageSummary,
)
from pydantic import BaseModel, Field

__all__ = [
    "ModelPricing",
    "CostBreakdown",
    "PurposeCost",
    "OriginCost",
    "CostTracker",
]


# ====================================================================== #
# Pricing model                                                            #
# ====================================================================== #


class ModelPricing(BaseModel):
    """Pricing for a single model.

    Prices are per **1,000 tokens** (the industry-standard unit).
    """

    prompt_price_per_1k: float = Field(
        ..., ge=0, description="Cost per 1K prompt/input tokens (USD)"
    )
    completion_price_per_1k: float = Field(
        ..., ge=0, description="Cost per 1K completion/output tokens (USD)"
    )
    effective_date: date | None = Field(
        None, description="When this pricing became effective"
    )


# ====================================================================== #
# Cost result models                                                       #
# ====================================================================== #


class PurposeCost(BaseModel):
    """Cost breakdown for a single call purpose."""

    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    tokens: int = 0
    calls: int = 0


class OriginCost(BaseModel):
    """Cost breakdown for a single token origin (user vs framework)."""

    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    tokens: int = 0
    calls: int = 0


class CostBreakdown(BaseModel):
    """Complete cost breakdown for an execution.

    Contains total cost, per-purpose breakdown (main, planning, tool_loop,
    critic, refiner), and per-origin breakdown (user vs framework).
    """

    model: str = ""
    total_cost: float = 0.0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    by_purpose: dict[str, PurposeCost] = Field(default_factory=dict)
    by_origin: dict[str, OriginCost] = Field(default_factory=dict)
    pricing_available: bool = True

    def summary(self) -> dict[str, Any]:
        """Return a plain dict for serialization."""
        return self.model_dump()

    def display(self) -> str:
        """Return a human-readable cost summary."""
        lines: list[str] = []
        lines.append(f"Cost Estimate ({self.model or 'unknown model'})")
        lines.append("-" * 40)

        if not self.pricing_available:
            lines.append("  (no pricing data available)")
            return "\n".join(lines)

        lines.append(f"  Prompt cost:     ${self.prompt_cost:>10.6f}")
        lines.append(f"  Completion cost: ${self.completion_cost:>10.6f}")
        lines.append(f"  Total cost:      ${self.total_cost:>10.6f}")

        if self.by_purpose:
            lines.append("")
            lines.append("By Purpose")
            for name, pc in self.by_purpose.items():
                lines.append(
                    f"  {name + ':':14s} ${pc.total_cost:>10.6f} "
                    f"({pc.calls} call{'s' if pc.calls != 1 else ''})"
                )

        if self.by_origin:
            lines.append("")
            lines.append("By Origin")
            total = self.total_cost or 1.0
            for name, oc in self.by_origin.items():
                pct = oc.total_cost / total * 100 if total > 0 else 0
                lines.append(
                    f"  {name + ':':14s} ${oc.total_cost:>10.6f} "
                    f"({oc.calls} call{'s' if oc.calls != 1 else ''})  "
                    f"-- {pct:.0f}%"
                )

        return "\n".join(lines)


# ====================================================================== #
# Built-in pricing tables                                                  #
# ====================================================================== #

_BUILTIN_PRICING: dict[str, ModelPricing] = {
    # OpenAI — GPT-4o family
    "gpt-4o": ModelPricing(prompt_price_per_1k=0.0025, completion_price_per_1k=0.01),
    "gpt-4o-2024-08-06": ModelPricing(
        prompt_price_per_1k=0.0025, completion_price_per_1k=0.01
    ),
    "gpt-4o-mini": ModelPricing(
        prompt_price_per_1k=0.00015, completion_price_per_1k=0.0006
    ),
    "gpt-4o-mini-2024-07-18": ModelPricing(
        prompt_price_per_1k=0.00015, completion_price_per_1k=0.0006
    ),
    # OpenAI — GPT-4.1 family
    "gpt-4.1": ModelPricing(prompt_price_per_1k=0.002, completion_price_per_1k=0.008),
    "gpt-4.1-mini": ModelPricing(
        prompt_price_per_1k=0.0004, completion_price_per_1k=0.0016
    ),
    "gpt-4.1-nano": ModelPricing(
        prompt_price_per_1k=0.0001, completion_price_per_1k=0.0004
    ),
    # OpenAI — o-series reasoning
    "o3": ModelPricing(prompt_price_per_1k=0.01, completion_price_per_1k=0.04),
    "o3-mini": ModelPricing(prompt_price_per_1k=0.0011, completion_price_per_1k=0.0044),
    "o4-mini": ModelPricing(prompt_price_per_1k=0.0011, completion_price_per_1k=0.0044),
    "o1": ModelPricing(prompt_price_per_1k=0.015, completion_price_per_1k=0.06),
    "o1-mini": ModelPricing(prompt_price_per_1k=0.003, completion_price_per_1k=0.012),
    # OpenAI — legacy
    "gpt-3.5-turbo": ModelPricing(
        prompt_price_per_1k=0.0005, completion_price_per_1k=0.0015
    ),
    # Google Gemini
    "gemini-2.5-pro": ModelPricing(
        prompt_price_per_1k=0.00125, completion_price_per_1k=0.01
    ),
    "gemini-2.5-flash": ModelPricing(
        prompt_price_per_1k=0.000075, completion_price_per_1k=0.0003
    ),
    "gemini-2.0-flash": ModelPricing(
        prompt_price_per_1k=0.0001, completion_price_per_1k=0.0004
    ),
    "gemini-1.5-pro": ModelPricing(
        prompt_price_per_1k=0.00125, completion_price_per_1k=0.005
    ),
    "gemini-1.5-flash": ModelPricing(
        prompt_price_per_1k=0.000075, completion_price_per_1k=0.0003
    ),
}


# ====================================================================== #
# CostTracker                                                              #
# ====================================================================== #


def _compute_cost(tokens: int, price_per_1k: float) -> float:
    """Compute cost for a token count at the given rate."""
    return (tokens / 1000.0) * price_per_1k


class CostTracker:
    """Estimates dollar cost from token usage and model pricing.

    Holds a registry of ``ModelPricing`` entries (built-in + user-registered).
    Call ``estimate()`` with a ``UsageSummary`` and model name to get a
    ``CostBreakdown``.

    Thread-safe for reads; register should be called during setup.
    """

    def __init__(self) -> None:
        self._pricing: dict[str, ModelPricing] = dict(_BUILTIN_PRICING)

    # ---- Registration ------------------------------------------------ #

    def register(self, model: str, pricing: ModelPricing) -> None:
        """Register or update pricing for a model.

        Parameters
        ----------
        model : str
            Model name (exact match, case-sensitive).
        pricing : ModelPricing
            Pricing data for the model.
        """
        self._pricing[model] = pricing

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Look up pricing for a model, trying exact match then prefix match."""
        if model in self._pricing:
            return self._pricing[model]

        lower = model.lower()
        for key, pricing in self._pricing.items():
            if lower.startswith(key.lower()):
                return pricing

        return None

    @property
    def registered_models(self) -> list[str]:
        """Return all registered model names."""
        return sorted(self._pricing.keys())

    # ---- Estimation -------------------------------------------------- #

    def estimate(self, usage: UsageSummary, *, model: str) -> CostBreakdown:
        """Compute cost breakdown from a ``UsageSummary``.

        Parameters
        ----------
        usage : UsageSummary
            Token usage from ``agent.last_usage`` or ``UsageTracker.summary``.
        model : str
            Model name for pricing lookup.

        Returns
        -------
        CostBreakdown
            Total and per-purpose/origin cost breakdown.
            If no pricing is found, ``pricing_available`` is ``False``.
        """
        pricing = self.get_pricing(model)
        if pricing is None:
            return CostBreakdown(model=model, pricing_available=False)

        prompt_cost = _compute_cost(
            usage.total.prompt_tokens, pricing.prompt_price_per_1k
        )
        completion_cost = _compute_cost(
            usage.total.completion_tokens, pricing.completion_price_per_1k
        )

        by_purpose: dict[str, PurposeCost] = {}
        for name, bucket in usage.by_purpose.items():
            pc = _compute_cost(bucket.prompt_tokens, pricing.prompt_price_per_1k)
            cc = _compute_cost(
                bucket.completion_tokens, pricing.completion_price_per_1k
            )
            by_purpose[name] = PurposeCost(
                prompt_cost=pc,
                completion_cost=cc,
                total_cost=pc + cc,
                tokens=bucket.total_tokens,
                calls=bucket.calls,
            )

        by_origin: dict[str, OriginCost] = {}
        for name, bucket in usage.by_origin.items():
            pc = _compute_cost(bucket.prompt_tokens, pricing.prompt_price_per_1k)
            cc = _compute_cost(
                bucket.completion_tokens, pricing.completion_price_per_1k
            )
            by_origin[name] = OriginCost(
                prompt_cost=pc,
                completion_cost=cc,
                total_cost=pc + cc,
                tokens=bucket.total_tokens,
                calls=bucket.calls,
            )

        return CostBreakdown(
            model=model,
            total_cost=prompt_cost + completion_cost,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            by_purpose=by_purpose,
            by_origin=by_origin,
            pricing_available=True,
        )
