"""Usage tracking and cost estimation — public API.

This package owns all token-accounting and pricing logic:

* **UsageTracker** — accumulates token records across an execution.
* **CostTracker** — estimates dollar cost from token usage + model pricing.

Import from here::

    from nucleusiq.agents.usage import UsageTracker, CostTracker
"""

from __future__ import annotations

from nucleusiq.agents.usage.pricing import (
    CostBreakdown,
    CostTracker,
    ModelPricing,
    OriginCost,
    PurposeCost,
)
from nucleusiq.agents.usage.usage_tracker import (
    BucketStats,
    CallPurpose,
    TokenCount,
    TokenOrigin,
    UsageRecord,
    UsageSummary,
    UsageTracker,
)

__all__ = [
    # Usage
    "CallPurpose",
    "TokenOrigin",
    "TokenCount",
    "BucketStats",
    "UsageSummary",
    "UsageRecord",
    "UsageTracker",
    # Pricing
    "ModelPricing",
    "CostBreakdown",
    "PurposeCost",
    "OriginCost",
    "CostTracker",
]
