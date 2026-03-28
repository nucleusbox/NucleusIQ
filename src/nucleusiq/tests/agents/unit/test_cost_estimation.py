"""Tests for the cost estimation module.

Covers:
- ModelPricing validation
- CostTracker registration and lookup (exact + prefix)
- Cost estimation from UsageSummary
- Per-purpose and per-origin cost breakdown
- Built-in pricing tables (OpenAI + Gemini)
- Custom pricing registration
- Unknown model handling
- CostBreakdown display formatting
- Edge cases: zero tokens, zero cost, large token counts
"""

from __future__ import annotations

from datetime import date

import pytest
from nucleusiq.agents.components.pricing import (
    _BUILTIN_PRICING,
    CostBreakdown,
    CostTracker,
    ModelPricing,
    _compute_cost,
)
from nucleusiq.agents.components.usage_tracker import (
    BucketStats,
    CallPurpose,
    TokenCount,
    UsageSummary,
    UsageTracker,
)
from pydantic import ValidationError

# ====================================================================== #
# ModelPricing validation                                                  #
# ====================================================================== #


class TestModelPricing:
    """ModelPricing Pydantic validation."""

    def test_valid_pricing(self):
        p = ModelPricing(prompt_price_per_1k=0.01, completion_price_per_1k=0.03)
        assert p.prompt_price_per_1k == 0.01
        assert p.completion_price_per_1k == 0.03
        assert p.effective_date is None

    def test_with_effective_date(self):
        p = ModelPricing(
            prompt_price_per_1k=0.01,
            completion_price_per_1k=0.03,
            effective_date=date(2026, 1, 1),
        )
        assert p.effective_date == date(2026, 1, 1)

    def test_zero_pricing_allowed(self):
        p = ModelPricing(prompt_price_per_1k=0.0, completion_price_per_1k=0.0)
        assert p.prompt_price_per_1k == 0.0

    def test_negative_pricing_rejected(self):
        with pytest.raises(ValidationError):
            ModelPricing(prompt_price_per_1k=-0.01, completion_price_per_1k=0.03)

    def test_negative_completion_rejected(self):
        with pytest.raises(ValidationError):
            ModelPricing(prompt_price_per_1k=0.01, completion_price_per_1k=-0.01)


# ====================================================================== #
# _compute_cost helper                                                     #
# ====================================================================== #


class TestComputeCost:
    """Unit cost calculation."""

    def test_basic_computation(self):
        assert _compute_cost(1000, 0.01) == pytest.approx(0.01)

    def test_fractional_tokens(self):
        assert _compute_cost(500, 0.01) == pytest.approx(0.005)

    def test_zero_tokens(self):
        assert _compute_cost(0, 0.01) == 0.0

    def test_zero_rate(self):
        assert _compute_cost(1000, 0.0) == 0.0

    def test_large_token_count(self):
        cost = _compute_cost(1_000_000, 0.01)
        assert cost == pytest.approx(10.0)


# ====================================================================== #
# CostTracker registration and lookup                                      #
# ====================================================================== #


class TestCostTrackerRegistration:
    """Model registration and pricing lookup."""

    def test_builtin_pricing_loaded(self):
        tracker = CostTracker()
        assert "gpt-4o" in tracker.registered_models
        assert "gemini-2.5-flash" in tracker.registered_models

    def test_exact_match_lookup(self):
        tracker = CostTracker()
        pricing = tracker.get_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.prompt_price_per_1k > 0

    def test_prefix_match_lookup(self):
        tracker = CostTracker()
        pricing = tracker.get_pricing("gpt-4o-2024-11-20-custom")
        assert pricing is not None

    def test_unknown_model_returns_none(self):
        tracker = CostTracker()
        assert tracker.get_pricing("totally-unknown-model-xyz") is None

    def test_custom_registration(self):
        tracker = CostTracker()
        custom = ModelPricing(prompt_price_per_1k=0.05, completion_price_per_1k=0.10)
        tracker.register("my-custom-model", custom)

        pricing = tracker.get_pricing("my-custom-model")
        assert pricing is not None
        assert pricing.prompt_price_per_1k == 0.05

    def test_registration_overrides_existing(self):
        tracker = CostTracker()
        original = tracker.get_pricing("gpt-4o")
        assert original is not None

        updated = ModelPricing(prompt_price_per_1k=0.001, completion_price_per_1k=0.002)
        tracker.register("gpt-4o", updated)

        new = tracker.get_pricing("gpt-4o")
        assert new.prompt_price_per_1k == 0.001

    def test_registered_models_sorted(self):
        tracker = CostTracker()
        models = tracker.registered_models
        assert models == sorted(models)

    def test_gemini_models_in_builtin(self):
        assert "gemini-2.5-flash" in _BUILTIN_PRICING
        assert "gemini-2.5-pro" in _BUILTIN_PRICING
        assert "gemini-2.0-flash" in _BUILTIN_PRICING
        assert "gemini-1.5-pro" in _BUILTIN_PRICING
        assert "gemini-1.5-flash" in _BUILTIN_PRICING

    def test_openai_models_in_builtin(self):
        assert "gpt-4o" in _BUILTIN_PRICING
        assert "gpt-4o-mini" in _BUILTIN_PRICING
        assert "gpt-3.5-turbo" in _BUILTIN_PRICING
        assert "o3" in _BUILTIN_PRICING
        assert "o1" in _BUILTIN_PRICING


# ====================================================================== #
# Cost estimation from UsageSummary                                        #
# ====================================================================== #


def _make_usage(
    prompt: int = 100,
    completion: int = 50,
    by_purpose: dict[str, tuple[int, int, int]] | None = None,
    by_origin: dict[str, tuple[int, int, int]] | None = None,
) -> UsageSummary:
    """Build a UsageSummary for testing.

    ``by_purpose``/``by_origin`` values are ``(prompt, completion, calls)``.
    """
    total = TokenCount(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )

    purposes = {}
    if by_purpose:
        for name, (p, c, n) in by_purpose.items():
            purposes[name] = BucketStats(
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=p + c,
                calls=n,
            )

    origins = {}
    if by_origin:
        for name, (p, c, n) in by_origin.items():
            origins[name] = BucketStats(
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=p + c,
                calls=n,
            )

    return UsageSummary(
        total=total,
        call_count=sum(b.calls for b in purposes.values()) if purposes else 1,
        by_purpose=purposes,
        by_origin=origins,
    )


class TestCostEstimation:
    """Cost estimation from UsageSummary."""

    def test_basic_estimation(self):
        tracker = CostTracker()
        usage = _make_usage(prompt=1000, completion=500)
        result = tracker.estimate(usage, model="gpt-4o")

        assert result.pricing_available is True
        assert result.model == "gpt-4o"
        assert result.prompt_cost > 0
        assert result.completion_cost > 0
        assert result.total_cost == pytest.approx(
            result.prompt_cost + result.completion_cost
        )

    def test_gpt4o_known_pricing(self):
        tracker = CostTracker()
        usage = _make_usage(prompt=1000, completion=1000)
        result = tracker.estimate(usage, model="gpt-4o")

        expected_prompt = 1000 / 1000 * 0.0025
        expected_completion = 1000 / 1000 * 0.01
        assert result.prompt_cost == pytest.approx(expected_prompt)
        assert result.completion_cost == pytest.approx(expected_completion)

    def test_gemini_flash_known_pricing(self):
        tracker = CostTracker()
        usage = _make_usage(prompt=10000, completion=5000)
        result = tracker.estimate(usage, model="gemini-2.5-flash")

        expected_prompt = 10000 / 1000 * 0.000075
        expected_completion = 5000 / 1000 * 0.0003
        assert result.prompt_cost == pytest.approx(expected_prompt)
        assert result.completion_cost == pytest.approx(expected_completion)

    def test_unknown_model_returns_no_pricing(self):
        tracker = CostTracker()
        usage = _make_usage()
        result = tracker.estimate(usage, model="unknown-model-999")

        assert result.pricing_available is False
        assert result.total_cost == 0.0

    def test_zero_tokens_zero_cost(self):
        tracker = CostTracker()
        usage = _make_usage(prompt=0, completion=0)
        result = tracker.estimate(usage, model="gpt-4o")

        assert result.total_cost == 0.0
        assert result.pricing_available is True

    def test_per_purpose_breakdown(self):
        tracker = CostTracker()
        usage = _make_usage(
            prompt=300,
            completion=200,
            by_purpose={
                "main": (100, 100, 1),
                "tool_loop": (200, 100, 2),
            },
        )
        result = tracker.estimate(usage, model="gpt-4o")

        assert "main" in result.by_purpose
        assert "tool_loop" in result.by_purpose
        assert result.by_purpose["main"].calls == 1
        assert result.by_purpose["tool_loop"].calls == 2
        assert result.by_purpose["main"].total_cost > 0
        assert result.by_purpose["tool_loop"].total_cost > 0

    def test_per_origin_breakdown(self):
        tracker = CostTracker()
        usage = _make_usage(
            prompt=300,
            completion=200,
            by_origin={
                "user": (100, 50, 1),
                "framework": (200, 150, 3),
            },
        )
        result = tracker.estimate(usage, model="gpt-4o")

        assert "user" in result.by_origin
        assert "framework" in result.by_origin
        assert result.by_origin["user"].total_cost > 0
        assert (
            result.by_origin["framework"].total_cost
            > result.by_origin["user"].total_cost
        )


# ====================================================================== #
# Integration with UsageTracker                                            #
# ====================================================================== #


class TestCostTrackerWithUsageTracker:
    """CostTracker works with real UsageTracker output."""

    def test_end_to_end_with_usage_tracker(self):
        usage_tracker = UsageTracker()
        usage_tracker.record(
            CallPurpose.MAIN,
            {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
        )
        usage_tracker.record(
            CallPurpose.TOOL_LOOP,
            {"prompt_tokens": 300, "completion_tokens": 100, "total_tokens": 400},
        )

        summary = usage_tracker.summary
        cost_tracker = CostTracker()
        breakdown = cost_tracker.estimate(summary, model="gpt-4o-mini")

        assert breakdown.pricing_available is True
        assert breakdown.total_cost > 0
        assert "main" in breakdown.by_purpose
        assert "tool_loop" in breakdown.by_purpose
        assert "user" in breakdown.by_origin
        assert "framework" in breakdown.by_origin


# ====================================================================== #
# CostBreakdown display                                                    #
# ====================================================================== #


class TestCostBreakdownDisplay:
    """Display formatting."""

    def test_display_includes_model(self):
        tracker = CostTracker()
        usage = _make_usage(prompt=1000, completion=500)
        result = tracker.estimate(usage, model="gpt-4o")

        text = result.display()
        assert "gpt-4o" in text
        assert "$" in text

    def test_display_no_pricing(self):
        breakdown = CostBreakdown(model="unknown", pricing_available=False)
        text = breakdown.display()
        assert "no pricing data" in text

    def test_display_by_purpose(self):
        tracker = CostTracker()
        usage = _make_usage(
            prompt=1000,
            completion=500,
            by_purpose={"main": (500, 250, 1), "tool_loop": (500, 250, 2)},
        )
        result = tracker.estimate(usage, model="gpt-4o")
        text = result.display()
        assert "By Purpose" in text
        assert "main" in text
        assert "tool_loop" in text

    def test_display_by_origin(self):
        tracker = CostTracker()
        usage = _make_usage(
            prompt=1000,
            completion=500,
            by_origin={"user": (500, 250, 1), "framework": (500, 250, 2)},
        )
        result = tracker.estimate(usage, model="gpt-4o")
        text = result.display()
        assert "By Origin" in text
        assert "user" in text
        assert "framework" in text
        assert "%" in text

    def test_summary_returns_dict(self):
        breakdown = CostBreakdown(model="test", total_cost=0.05, pricing_available=True)
        d = breakdown.summary()
        assert isinstance(d, dict)
        assert d["total_cost"] == 0.05
        assert d["model"] == "test"


# ====================================================================== #
# Edge cases                                                               #
# ====================================================================== #


class TestEdgeCases:
    """Unusual or boundary inputs."""

    def test_very_large_token_count(self):
        tracker = CostTracker()
        usage = _make_usage(prompt=10_000_000, completion=5_000_000)
        result = tracker.estimate(usage, model="gpt-4o")

        assert result.total_cost > 0
        assert result.total_cost == result.prompt_cost + result.completion_cost

    def test_custom_pricing_with_estimation(self):
        tracker = CostTracker()
        tracker.register(
            "my-model",
            ModelPricing(prompt_price_per_1k=0.1, completion_price_per_1k=0.2),
        )
        usage = _make_usage(prompt=2000, completion=1000)
        result = tracker.estimate(usage, model="my-model")

        assert result.prompt_cost == pytest.approx(0.2)
        assert result.completion_cost == pytest.approx(0.2)
        assert result.total_cost == pytest.approx(0.4)

    def test_empty_usage_summary(self):
        tracker = CostTracker()
        usage = UsageSummary()
        result = tracker.estimate(usage, model="gpt-4o")

        assert result.total_cost == 0.0
        assert result.pricing_available is True
        assert result.by_purpose == {}
        assert result.by_origin == {}
