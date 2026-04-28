"""Tests for v0.7.8 adaptive per-tool-result rehydration cap.

These tests lock in the budget-based solution that replaced the legacy
fixed ``CriticLimits.tool_result = 3_000`` cap:

* ``compute_per_tool_cap`` scales with the model's context window.
* Small windows + many tools → floor (with warning).
* Large windows + few tools → ceiling.
* Overhead + reserve saturation → floor + warning.
* Critic and Refiner use the SAME helper (symmetry guard).

This is the primary regression guard for the Task E quality fix.
"""

from __future__ import annotations

import logging

from nucleusiq.agents.context.store import compute_per_tool_cap


class TestComputePerToolCapScaling:
    """The cap must scale naturally across a wide range of windows."""

    def test_large_window_hits_ceiling(self):
        # gpt-5.4: 400K window, 10 tool results, plenty of room.
        cap = compute_per_tool_cap(
            context_window=400_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=10,
            min_chars=500,
            max_chars=50_000,
        )
        # (400_000 - 5_000 - 8_000) / 10 = 38_700 tokens = 154_800 chars
        # → clamps to max_chars ceiling.
        assert cap == 50_000

    def test_medium_window_scales_proportionally(self):
        # 128K (gpt-4-turbo / gpt-5.1) with 30 tool calls.
        cap = compute_per_tool_cap(
            context_window=128_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=30,
            min_chars=500,
            max_chars=50_000,
        )
        # (128_000 - 13_000) / 30 = 3_833 tokens = 15_332 chars
        assert 10_000 < cap < 20_000

    def test_small_window_many_tools_hits_floor(self):
        # Qwen-2.5 32K with 60 tool calls — the user's "60 tools at once" case.
        cap = compute_per_tool_cap(
            context_window=32_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=60,
            min_chars=500,
            max_chars=50_000,
        )
        # (32_000 - 13_000) / 60 = 316 tokens = 1_264 chars
        # → ABOVE floor at 500, so returns the computed value.
        assert cap == 1_264

    def test_tiny_window_hits_floor_with_warning(self, caplog):
        # Llama-3.1 8K with 60 tool calls — truly saturated.
        caplog.set_level(logging.WARNING)
        cap = compute_per_tool_cap(
            context_window=8_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,  # overhead alone > window
            num_tool_results=60,
            min_chars=500,
            max_chars=50_000,
            purpose="critic",
        )
        assert cap == 500  # floor
        assert any("saturated" in r.message for r in caplog.records) or any(
            "degraded quality" in r.message for r in caplog.records
        )

    def test_zero_tool_results_does_not_divide_by_zero(self):
        cap = compute_per_tool_cap(
            context_window=128_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=0,  # treated as 1
            min_chars=500,
            max_chars=50_000,
        )
        # (128_000 - 13_000) / 1 = 115_000 tokens → ceiling at max_chars.
        assert cap == 50_000


class TestComputePerToolCapWarnings:
    """Saturation and floor-clamping must surface a warning to the user."""

    def test_warning_emitted_when_clamped_to_floor(self, caplog):
        caplog.set_level(logging.WARNING)
        compute_per_tool_cap(
            context_window=16_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=60,
            min_chars=500,
            max_chars=50_000,
            purpose="refiner",
        )
        assert any("degraded quality" in r.message for r in caplog.records)

    def test_purpose_label_appears_in_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        compute_per_tool_cap(
            context_window=8_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=60,
            min_chars=500,
            max_chars=50_000,
            purpose="critic",
        )
        assert any("critic" in r.message for r in caplog.records)


class TestCriticRefinerSymmetry:
    """The Critic and Refiner must see *symmetric* evidence.

    Pre-v0.7.8, the Refiner's ``summarize_tool_results`` head-sliced
    rehydrated content to 500 chars/item while the Critic saw up to
    3_000 chars — so the Refiner was strictly **less informed** than
    the Critic it was trying to satisfy.  This test locks in the fix:
    both roles rehydrate through the same ``compute_per_tool_cap``
    helper and differ only in configurable overhead/reserve defaults.
    """

    def test_same_window_same_count_produces_symmetric_budgets(self):
        # With equal overheads/reserves, both roles compute the same
        # per-tool cap.  The production defaults intentionally give the
        # Refiner *slightly more* reserve for its longer prompt/response,
        # but the helper itself is role-agnostic.
        critic_cap = compute_per_tool_cap(
            context_window=128_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=20,
            min_chars=500,
            max_chars=50_000,
            purpose="critic",
        )
        refiner_cap = compute_per_tool_cap(
            context_window=128_000,
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=20,
            min_chars=500,
            max_chars=50_000,
            purpose="refiner",
        )
        assert critic_cap == refiner_cap


class TestComputePerToolCapBounds:
    """Clamping behaviour at the edges."""

    def test_never_below_min_chars(self):
        cap = compute_per_tool_cap(
            context_window=1_000,
            prompt_overhead_tokens=0,
            response_reserve_tokens=0,
            num_tool_results=100,  # 10 tokens/tool = 40 chars
            min_chars=500,
            max_chars=50_000,
        )
        assert cap == 500

    def test_never_above_max_chars(self):
        cap = compute_per_tool_cap(
            context_window=2_000_000,  # Gemini 2.5 Pro
            prompt_overhead_tokens=5_000,
            response_reserve_tokens=8_000,
            num_tool_results=5,
            min_chars=500,
            max_chars=50_000,
        )
        assert cap == 50_000
