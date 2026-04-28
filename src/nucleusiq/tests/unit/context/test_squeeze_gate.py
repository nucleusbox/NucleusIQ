"""Tests for Context Mgmt v2 — Step 1 (budget-gated ObservationMasker).

The gate ensures the masker only runs when ``util >= squeeze_threshold``.
Below the threshold the masker is a strict no-op and ``masker_skipped_count``
ticks up.  At or above the threshold the masker runs as before and
``masker_triggered_count`` ticks up.

Test groups
-----------
1. TestGateBelowThreshold     — masking skipped, telemetry tracks skips
2. TestGateAboveThreshold     — masking runs, telemetry tracks triggers
3. TestGateDisabled           — squeeze_threshold=0.0 reproduces v1 behaviour
4. TestGateExtremes           — squeeze_threshold=1.0 disables masking
5. TestGateWithMaskerOff      — gate is a no-op when masker is disabled
6. TestGateInteractionWithRecount — the gate's recount does not corrupt ledger
"""

from __future__ import annotations

import pytest
from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.context.engine import ContextEngine


def _msg(
    role: str,
    content: str,
    *,
    name: str | None = None,
    tool_call_id: str | None = None,
):
    return ChatMessage(role=role, content=content, name=name, tool_call_id=tool_call_id)


# ---------------------------------------------------------------------- #
# 1. Gate skips masking when utilization is below threshold              #
# ---------------------------------------------------------------------- #


class TestGateBelowThreshold:
    def test_no_masking_when_util_far_below_threshold(self):
        """A small message list at default 0.70 threshold leaves tool results untouched."""
        config = ContextConfig(
            optimal_budget=100_000,
            squeeze_threshold=0.70,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        original = "tool result content " * 25  # ~125 tokens
        messages = [
            _msg("system", "You are helpful."),
            _msg("tool", original, name="search", tool_call_id="tc1"),
            _msg("assistant", "Done."),
        ]
        result = engine.post_response(messages)

        assert result[1].content == original, (
            "Tool result must be unchanged when util is below squeeze_threshold"
        )

    def test_skipped_count_increments_when_below(self):
        config = ContextConfig(
            optimal_budget=100_000,
            squeeze_threshold=0.70,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("tool", "x" * 200, name="search", tool_call_id="tc1"),
            _msg("assistant", "ok"),
        ]
        engine.post_response(messages)

        tel = engine.telemetry
        assert tel.masker_skipped_count == 1
        assert tel.masker_triggered_count == 0
        assert tel.observations_masked == 0
        assert tel.tokens_masked == 0

    def test_repeated_skips_accumulate(self):
        config = ContextConfig(
            optimal_budget=100_000,
            squeeze_threshold=0.70,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        msgs = [
            _msg("tool", "y" * 200, name="search", tool_call_id="tc1"),
            _msg("assistant", "ok"),
        ]
        for _ in range(5):
            engine.post_response(msgs)

        assert engine.telemetry.masker_skipped_count == 5


# ---------------------------------------------------------------------- #
# 2. Gate triggers masking when utilization >= threshold                  #
# ---------------------------------------------------------------------- #


class TestGateAboveThreshold:
    def test_masking_runs_when_util_above_threshold(self):
        """A tight budget pushes util above threshold and engages the masker."""
        # ``optimal_budget`` minus ``response_reserve`` = effective limit.
        # Pick numbers so a single sizeable tool result takes us > 70%.
        config = ContextConfig(
            optimal_budget=2_000,
            response_reserve=200,
            squeeze_threshold=0.70,
        )
        engine = ContextEngine(config, max_tokens=200_000)

        messages = [
            _msg("system", "You are helpful."),
            _msg("tool", "X" * 6_000, name="search", tool_call_id="tc1"),
            _msg("assistant", "Here are the results."),
        ]
        result = engine.post_response(messages)

        assert "[observation consumed" in result[1].content, (
            "Tool result must be masked once util reaches squeeze_threshold"
        )

    def test_triggered_count_increments_when_above(self):
        config = ContextConfig(
            optimal_budget=2_000,
            response_reserve=200,
            squeeze_threshold=0.70,
        )
        engine = ContextEngine(config, max_tokens=200_000)

        messages = [
            _msg("tool", "Z" * 6_000, name="search", tool_call_id="tc1"),
            _msg("assistant", "ok"),
        ]
        engine.post_response(messages)

        tel = engine.telemetry
        assert tel.masker_triggered_count == 1
        assert tel.masker_skipped_count == 0
        assert tel.observations_masked >= 1
        assert tel.tokens_masked > 0

    def test_content_preserved_in_store_when_gated_through(self):
        config = ContextConfig(
            optimal_budget=2_000,
            response_reserve=200,
            squeeze_threshold=0.70,
        )
        engine = ContextEngine(config, max_tokens=200_000)
        original = "PAYLOAD " * 1_000

        messages = [
            _msg("tool", original, name="big", tool_call_id="tc1"),
            _msg("assistant", "Processed."),
        ]
        engine.post_response(messages)

        assert engine.store.size >= 1
        keys = engine.store.keys()
        assert engine.store.retrieve(keys[0]) == original


# ---------------------------------------------------------------------- #
# 3. squeeze_threshold == 0.0 reproduces v1 always-on behaviour           #
# ---------------------------------------------------------------------- #


class TestGateDisabled:
    def test_zero_threshold_always_runs_masker(self):
        config = ContextConfig(
            optimal_budget=100_000,
            squeeze_threshold=0.0,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("tool", "A" * 500, name="search", tool_call_id="tc1"),
            _msg("assistant", "ok"),
        ]
        result = engine.post_response(messages)

        assert "[observation consumed" in result[0].content
        tel = engine.telemetry
        assert tel.masker_triggered_count == 1
        assert tel.masker_skipped_count == 0

    def test_zero_threshold_with_low_util_still_masks(self):
        """Even at near-zero util, threshold=0.0 lets the masker run."""
        config = ContextConfig(
            optimal_budget=1_000_000,
            squeeze_threshold=0.0,
        )
        engine = ContextEngine(config, max_tokens=2_000_000)

        messages = [
            _msg("tool", "B" * 800, name="search", tool_call_id="tc1"),
            _msg("assistant", "ok"),
        ]
        result = engine.post_response(messages)

        assert "[observation consumed" in result[0].content


# ---------------------------------------------------------------------- #
# 4. squeeze_threshold == 1.0 disables masking entirely                   #
# ---------------------------------------------------------------------- #


class TestGateExtremes:
    def test_threshold_one_never_masks(self):
        """util can only equal 1.0 in pathological cases — gate stays closed."""
        config = ContextConfig(
            optimal_budget=100_000,
            squeeze_threshold=1.0,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        original = "C" * 1_000
        messages = [
            _msg("tool", original, name="search", tool_call_id="tc1"),
            _msg("assistant", "ok"),
        ]
        result = engine.post_response(messages)

        assert result[0].content == original
        assert engine.telemetry.masker_triggered_count == 0


# ---------------------------------------------------------------------- #
# 5. enable_observation_masking=False short-circuits the gate             #
# ---------------------------------------------------------------------- #


class TestGateWithMaskerOff:
    def test_gate_irrelevant_when_masker_disabled(self):
        """Even at a forced high-util, masker stays off when explicitly disabled."""
        config = ContextConfig(
            optimal_budget=2_000,
            response_reserve=200,
            enable_observation_masking=False,
            squeeze_threshold=0.0,
        )
        engine = ContextEngine(config, max_tokens=200_000)

        original = "D" * 6_000
        messages = [
            _msg("tool", original, name="search", tool_call_id="tc1"),
            _msg("assistant", "ok"),
        ]
        result = engine.post_response(messages)

        assert result[0].content == original
        tel = engine.telemetry
        assert tel.masker_triggered_count == 0
        assert tel.masker_skipped_count == 0


# ---------------------------------------------------------------------- #
# 6. The recount the gate performs does not corrupt the ledger             #
# ---------------------------------------------------------------------- #


class TestGateInteractionWithRecount:
    @pytest.mark.asyncio
    async def test_gate_recount_consistent_with_prepare(self):
        """post_response recount must produce the same allocated total as prepare."""
        config = ContextConfig(
            optimal_budget=100_000,
            squeeze_threshold=0.70,
        )
        engine = ContextEngine(config, max_tokens=128_000)

        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "Analyze this data carefully."),
            _msg("tool", "x" * 1_000, name="search", tool_call_id="tc1"),
            _msg("assistant", "Initial response."),
        ]
        await engine.prepare(messages)
        allocated_after_prepare = engine.budget.allocated

        engine.post_response(messages)
        allocated_after_gate = engine.budget.allocated

        assert allocated_after_gate == allocated_after_prepare, (
            "Gate's recount must not mutate the ledger differently from prepare's recount"
        )
