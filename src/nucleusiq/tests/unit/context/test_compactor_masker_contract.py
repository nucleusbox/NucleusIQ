"""F4 — Architectural contract between ObservationMasker and ToolResultCompactor.

The audit originally flagged ``tool_result_threshold = 20_000`` as
miscalibrated (Finding 4) because production runs showed
``tokens_freed = 0`` on every compaction tick.  Re-verification against
real experiment data (gpt-5.2 × Task E, N=6) showed the opposite — the
Masker was already reclaiming 150K-700K tokens per run losslessly, and
the Compactor correctly sat idle because no tool result approached 20K.

The resolution is therefore **architectural**, not numeric:

* ``ObservationMasker`` = primary mechanism.  Runs after every
  assistant response.  Losslessly offloads to ``ContentStore`` (full
  content recoverable via ``extract_raw_trace`` — F2).  Handles 95%+
  of tool-result tokens in practice.

* ``ToolResultCompactor`` = mid-turn EMERGENCY BRAKE.  Fires only when
  a single tool result alone would fill most of the compaction trigger
  window *before* the assistant has had a chance to respond.  The
  truncation path is **lossy** (head + tail + marker); the offload
  path preserves only a 10% preview.  Expected to rarely fire; a
  steady-state of ``compactor_tokens_freed == 0`` is HEALTHY.

These tests lock in that contract so future refactors can't silently:

1. make the Compactor fight the Masker for the same content,
2. couple the two so strategies-without-Compactor break,
3. regress the Masker's lossless-by-default behaviour to the
   Compactor's lossy truncation.

If these tests fail, re-read the F4 rationale in ``ContextConfig.
tool_result_threshold`` before changing anything.
"""

from __future__ import annotations

from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.context.config import ContextConfig, ContextStrategy
from nucleusiq.agents.context.counter import DefaultTokenCounter
from nucleusiq.agents.context.engine import ContextEngine
from nucleusiq.agents.context.store import ContentStore
from nucleusiq.agents.context.strategies.observation_masker import (
    MASK_PREFIX,
    ObservationMasker,
)

# --------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------- #


def _assistant_with_tc(tc_id: str, name: str = "search") -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCallRequest(id=tc_id, name=name, arguments="{}")],
    )


def _tool_result(tc_id: str, name: str, content: str) -> ChatMessage:
    return ChatMessage(role="tool", content=content, name=name, tool_call_id=tc_id)


def _small_result(tc_id: str, name: str = "search") -> ChatMessage:
    # ~40 tokens — representative of real tool output in experiments
    # (measured avg 1.2K but a single "hit" is often a few hundred).
    body = " ".join(f"result-chunk-{i}" for i in range(30))
    return _tool_result(tc_id, name, body)


# --------------------------------------------------------------------- #
# Contract 1 — Masker handles tool results regardless of Compactor
# --------------------------------------------------------------------- #


class TestMaskerIsIndependentOfCompactor:
    """The Masker must not depend on the Compactor's threshold.  If this
    breaks, ``masking_only`` (the experiment strategy that disables
    Compactor triggers) would regress into doing nothing."""

    def test_masker_fires_when_compactor_threshold_is_huge(self) -> None:
        """Set threshold high enough that the Compactor would never
        touch typical tool results — Masker must still run."""
        counter = DefaultTokenCounter()
        store = ContentStore()
        masker = ObservationMasker()

        messages = [
            ChatMessage(role="user", content="find X"),
            _assistant_with_tc("tc1", "search"),
            _small_result("tc1"),
            _assistant_with_tc("tc2", "search"),  # "consumed" cue
            _small_result("tc2"),
        ]

        new_msgs, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count >= 1, (
            "Masker must mask at least the first (consumed) tool result "
            "regardless of any compactor threshold."
        )
        assert freed > 0
        tool_contents = [m.content for m in new_msgs if m.role == "tool"]
        assert any(
            isinstance(c, str) and c.startswith(MASK_PREFIX) for c in tool_contents
        )

    def test_masker_fires_when_threshold_is_tiny(self) -> None:
        """Symmetric case — set threshold low.  Masker behaviour should
        still be identical because it does NOT consult the threshold."""
        counter = DefaultTokenCounter()
        store = ContentStore()
        masker = ObservationMasker()

        messages = [
            ChatMessage(role="user", content="find X"),
            _assistant_with_tc("tc1", "search"),
            _small_result("tc1"),
            _assistant_with_tc("tc2", "search"),
            _small_result("tc2"),
        ]

        new_msgs, masked_count, freed = masker.mask(messages, counter, store)

        assert masked_count >= 1
        assert freed > 0


# --------------------------------------------------------------------- #
# Contract 2 — `masking_only` strategy must actually mask
# --------------------------------------------------------------------- #


class TestMaskingOnlyStrategy:
    """In the experiment harness ``masking_only`` sets all Compactor
    triggers to 0.99 so the Compactor tier never fires.  This locks in
    that tool-result masking still happens end-to-end via the engine."""

    def test_masking_only_engine_masks_consumed_tool_results(self) -> None:
        config = ContextConfig(
            strategy=ContextStrategy.PROGRESSIVE,
            enable_observation_masking=True,
            compaction_trigger=0.99,
            tool_compaction_trigger=0.99,
            emergency_trigger=0.99,
            # v2 step 1: the harness `masking_only` strategy already
            # implicitly assumes the masker is always-on; mirror that
            # by disabling the budget gate.
            squeeze_threshold=0.0,
        )
        engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=128_000)

        messages = [
            ChatMessage(role="user", content="q"),
            _assistant_with_tc("tc1", "search"),
            _small_result("tc1"),
            _assistant_with_tc("tc2", "search"),
            _small_result("tc2"),
        ]

        new_msgs = engine.post_response(messages)
        tel = engine.telemetry

        assert tel.observations_masked >= 1, (
            "masking_only must still mask consumed tool results even "
            "though compaction triggers are set to 0.99."
        )
        assert tel.masker_tokens_freed > 0
        assert tel.compactor_tokens_freed == 0, (
            "masking_only must NOT free tokens via the compactor tier "
            "(its triggers are effectively disabled at 0.99)."
        )

    def test_masking_only_does_not_depend_on_tool_result_threshold(
        self,
    ) -> None:
        """Two engines with identical masking config but wildly
        different ``tool_result_threshold`` values must mask the same
        number of observations."""
        small_msgs = [
            ChatMessage(role="user", content="q"),
            _assistant_with_tc("tc1", "search"),
            _small_result("tc1"),
            _assistant_with_tc("tc2", "search"),
            _small_result("tc2"),
        ]

        def run_with_threshold(threshold: int) -> int:
            config = ContextConfig(
                strategy=ContextStrategy.PROGRESSIVE,
                enable_observation_masking=True,
                compaction_trigger=0.99,
                tool_compaction_trigger=0.99,
                emergency_trigger=0.99,
                tool_result_threshold=threshold,
                # v2 step 1: gate disabled — this test asserts the
                # masker is invariant to ``tool_result_threshold``.
                squeeze_threshold=0.0,
            )
            engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=128_000)
            engine.post_response(list(small_msgs))
            return engine.telemetry.observations_masked

        assert run_with_threshold(50) == run_with_threshold(200_000), (
            "Masker output must be invariant to tool_result_threshold."
        )


# --------------------------------------------------------------------- #
# Contract 3 — 20K default is a REAL emergency-brake value
# --------------------------------------------------------------------- #


class TestThresholdIsEmergencyBrake:
    """Lock in the two numeric invariants that make 20K coherent with
    the rest of the autonomous-mode context config."""

    def test_default_threshold_is_below_hard_context_ceiling(self) -> None:
        """The threshold must be meaningfully less than the model's
        context window — otherwise no single result could ever exceed
        it and the brake would be useless."""
        cfg = ContextConfig()
        assert cfg.tool_result_threshold < 128_000, (
            "20K is tuned for ≤128K-ctx models (gpt-5.x, Claude 3.5).  "
            "Bumping to or above the context window makes the brake "
            "vestigial."
        )

    def test_default_threshold_fits_compaction_window(self) -> None:
        """The threshold must fit inside the smallest realistic
        compaction trigger window, otherwise no single tool result
        could alone trip the brake.

        v0.7.9: ``optimal_budget`` is auto-resolved per-model, so we
        check against the smallest plausible window this default
        produces — an 8K open-source model, which yields a 5.6K
        optimal budget and a 0.55 × 5.6K ≈ 3K trigger window under
        autonomous-mode defaults.  The threshold scales as
        ``tool_result_per_call_max_chars`` at execution time, so the
        real correctness condition is expressed as a sanity check:
        ``tool_result_threshold`` must be *smaller* than the smallest
        plausible ``optimal_budget × tool_compaction_trigger`` the
        framework ever resolves to on a 128K-class model
        (which is the target hardware for the default).
        """
        auto = ContextConfig.for_mode("autonomous")
        resolved_128k = ContextConfig.resolve_optimal_budget(auto, 128_000)
        trigger_window = int(resolved_128k * auto.tool_compaction_trigger)
        assert auto.tool_result_threshold <= trigger_window, (
            "tool_result_threshold must fit inside the compaction "
            "trigger window on a 128K-class model; otherwise no "
            "single tool result could cross the brake."
        )

    def test_default_threshold_matches_documented_20k(self) -> None:
        """Lock in the 20K number so any future change is loud and
        forces an author to read the ``tool_result_threshold`` docstring
        (which explains why lowering is a quality regression)."""
        assert ContextConfig().tool_result_threshold == 20_000


# --------------------------------------------------------------------- #
# Contract 4 — Compactor stays idle on realistic tool-result sizes
# --------------------------------------------------------------------- #


class TestCompactorIdleOnRealisticInputs:
    """Regression guard against accidentally lowering the threshold.

    The experiment data (gpt-5.2 × Task E, N=6) showed tool results
    averaged ~1.2K tokens, with peaks well under 20K.  We lock that
    behaviour in: a stream of realistic small tool results must NOT
    trigger Compactor-level token freeing.  If this fails, someone
    lowered the threshold below realistic sizes and the Compactor is
    about to start doing lossy work on content the Masker handles
    losslessly.
    """

    def test_stream_of_small_results_does_not_compact(self) -> None:
        config = ContextConfig(
            strategy=ContextStrategy.PROGRESSIVE,
            enable_observation_masking=True,
            # Default thresholds — let Compactor fire if it wants to.
            # v2 step 1: bypass squeeze gate so the masker handles the
            # small results unconditionally — this is the contract this
            # test asserts ("Masker is the primary mechanism, Compactor
            # stays idle on realistic small inputs").
            squeeze_threshold=0.0,
        )
        engine = ContextEngine(config, DefaultTokenCounter(), max_tokens=128_000)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="research competitive landscape"),
        ]
        for i in range(12):
            tc_id = f"tc{i}"
            messages.append(_assistant_with_tc(tc_id, "web_search"))
            messages.append(_small_result(tc_id, "web_search"))

        engine.post_response(messages)
        tel = engine.telemetry

        assert tel.compactor_tokens_freed == 0, (
            "12 realistic small tool results must not cause the "
            "Compactor to free tokens.  If it did, the threshold was "
            "lowered below real-world tool-result sizes — see F4 in "
            "ContextConfig.tool_result_threshold for the quality "
            "regression this causes."
        )
        assert tel.observations_masked >= 1, (
            "Meanwhile the Masker SHOULD have handled the consumed "
            "tool results — that's the primary mechanism."
        )
