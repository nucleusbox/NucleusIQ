"""
F4 — Best-of-N parallel attempts.

Wraps the simple / complex runner paths so that, when
``AgentConfig.n_parallel_attempts > 1``, the autonomous loop runs
``N`` independent Generator → Verifier → Reviser attempts and the
``Critic`` selects the winner.

Design notes
------------
* **Zero overhead when ``N == 1``**: the mode short-circuits this
  runner entirely (see ``AutonomousMode.run``); this file only runs
  for ``N >= 2``.
* **Sequential execution today**: each attempt mutates shared agent
  state (``_tracer`` autonomous detail, ``_last_messages``, plugin
  state, ``AgentConfig.llm_params.temperature``).  True parallelism
  via ``asyncio.gather`` would need per-attempt tracer isolation,
  which lands with PI-2 (Agent-as-Tool) once full ``Agent`` cloning
  is available.  Sequential still delivers the Best-of-N quality
  lever (different seeds → independent candidates) at the documented
  ``O(N × baseline)`` cost — see ``docs/design/
  AUTONOMOUS_MODE_ALETHEIA_ALIGNMENT.md`` §F4.5 / §F4.6.
* **LLM call accumulation is intentional**: ``llm_calls`` /
  ``tool_calls`` on the tracer accumulate across all N attempts so
  cost reporting reflects *total* work (matches Aletheia's "compute
  spent" accounting).  Only ``_autonomous_detail`` is snapshotted
  per attempt and reset between attempts, so each ``AutonomousDetail``
  in ``parallel_attempts`` describes exactly one attempt.
* **SOLID**: ``ParallelRunner`` depends on an *inner runner factory*
  (``SimpleRunner`` / ``ComplexRunner``) and on pure helpers
  (``_selection_rule``, ``_perturbed_temperature``).  It adds no
  LLM call sites of its own.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from nucleusiq.agents.agent_result import (
    AbstentionSignal,
    AutonomousDetail,
    CritiqueSnapshot,
)
from nucleusiq.agents.components.critic import CritiqueResult, Verdict
from nucleusiq.agents.task import Task
from nucleusiq.streaming.events import StreamEvent

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent


# F4.1 — hard cap mirrors ``AgentConfig.n_parallel_attempts`` validation.
MAX_PARALLEL_ATTEMPTS: int = 5

# F4.4 — seed-diversity deltas applied to the configured base temperature.
# Capped at 1.0 so providers that forbid temperatures above that ceiling
# (e.g. strict JSON-mode clients) keep working.  The list is indexed by
# attempt number; attempt ``i >= len(list)`` falls back to the last entry.
_TEMPERATURE_DELTAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)

# F4.3 — accept threshold for UNCERTAIN verdicts during selection.  Must
# match the one used by SimpleRunner / ComplexRunner so a candidate that
# would be *accepted* by a single-attempt run is never *discarded* by the
# selection rule.
_UNCERTAIN_ACCEPT_THRESHOLD: float = 0.7


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #


class ParallelRunner:
    """Run N independent Generator→Verifier→Reviser attempts and pick one.

    Parameters
    ----------
    n:
        Number of attempts to run.  Must be ``>= 2``; ``N == 1`` is
        handled by the caller short-circuiting to the inner runner.
    run_one_sync:
        Async callable that runs a single attempt to completion and
        returns the final candidate (or raises ``AbstentionSignal``).
    run_one_stream:
        Async generator callable for the streaming path.  Yields
        ``StreamEvent`` instances; final content is captured from the
        terminal ``StreamEventType.COMPLETE`` event.  May be ``None``
        if the caller does not need streaming.
    """

    def __init__(
        self,
        *,
        n: int,
        run_one_sync: Callable[[], Awaitable[Any]],
        run_one_stream: (
            Callable[[], AsyncGenerator[StreamEvent, None]] | None
        ) = None,
    ) -> None:
        if n < 2:
            raise ValueError(
                f"ParallelRunner requires n >= 2 (got {n}); "
                "the caller must short-circuit N == 1 runs."
            )
        if n > MAX_PARALLEL_ATTEMPTS:
            raise ValueError(
                f"n_parallel_attempts must be <= {MAX_PARALLEL_ATTEMPTS} "
                f"(got {n})"
            )
        self._n = n
        self._run_one_sync = run_one_sync
        self._run_one_stream = run_one_stream

    # ------------------------------------------------------------------ #
    # Sync entrypoint                                                     #
    # ------------------------------------------------------------------ #

    async def run_sync(self, agent: "Agent") -> Any:
        attempts = await self._collect_sync_attempts(agent)
        return self._finalize(agent, attempts)

    # ------------------------------------------------------------------ #
    # Streaming entrypoint                                                #
    # ------------------------------------------------------------------ #

    async def run_stream(
        self, agent: "Agent", task: Task
    ) -> AsyncGenerator[StreamEvent, None]:
        if self._run_one_stream is None:
            raise RuntimeError(
                "ParallelRunner.run_stream requires a ``run_one_stream`` "
                "factory (none was provided at construction)."
            )

        yield StreamEvent.thinking_event(
            f"Best-of-{self._n}: running {self._n} independent attempts "
            f"with temperature perturbation…"
        )

        attempts: list[_AttemptOutcome] = []
        tracer = getattr(agent, "_tracer", None)
        temp_snapshot = _snapshot_temperature(agent)

        try:
            for i in range(self._n):
                _apply_temperature(agent, temp_snapshot, attempt_index=i)
                _reset_autonomous_detail(tracer)
                yield StreamEvent.thinking_event(
                    f"Best-of-{self._n}: attempt {i + 1}/{self._n} "
                    f"(temperature delta={_TEMPERATURE_DELTAS[min(i, len(_TEMPERATURE_DELTAS) - 1)]})"
                )
                last_content: Any = None
                abstained = False
                abstention: AbstentionSignal | None = None
                try:
                    async for event in self._run_one_stream():  # type: ignore[misc]
                        if event.type.name == "COMPLETE":
                            last_content = event.content
                        yield event
                except AbstentionSignal as sig:
                    abstained = True
                    abstention = sig
                    last_content = sig.best_candidate
                except Exception as e:
                    agent._logger.error(
                        "Best-of-N attempt %d failed with %s", i + 1, e
                    )
                    yield StreamEvent.error_event(str(e))
                    continue

                detail_snapshot = _build_detail(tracer)
                critique = (
                    abstention.critique
                    if abstention is not None
                    else _extract_last_critique(detail_snapshot)
                )
                attempts.append(
                    _AttemptOutcome(
                        index=i,
                        result=last_content,
                        critique=critique,
                        abstained=abstained,
                        abstention_reason=(
                            abstention.reason if abstention else None
                        ),
                        detail=detail_snapshot,
                    )
                )
        finally:
            _restore_temperature(agent, temp_snapshot)

        # Selection + final telemetry for streaming path.
        try:
            self._finalize(agent, attempts, raise_on_abstain=False)
        except AbstentionSignal as sig:
            yield StreamEvent.error_event(
                f"Best-of-{self._n} abstained: {sig.reason}"
            )
            raise

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    async def _collect_sync_attempts(
        self, agent: "Agent"
    ) -> list["_AttemptOutcome"]:
        attempts: list[_AttemptOutcome] = []
        tracer = getattr(agent, "_tracer", None)
        temp_snapshot = _snapshot_temperature(agent)

        try:
            for i in range(self._n):
                _apply_temperature(agent, temp_snapshot, attempt_index=i)
                _reset_autonomous_detail(tracer)

                agent._logger.info(
                    "Best-of-%d: attempt %d/%d (temperature delta=%s)",
                    self._n,
                    i + 1,
                    self._n,
                    _TEMPERATURE_DELTAS[min(i, len(_TEMPERATURE_DELTAS) - 1)],
                )

                result: Any = None
                abstained = False
                abstention: AbstentionSignal | None = None
                try:
                    result = await self._run_one_sync()
                except AbstentionSignal as sig:
                    abstained = True
                    abstention = sig
                    result = sig.best_candidate
                except Exception as e:
                    agent._logger.error(
                        "Best-of-N attempt %d failed: %s", i + 1, e
                    )
                    # Do not append — this attempt produced no candidate;
                    # selection rule will skip over it.  Other attempts
                    # continue.
                    continue

                detail_snapshot = _build_detail(tracer)
                critique = (
                    abstention.critique
                    if abstention is not None
                    else _extract_last_critique(detail_snapshot)
                )
                attempts.append(
                    _AttemptOutcome(
                        index=i,
                        result=result,
                        critique=critique,
                        abstained=abstained,
                        abstention_reason=(
                            abstention.reason if abstention else None
                        ),
                        detail=detail_snapshot,
                    )
                )
        finally:
            _restore_temperature(agent, temp_snapshot)

        return attempts

    def _finalize(
        self,
        agent: "Agent",
        attempts: list["_AttemptOutcome"],
        *,
        raise_on_abstain: bool = True,
    ) -> Any:
        if not attempts:
            # All attempts errored out.  Raise an abstention signal with
            # a synthesised reason so ``Agent.execute`` can still build
            # an ``AgentResult`` with ``status=ABSTAINED``.
            synth = CritiqueResult(
                verdict=Verdict.FAIL,
                score=0.0,
                feedback=f"All {self._n} Best-of-N attempts raised exceptions",
            )
            _set_parallel_detail(agent, attempts=(), selected=-1)
            raise AbstentionSignal(
                best_candidate=None,
                critique=synth,
                reason=synth.feedback,
            )

        selected_idx = _selection_rule(attempts)
        selected = attempts[selected_idx]

        _set_parallel_detail(agent, attempts=attempts, selected=selected_idx)

        if selected.abstained and raise_on_abstain:
            agent._logger.info(
                "Best-of-%d: no attempt passed the Critic — abstaining with "
                "best of %d candidates (score=%.2f)",
                self._n,
                len(attempts),
                selected.critique.score if selected.critique else 0.0,
            )
            raise AbstentionSignal(
                best_candidate=selected.result,
                critique=selected.critique
                or CritiqueResult(
                    verdict=Verdict.FAIL,
                    score=0.0,
                    feedback="Best-of-N abstained (no critique available)",
                ),
                reason=(
                    selected.abstention_reason
                    or (
                        selected.critique.feedback
                        if selected.critique
                        else "All Best-of-N candidates failed the Critic"
                    )
                ),
            )

        return selected.result


# --------------------------------------------------------------------- #
# Attempt outcome record                                                 #
# --------------------------------------------------------------------- #


class _AttemptOutcome:
    """Per-attempt result bundle (internal to this module)."""

    __slots__ = (
        "index",
        "result",
        "critique",
        "abstained",
        "abstention_reason",
        "detail",
    )

    def __init__(
        self,
        *,
        index: int,
        result: Any,
        critique: CritiqueResult | None,
        abstained: bool,
        abstention_reason: str | None,
        detail: AutonomousDetail,
    ) -> None:
        self.index = index
        self.result = result
        self.critique = critique
        self.abstained = abstained
        self.abstention_reason = abstention_reason
        self.detail = detail


# --------------------------------------------------------------------- #
# Selection rule (pure, testable)                                        #
# --------------------------------------------------------------------- #


def _selection_rule(attempts: list[_AttemptOutcome]) -> int:
    """Return the index of the winning attempt.

    Implements F4.3:

    1. PASS dominates UNCERTAIN: any PASS wins outright; tie-break by
       Critic score.
    2. Otherwise, UNCERTAIN verdicts with ``score >= UNCERTAIN_ACCEPT``
       are accepted; tie-break by score.
    3. Otherwise, return the attempt with the highest score (best of
       the bad) and let the caller abstain.

    Never returns an attempt that raised an exception (those are
    filtered out before this function is called).
    """
    if not attempts:
        raise ValueError("_selection_rule requires at least one attempt")

    passes = [
        a for a in attempts
        if a.critique is not None and a.critique.verdict == Verdict.PASS
    ]
    if passes:
        best = max(passes, key=lambda a: a.critique.score if a.critique else 0.0)
        return best.index

    uncertain_accept = [
        a for a in attempts
        if a.critique is not None
        and a.critique.verdict == Verdict.UNCERTAIN
        and a.critique.score >= _UNCERTAIN_ACCEPT_THRESHOLD
    ]
    if uncertain_accept:
        best = max(
            uncertain_accept,
            key=lambda a: a.critique.score if a.critique else 0.0,
        )
        return best.index

    # All attempts failed or were below threshold — return the least-bad
    # one so the caller can raise ``AbstentionSignal`` with a useful
    # best-candidate payload.
    best = max(
        attempts,
        key=lambda a: (a.critique.score if a.critique else 0.0),
    )
    return best.index


# --------------------------------------------------------------------- #
# Temperature perturbation                                               #
# --------------------------------------------------------------------- #


def _snapshot_temperature(agent: "Agent") -> dict[str, Any]:
    """Capture the current ``llm_params`` so attempt overrides can be undone."""
    cfg = agent.config
    params = getattr(cfg, "llm_params", None)
    return {
        "had_params": params is not None,
        "temperature": (
            getattr(params, "temperature", None)
            if params is not None
            else None
        ),
    }


def _apply_temperature(
    agent: "Agent",
    snapshot: dict[str, Any],
    *,
    attempt_index: int,
) -> None:
    """Mutate ``agent.config.llm_params.temperature`` for this attempt.

    - Attempt 0 uses the configured base temperature (no change), so
      ``N = 1`` runs stay bit-for-bit identical to pre-F4.
    - Attempts ``>= 1`` add ``_TEMPERATURE_DELTAS[i]`` to the base,
      capped at ``1.0``.
    """
    if attempt_index == 0:
        return  # baseline — never touch config on first attempt

    delta = _TEMPERATURE_DELTAS[
        min(attempt_index, len(_TEMPERATURE_DELTAS) - 1)
    ]
    base = snapshot.get("temperature")
    if base is None:
        # Provider default is unknown; assume 0.0 baseline so the
        # perturbation is still meaningful.
        base = 0.0
    new_temp = min(float(base) + float(delta), 1.0)

    cfg = agent.config
    params = getattr(cfg, "llm_params", None)
    if params is None:
        # Construct a fresh LLMParams carrying only the perturbed
        # temperature; other provider defaults remain in effect.
        try:
            from nucleusiq.llms.llm_params import LLMParams

            cfg.llm_params = LLMParams(temperature=new_temp)  # type: ignore[attr-defined]
        except Exception:
            return
    else:
        try:
            params.temperature = new_temp
        except Exception:
            pass


def _restore_temperature(agent: "Agent", snapshot: dict[str, Any]) -> None:
    """Undo ``_apply_temperature`` so the user's config is unchanged."""
    cfg = agent.config
    if not snapshot.get("had_params"):
        try:
            cfg.llm_params = None  # type: ignore[attr-defined]
        except Exception:
            pass
        return
    params = getattr(cfg, "llm_params", None)
    if params is not None:
        try:
            params.temperature = snapshot.get("temperature")
        except Exception:
            pass


# --------------------------------------------------------------------- #
# Per-attempt autonomous-detail isolation                                #
# --------------------------------------------------------------------- #


def _reset_autonomous_detail(tracer: Any) -> None:
    """Clear ``_autonomous_detail`` so the next attempt starts fresh.

    ``_llm_calls`` / ``_tool_calls`` are deliberately *not* reset —
    cost accounting must reflect total work across all attempts.
    """
    if tracer is None:
        return
    try:
        tracer._autonomous_detail.clear()
    except Exception:
        pass


def _build_detail(tracer: Any) -> AutonomousDetail:
    """Snapshot the current ``autonomous_detail`` as a frozen model."""
    if tracer is None:
        return AutonomousDetail()
    raw = tracer.autonomous_detail or {}
    try:
        return AutonomousDetail(**{k: v for k, v in raw.items() if k in _DETAIL_FIELDS})
    except Exception:
        return AutonomousDetail()


_DETAIL_FIELDS: frozenset[str] = frozenset(
    AutonomousDetail.model_fields.keys()
)


def _extract_last_critique(detail: AutonomousDetail) -> CritiqueResult | None:
    """Return the most recent Critic verdict as a ``CritiqueResult``.

    We reconstruct from the ``CritiqueSnapshot`` stored in the detail
    because the live ``CritiqueResult`` object is not retained after the
    inner runner returns.
    """
    snapshots: tuple[CritiqueSnapshot, ...] = detail.critic_verdicts
    if not snapshots:
        return None
    last = snapshots[-1]
    try:
        verdict = Verdict(last.verdict)
    except ValueError:
        verdict = Verdict.UNCERTAIN
    return CritiqueResult(
        verdict=verdict,
        score=last.score,
        feedback=last.feedback,
        issues=list(last.issues),
        suggestions=list(last.suggestions),
    )


# --------------------------------------------------------------------- #
# Final telemetry — roll per-attempt details onto the outer detail       #
# --------------------------------------------------------------------- #


def _set_parallel_detail(
    agent: "Agent",
    *,
    attempts: list[_AttemptOutcome] | tuple,
    selected: int,
) -> None:
    tracer = getattr(agent, "_tracer", None)
    if tracer is None:
        return

    parallel_details = tuple(a.detail for a in attempts)
    try:
        if attempts and 0 <= selected < len(attempts):
            winner = attempts[selected]
            winner_detail = winner.detail
            rollup: dict[str, Any] = {}
            for field in (
                "attempts",
                "max_attempts",
                "sub_tasks",
                "complexity",
                "refined",
                "revisions",
                "critic_verdicts",
                "escalations",
                "cumulative_tokens",
                "validations",
            ):
                rollup[field] = getattr(winner_detail, field)
            rollup["parallel_attempts"] = parallel_details
            rollup["selected_attempt"] = selected
            tracer.set_autonomous_detail(**rollup)
        else:
            tracer.set_autonomous_detail(
                parallel_attempts=parallel_details,
                selected_attempt=None,
            )
    except Exception:
        pass


__all__ = [
    "MAX_PARALLEL_ATTEMPTS",
    "ParallelRunner",
]
