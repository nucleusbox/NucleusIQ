"""Context Mgmt v2 — Step 2: recall tracker.

Tiny per-execution data structure that records *when* the model
called ``recall_tool_result`` (or ``list_recalled_evidence``) and
*which* refs it asked for.

Two consumers in v2:

* The :class:`PolicyClassifier` doesn't use this directly, but a
  recently-recalled ref is by definition still *useful* to the
  Generator — so the Compactor's pin-set logic asks the tracker for
  the **hot set** (refs recalled within the last ``N`` turns) and
  refuses to evict them.
* :class:`~nucleusiq.agents.context.telemetry.ContextTelemetry`
  reports ``recall_count`` / ``recall_tokens`` so operators can see
  how often the model is exercising the recall path.

Why a class and not a dict?  Because we need a stable *turn counter*
to compute the hot set ("recalled within the last N turns" only
makes sense if we know how many turns have passed since the recall).
The class owns that counter and exposes it as
:meth:`mark_turn_completed`, which the engine bumps once per
``post_response``.

Step 2 only *records* the data and exposes the hot set; Step 3 is
where the new ``Compactor`` actually consumes :meth:`hot_set` to
build its pin set.  Wiring the tracker now means Step 3 has no
schema work to do.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecallEvent:
    """One recall invocation, captured at the moment it happens."""

    ref: str
    turn: int
    tokens: int


__all__ = ["RecallEvent", "RecallTracker"]


class RecallTracker:
    """Tracks recall invocations within a single agent execution.

    Lifecycle parallels :class:`ContextEngine`: instantiated once per
    execute call, garbage-collected when the call ends.  Not
    intended to be persistent or thread-safe — every consumer runs
    on the agent's loop.
    """

    __slots__ = ("_events", "_turn", "_total_tokens", "_max_history")

    #: Soft cap on the number of recent recall events kept in memory.
    #: We never *need* more than :attr:`hot_set_lookback_turns` worth
    #: of history, but the deque stays bounded for safety on very
    #: long runs.  1024 is overkill for any realistic agent — at one
    #: recall per turn that's well past the
    #: ``preserve_recent_turns`` window any compactor would consider.
    _DEFAULT_MAX_HISTORY: int = 1024

    def __init__(self, *, max_history: int | None = None) -> None:
        self._events: deque[RecallEvent] = deque(
            maxlen=max_history or self._DEFAULT_MAX_HISTORY
        )
        self._turn: int = 0
        self._total_tokens: int = 0
        self._max_history: int = max_history or self._DEFAULT_MAX_HISTORY

    # ------------------------------------------------------------------ #
    # Recording                                                          #
    # ------------------------------------------------------------------ #

    def record_recall(self, ref: str, *, tokens: int) -> None:
        """Record one ``recall_tool_result`` invocation.

        Called by the recall tool itself (see
        ``nucleusiq.agents.context.recall_tools``).  ``tokens`` is the
        *recalled* content's token count — used for telemetry, not
        for any control-flow decision.

        The event is tagged with the *current* turn number; the turn
        counter is bumped externally by :meth:`mark_turn_completed`.
        That separation keeps recording cheap and auditable: a recall
        observed mid-turn always belongs to the turn-in-progress.
        """
        if not ref:
            return
        event = RecallEvent(ref=ref, turn=self._turn, tokens=max(0, int(tokens)))
        self._events.append(event)
        self._total_tokens += event.tokens

    def mark_turn_completed(self) -> None:
        """Advance the internal turn counter by one.

        Called by :class:`ContextEngine` once per ``post_response``
        (i.e. once per LLM round-trip).  Keeps :meth:`hot_set` in
        sync with the live conversation.
        """
        self._turn += 1

    # ------------------------------------------------------------------ #
    # Queries                                                            #
    # ------------------------------------------------------------------ #

    def hot_set(self, *, lookback_turns: int) -> set[str]:
        """Return refs recalled within the last ``lookback_turns`` turns.

        ``lookback_turns == 0`` returns only refs recalled *during the
        current turn*; ``lookback_turns >= turn`` returns every ref
        ever recalled.

        The set is recomputed each call — cheap because the deque is
        bounded and most agents only recall a handful of times per
        turn.
        """
        if lookback_turns < 0:
            return set()
        cutoff = self._turn - lookback_turns
        return {e.ref for e in self._events if e.turn >= cutoff}

    @property
    def recall_count(self) -> int:
        """Total number of recall invocations across this execution."""
        return len(self._events)

    @property
    def total_recalled_tokens(self) -> int:
        """Total tokens returned across every recall.

        Useful for telemetry — operators want to see how much "extra
        evidence" the model pulled back via recall.  Not the same as
        ``store.size``, which is *offloaded* tokens.
        """
        return self._total_tokens

    @property
    def turn(self) -> int:
        """Current turn counter (read-only)."""
        return self._turn
