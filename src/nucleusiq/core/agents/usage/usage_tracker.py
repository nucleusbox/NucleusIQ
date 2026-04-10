"""
UsageTracker — accumulates token usage across all LLM calls in one execution.

Provides a purpose-tagged **and** origin-tagged breakdown so users
understand where their token budget is spent:

* **by_purpose** — planning vs main vs tool_loop vs critic vs refiner
* **by_origin** — *user* tokens (the user's actual task) vs *framework*
  tokens (system prompts, orchestration, tool-loop overhead, critic
  passes, etc.)

The ``by_origin`` split is designed so that a future Observability
plugin can consume ``UsageRecord`` objects directly (each record is
already tagged with ``purpose`` + ``origin``).

Usage:
    tracker = UsageTracker()
    tracker.record(CallPurpose.MAIN, {"prompt_tokens": 100, "completion_tokens": 50})
    tracker.record(CallPurpose.TOOL_LOOP, {"prompt_tokens": 200, "completion_tokens": 80})
    print(tracker.summary)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CallPurpose(str, Enum):
    """Labels for different LLM call purposes within one execution."""

    MAIN = "main"
    PLANNING = "planning"
    TOOL_LOOP = "tool_loop"
    SYNTHESIS = "synthesis"
    CRITIC = "critic"
    REFINER = "refiner"


class TokenOrigin(str, Enum):
    """Distinguishes user-initiated tokens from framework overhead.

    * ``USER`` — the initial MAIN call that carries the user's task
      objective.  This is the irreducible token cost of the user's
      request.
    * ``FRAMEWORK`` — everything the framework adds: system prompt,
      tool-loop iterations, planning, critic, refiner, empty-response
      retries, memory injection, etc.

    The mapping is intentionally coarse (per-call, not per-message)
    because providers return aggregate usage per API call.  A future
    Observability plugin can compute finer-grained attribution by
    counting message tokens before the call.
    """

    USER = "user"
    FRAMEWORK = "framework"


#: Default mapping from CallPurpose to TokenOrigin.
#: MAIN is tagged as USER; everything else is FRAMEWORK.
#: Callers can override per-record via the ``origin`` parameter.
PURPOSE_ORIGIN_MAP: dict[CallPurpose, TokenOrigin] = {
    CallPurpose.MAIN: TokenOrigin.USER,
    CallPurpose.PLANNING: TokenOrigin.FRAMEWORK,
    CallPurpose.TOOL_LOOP: TokenOrigin.FRAMEWORK,
    CallPurpose.SYNTHESIS: TokenOrigin.FRAMEWORK,
    CallPurpose.CRITIC: TokenOrigin.FRAMEWORK,
    CallPurpose.REFINER: TokenOrigin.FRAMEWORK,
}


class TokenCount(BaseModel):
    """Aggregate token counts shared by totals, purpose buckets, and origin buckets."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


class BucketStats(TokenCount):
    """Token counts for a single purpose or origin bucket, with call count."""

    calls: int = 0


class UsageSummary(BaseModel):
    """Fixed schema for the output of ``UsageTracker.summary``.

    Returned by ``agent.last_usage`` after every ``execute()`` call.

    Quick usage::

        usage = agent.last_usage

        # Plain dict -- for logging, dashboards, JSON serialization
        usage.summary()

        # Human-readable formatted string -- for console / debug
        print(usage.display())

        # Individual fields (typed, IDE-autocomplete)
        usage.total.prompt_tokens
        usage.by_origin["user"].total_tokens
    """

    total: TokenCount = Field(default_factory=TokenCount)
    call_count: int = 0
    by_purpose: dict[str, BucketStats] = Field(default_factory=dict)
    by_origin: dict[str, BucketStats] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a plain dict of the full usage breakdown.

        Convenience wrapper so users never need to call ``model_dump()``
        directly.  The returned dict is JSON-serializable.

        Example::

            usage = agent.last_usage
            d = usage.summary()
            # d == {
            #   "total": {"prompt_tokens": 100, ...},
            #   "call_count": 2,
            #   "by_purpose": {"main": {...}, "tool_loop": {...}},
            #   "by_origin":  {"user": {...}, "framework": {...}},
            # }
        """
        return self.model_dump()

    def display(self) -> str:
        """Return a consolidated, human-readable summary string.

        Includes totals, per-purpose breakdown, per-origin breakdown,
        and a percentage split between user and framework tokens.

        Example output::

            Usage Summary (2 LLM calls)
            ─────────────────────────────
            Totals
              Prompt:     100 tokens
              Completion:  50 tokens
              Reasoning:    0 tokens
              Total:      150 tokens

            By Purpose
              main:        75 tokens (1 call)
              tool_loop:   75 tokens (1 call)

            By Origin
              user:        75 tokens (1 call)  — 50%
              framework:   75 tokens (1 call)  — 50%
        """
        lines: list[str] = []
        lines.append(
            f"Usage Summary ({self.call_count} LLM call{'s' if self.call_count != 1 else ''})"
        )
        lines.append("-" * 36)

        lines.append("Totals")
        lines.append(f"  Prompt:     {self.total.prompt_tokens:>8} tokens")
        lines.append(f"  Completion: {self.total.completion_tokens:>8} tokens")
        lines.append(f"  Reasoning:  {self.total.reasoning_tokens:>8} tokens")
        lines.append(f"  Total:      {self.total.total_tokens:>8} tokens")

        if self.by_purpose:
            lines.append("")
            lines.append("By Purpose")
            for name, b in self.by_purpose.items():
                lines.append(
                    f"  {name + ':':14s} {b.total_tokens:>8} tokens "
                    f"({b.calls} call{'s' if b.calls != 1 else ''})"
                )

        if self.by_origin:
            lines.append("")
            lines.append("By Origin")
            total = self.total.total_tokens or 1
            for name, b in self.by_origin.items():
                pct = b.total_tokens / total * 100
                lines.append(
                    f"  {name + ':':14s} {b.total_tokens:>8} tokens "
                    f"({b.calls} call{'s' if b.calls != 1 else ''})  "
                    f"-- {pct:.0f}%"
                )

        return "\n".join(lines)


class UsageRecord(BaseModel):
    """Token usage from a single LLM call."""

    purpose: CallPurpose
    origin: TokenOrigin = TokenOrigin.FRAMEWORK
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    call_round: int = 1


class UsageTracker:
    """Accumulates :class:`UsageRecord` objects across an entire ``execute()`` call.

    Each mode tags its LLM calls with a :class:`CallPurpose` so the final
    summary shows a per-purpose breakdown.
    """

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    @property
    def records(self) -> list[UsageRecord]:
        return list(self._records)

    def record(
        self,
        purpose: CallPurpose,
        usage: dict[str, Any] | None,
        *,
        call_round: int = 1,
        origin: TokenOrigin | None = None,
    ) -> None:
        """Record usage from a single LLM call.

        Parameters
        ----------
        purpose : CallPurpose
            What this call was for.
        usage : dict | None
            Raw usage dict from the provider (keys: ``prompt_tokens``,
            ``completion_tokens``, ``total_tokens``, ``reasoning_tokens``).
            ``None`` is accepted gracefully (no-op).
        call_round : int
            Which round of the tool loop this belongs to.
        origin : TokenOrigin | None
            Whether this call is user-initiated or framework overhead.
            Defaults to the mapping in ``PURPOSE_ORIGIN_MAP``.
        """
        if usage is None:
            return

        resolved_origin = origin or PURPOSE_ORIGIN_MAP.get(
            purpose, TokenOrigin.FRAMEWORK
        )

        prompt = _safe_int(usage, "prompt_tokens")
        completion = _safe_int(usage, "completion_tokens")
        total = _safe_int(usage, "total_tokens") or (prompt + completion)
        reasoning = _safe_int(usage, "reasoning_tokens")

        self._records.append(
            UsageRecord(
                purpose=purpose,
                origin=resolved_origin,
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                reasoning_tokens=reasoning,
                call_round=call_round,
            )
        )

    def record_from_response(
        self,
        purpose: CallPurpose,
        response: Any,
        *,
        call_round: int = 1,
        origin: TokenOrigin | None = None,
    ) -> None:
        """Extract usage from a provider response object and record it.

        Tries ``response.usage`` (attribute) and ``response["usage"]``
        (dict).  Handles both Pydantic models and plain dicts.
        """
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None and isinstance(response, dict):
            usage_obj = response.get("usage")

        if usage_obj is None:
            return

        if isinstance(usage_obj, dict):
            usage_dict = usage_obj
        elif hasattr(usage_obj, "model_dump"):
            usage_dict = usage_obj.model_dump()
        else:
            usage_dict = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
                "reasoning_tokens": getattr(usage_obj, "reasoning_tokens", 0),
            }

        self.record(purpose, usage_dict, call_round=call_round, origin=origin)

    def record_from_stream_metadata(
        self,
        purpose: CallPurpose,
        metadata: dict[str, Any] | None,
        *,
        call_round: int = 1,
        origin: TokenOrigin | None = None,
    ) -> None:
        """Extract usage from a StreamEvent's metadata dict."""
        if not metadata:
            return
        usage = metadata.get("usage")
        if usage is None:
            return
        if hasattr(usage, "model_dump"):
            self.record(
                purpose, usage.model_dump(), call_round=call_round, origin=origin
            )
        elif isinstance(usage, dict):
            self.record(purpose, usage, call_round=call_round, origin=origin)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    @property
    def summary(self) -> UsageSummary:
        """Return a structured breakdown of accumulated usage.

        Returns
        -------
        UsageSummary
            Pydantic model with ``total``, ``call_count``, ``by_purpose``,
            ``by_origin``.

            ``by_origin`` separates *user* tokens (the initial MAIN call
            carrying the user's task) from *framework* overhead (system
            prompts, tool loops, planning, critic, refiner).

            Call ``.model_dump()`` to get a plain ``dict``.
        """
        total = TokenCount()
        by_purpose: dict[str, BucketStats] = {}
        by_origin: dict[str, BucketStats] = {}

        for rec in self._records:
            total.prompt_tokens += rec.prompt_tokens
            total.completion_tokens += rec.completion_tokens
            total.total_tokens += rec.total_tokens
            total.reasoning_tokens += rec.reasoning_tokens

            for bucket_key, bucket_map in (
                (rec.purpose.value, by_purpose),
                (rec.origin.value, by_origin),
            ):
                if bucket_key not in bucket_map:
                    bucket_map[bucket_key] = BucketStats()
                bucket = bucket_map[bucket_key]
                bucket.prompt_tokens += rec.prompt_tokens
                bucket.completion_tokens += rec.completion_tokens
                bucket.total_tokens += rec.total_tokens
                bucket.reasoning_tokens += rec.reasoning_tokens
                bucket.calls += 1

        return UsageSummary(
            total=total,
            call_count=len(self._records),
            by_purpose=by_purpose,
            by_origin=by_origin,
        )

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._records)

    @property
    def call_count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        """Clear all recorded usage (for reuse across executions)."""
        self._records.clear()


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _safe_int(d: dict[str, Any], key: str) -> int:
    """Extract an int from *d*, returning 0 on missing/invalid values."""
    val = d.get(key)
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
