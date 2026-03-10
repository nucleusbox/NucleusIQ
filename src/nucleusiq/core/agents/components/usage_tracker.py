"""
UsageTracker — accumulates token usage across all LLM calls in one execution.

Provides a purpose-tagged breakdown so users understand where their
token budget is spent (planning vs main execution vs tool loop vs
critic vs refiner).

Usage:
    tracker = UsageTracker()
    tracker.record(CallPurpose.MAIN, {"prompt_tokens": 100, "completion_tokens": 50})
    tracker.record(CallPurpose.TOOL_LOOP, {"prompt_tokens": 200, "completion_tokens": 80})
    print(tracker.summary)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class CallPurpose(str, Enum):
    """Labels for different LLM call purposes within one execution."""

    MAIN = "main"
    PLANNING = "planning"
    TOOL_LOOP = "tool_loop"
    CRITIC = "critic"
    REFINER = "refiner"


class UsageRecord(BaseModel):
    """Token usage from a single LLM call."""

    purpose: CallPurpose
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
        """
        if usage is None:
            return

        prompt = _safe_int(usage, "prompt_tokens")
        completion = _safe_int(usage, "completion_tokens")
        total = _safe_int(usage, "total_tokens") or (prompt + completion)
        reasoning = _safe_int(usage, "reasoning_tokens")

        self._records.append(
            UsageRecord(
                purpose=purpose,
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

        self.record(purpose, usage_dict, call_round=call_round)

    def record_from_stream_metadata(
        self,
        purpose: CallPurpose,
        metadata: dict[str, Any] | None,
        *,
        call_round: int = 1,
    ) -> None:
        """Extract usage from a StreamEvent's metadata dict."""
        if not metadata:
            return
        usage = metadata.get("usage")
        if usage is None:
            return
        if hasattr(usage, "model_dump"):
            self.record(purpose, usage.model_dump(), call_round=call_round)
        elif isinstance(usage, dict):
            self.record(purpose, usage, call_round=call_round)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    @property
    def summary(self) -> dict[str, Any]:
        """Return a structured breakdown of accumulated usage.

        Returns
        -------
        dict
            ``{"total": {...}, "by_purpose": {...}, "call_count": int}``
        """
        total_prompt = 0
        total_completion = 0
        total_total = 0
        total_reasoning = 0

        by_purpose: dict[str, dict[str, int]] = {}

        for rec in self._records:
            total_prompt += rec.prompt_tokens
            total_completion += rec.completion_tokens
            total_total += rec.total_tokens
            total_reasoning += rec.reasoning_tokens

            key = rec.purpose.value
            if key not in by_purpose:
                by_purpose[key] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "reasoning_tokens": 0,
                    "calls": 0,
                }
            by_purpose[key]["prompt_tokens"] += rec.prompt_tokens
            by_purpose[key]["completion_tokens"] += rec.completion_tokens
            by_purpose[key]["total_tokens"] += rec.total_tokens
            by_purpose[key]["reasoning_tokens"] += rec.reasoning_tokens
            by_purpose[key]["calls"] += 1

        return {
            "total": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_total,
                "reasoning_tokens": total_reasoning,
            },
            "by_purpose": by_purpose,
            "call_count": len(self._records),
        }

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
