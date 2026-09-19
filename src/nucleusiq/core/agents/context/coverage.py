"""Resource coverage — which declared resources a run actually touched (WS-4).

``Task.resources`` names what the task must cover.  This module watches
tool traffic and records which of those resources were *touched*: named
in a tool call's arguments, echoed at the head of a tool result, or
indexed into the document corpus.  The parent uses the union over its
children to compute ``unprocessed = resources − touched`` and, when
allowed, runs one bounded follow-up (PR F).

Matching is deliberately forgiving (case-insensitive, trailing path
component) — a resource id like ``docs/Q3 report.pdf`` will be passed
to tools as ``Q3 report.pdf``, a URL, or an absolute path.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Only the head of a tool result is scanned for resource names; a full
#: scan of a 200 KB payload per call would dominate small tools.
RESULT_SCAN_CHARS = 4_000


def normalize_resource(value: Any) -> str:
    text = str(value or "").strip().lower().replace("\\", "/")
    return text.rstrip("/")


def resource_aliases(resource: str) -> set[str]:
    """Forms under which ``resource`` may appear in tool traffic."""
    norm = normalize_resource(resource)
    if not norm:
        return set()
    aliases = {norm}
    tail = norm.rsplit("/", 1)[-1]
    if tail and len(tail) >= 3:
        aliases.add(tail)
    return aliases


def _strings_in(value: Any, *, depth: int = 0) -> Iterable[str]:
    if depth > 4:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings_in(v, depth=depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _strings_in(v, depth=depth + 1)


@dataclass
class ResourceTouchTracker:
    """Records which declared resources tool traffic has touched."""

    resources: list[str] = field(default_factory=list)
    _touched: dict[str, str] = field(default_factory=dict, repr=False)
    _aliases: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for r in self.resources:
            text = str(r or "").strip()
            if text and text not in seen:
                seen.add(text)
                cleaned.append(text)
        self.resources = cleaned
        self._aliases = {r: resource_aliases(r) for r in cleaned}

    # ------------------------------------------------------------------ #
    # Observation                                                          #
    # ------------------------------------------------------------------ #

    def observe(
        self,
        *,
        tool_name: str,
        tool_args: Any = None,
        tool_result: Any = None,
    ) -> list[str]:
        """Record resources named by this call. Returns newly touched ones."""
        if not self.resources:
            return []
        haystacks: list[str] = []
        for text in _strings_in(tool_args):
            haystacks.append(text)
        if tool_args is not None and not isinstance(tool_args, (str, dict, list)):
            haystacks.append(str(tool_args))
        if tool_result is not None:
            head = (
                tool_result
                if isinstance(tool_result, str)
                else json.dumps(tool_result, default=str)
                if isinstance(tool_result, (dict, list))
                else str(tool_result)
            )
            haystacks.append(head[:RESULT_SCAN_CHARS])
        if not haystacks:
            return []
        blob = "\n".join(haystacks).lower().replace("\\", "/")
        newly: list[str] = []
        for resource, aliases in self._aliases.items():
            if resource in self._touched:
                continue
            if any(alias in blob for alias in aliases):
                self._touched[resource] = tool_name
                newly.append(resource)
        return newly

    def mark(self, resources: Sequence[str], *, via: str = "merge") -> None:
        """Mark resources touched by another tracker (child merge)."""
        for r in resources:
            for declared, aliases in self._aliases.items():
                if declared in self._touched:
                    continue
                if normalize_resource(r) in aliases or any(
                    a in normalize_resource(r) for a in aliases if len(a) >= 3
                ):
                    self._touched[declared] = via

    # ------------------------------------------------------------------ #
    # Views                                                                #
    # ------------------------------------------------------------------ #

    @property
    def touched(self) -> list[str]:
        return [r for r in self.resources if r in self._touched]

    @property
    def unprocessed(self) -> list[str]:
        return [r for r in self.resources if r not in self._touched]

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": list(self.resources),
            "touched": self.touched,
            "unprocessed": self.unprocessed,
            "touched_via": dict(self._touched),
        }


def acknowledged(resource: str, text: str) -> bool:
    """True when the answer names ``resource`` (e.g. "x.pdf: could not be read").

    A resource the model explicitly reports on is *accounted for* even if
    no tool touched it — re-running for it would only burn budget.
    """
    if not text:
        return False
    blob = text.lower().replace("\\", "/")
    return any(alias in blob for alias in resource_aliases(resource))


#: Resources listed in a retry prompt; the rest is a count.
_MAX_RETRY_RESOURCES = 40


@dataclass
class CoverageReport:
    """Where the run stands against ``Task.resources`` (WS-4)."""

    resources: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)
    unprocessed: list[str] = field(default_factory=list)
    #: Unprocessed resources the final answer names explicitly.
    acknowledged: list[str] = field(default_factory=list)
    #: ``{"kind": "child" | "retry", "before": [...], ...}`` when a bounded
    #: follow-up ran for the gap; ``None`` otherwise.
    followup: dict[str, Any] | None = None
    enforce: bool = False

    @property
    def complete(self) -> bool:
        return not self.unprocessed

    @property
    def unaccounted(self) -> list[str]:
        return [r for r in self.unprocessed if r not in self.acknowledged]

    @property
    def blocked(self) -> bool:
        return bool(self.enforce and self.unprocessed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resources": list(self.resources),
            "touched": list(self.touched),
            "unprocessed": list(self.unprocessed),
            "acknowledged": list(self.acknowledged),
            "unaccounted": self.unaccounted,
            "complete": self.complete,
            "followup": dict(self.followup) if self.followup else None,
            "enforce": self.enforce,
            "blocked": self.blocked,
        }


def build_coverage(
    tracker: ResourceTouchTracker,
    *,
    answer_text: str = "",
    followup: dict[str, Any] | None = None,
    enforce: bool = False,
    corpus_document_ids: Iterable[str] = (),
) -> CoverageReport:
    """Reconcile the tracker against the answer and the merged corpus."""
    docs = [d for d in corpus_document_ids if isinstance(d, str) and d.strip()]
    if docs:
        tracker.mark(docs, via="corpus")
    unprocessed = tracker.unprocessed
    return CoverageReport(
        resources=list(tracker.resources),
        touched=tracker.touched,
        unprocessed=unprocessed,
        acknowledged=[r for r in unprocessed if acknowledged(r, answer_text)],
        followup=dict(followup) if followup else None,
        enforce=bool(enforce),
    )


def has_business_tools(agent: Any) -> bool:
    """True when the agent has at least one non-context-management tool.

    Without one, no follow-up can process anything — the gap is recorded
    but no retry or child is spent on it.
    """
    try:
        from nucleusiq.agents.context.workspace_tools import (
            is_context_management_tool_name,
        )
    except Exception:  # pragma: no cover - import guard
        return bool(getattr(agent, "tools", None))
    return any(
        not is_context_management_tool_name(getattr(t, "name", None))
        for t in (getattr(agent, "tools", None) or [])
    )


def coverage_retry_message(agent: Any, result: Any) -> str | None:
    """One-shot retry prompt when declared resources were never processed.

    Returns ``None`` when there is nothing to do: no resources declared, no
    tracker, no business tools to process them with, every resource touched
    or named in the answer, or the run has already spent its single coverage
    follow-up (child or retry).  Marks the follow-up as used on the agent so
    the retry is bounded to one.
    """
    tracker = getattr(agent, "_resource_tracker", None)
    if tracker is None or not getattr(tracker, "resources", None):
        return None
    if getattr(agent, "_coverage_followup", None):
        return None
    if not bool(getattr(getattr(agent, "config", None), "coverage_followup", True)):
        return None
    if not has_business_tools(agent):
        return None
    corpus = getattr(agent, "_document_corpus", None)
    lister = getattr(corpus, "list_documents", None)
    if callable(lister):
        with contextlib.suppress(Exception):
            tracker.mark([getattr(d, "id", "") for d in lister()], via="corpus")
    text = str(result) if result is not None else ""
    pending = [r for r in tracker.unprocessed if not acknowledged(r, text)]
    if not pending:
        return None

    agent._coverage_followup = {
        "kind": "retry",
        "ran": True,
        "before": list(tracker.unprocessed),
        "pending": list(pending),
    }
    recorder = getattr(agent, "_run_recorder", None)
    if recorder is not None:
        with contextlib.suppress(Exception):
            recorder.record_event(
                "coverage_retry",
                f"{len(pending)} declared resource(s) never processed: "
                + ", ".join(pending[:5]),
            )

    listed = "\n".join(f"- {r}" for r in pending[:_MAX_RETRY_RESOURCES])
    more = (
        f"\n- … and {len(pending) - _MAX_RETRY_RESOURCES} more"
        if len(pending) > _MAX_RETRY_RESOURCES
        else ""
    )
    return (
        "Your answer does not account for every declared resource. The "
        "following were never processed and are not mentioned:\n"
        f"{listed}{more}\n\n"
        "Process ONLY these resources now with your tools (do not redo the "
        "others), then give the complete answer covering every resource. For "
        "any you cannot process, say so explicitly by name."
    )


__all__ = [
    "RESULT_SCAN_CHARS",
    "CoverageReport",
    "ResourceTouchTracker",
    "acknowledged",
    "build_coverage",
    "coverage_retry_message",
    "has_business_tools",
    "normalize_resource",
    "resource_aliases",
]
