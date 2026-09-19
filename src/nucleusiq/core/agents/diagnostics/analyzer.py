"""Offline, deterministic analyzer for :class:`RunReport` (WS-7 §7.2).

Every rule is a pure function ``RunReport -> Finding | None``.  Rules
read **only** counters, resolved configuration, decisions, children,
coverage and the termination record — never task text or payloads — so
the analyzer can run on a redacted report pasted from an issue exactly
as it runs in-process at the end of every ``Agent.execute()``.

Each :class:`Finding` names the report fields it used (``evidence``) so
a builder can verify the claim, what most likely caused it (``cause``),
what to change in their own configuration right now (``fix_now``), and
which framework change addresses the root cause (``fix_release``).

Adding a rule
-------------
Write a function decorated with :func:`rule`.  Keep it total: it must
never raise on a partially populated report (use ``.get`` with
defaults).  Rules are evaluated in registration order; findings are
returned sorted by severity (critical → warning → info) and then by
registration order so the first three shown by ``AgentResult.display()``
are always the most important ones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from nucleusiq.agents.diagnostics.run_report import (
    Finding,
    RunReport,
    TerminationReason,
)

Rule = Callable[[RunReport], "Finding | None"]

_RULES: list[Rule] = []

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}

#: Children spawned by the harness itself (gather-first, coverage
#: follow-up).  They intentionally run with tighter caps than the parent
#: and are excluded from the "inherited limits" rules.
_AUX_CHILD_IDS = frozenset({"gather", "coverage-followup"})

#: 0.7.13's hidden sub-agent plugin cap.
_LEGACY_CHILD_CAP = 15


def rule(fn: Rule) -> Rule:
    """Register ``fn`` as an analyzer rule (decorator)."""
    _RULES.append(fn)
    return fn


def registered_rules() -> tuple[str, ...]:
    """Names of all registered rules, in evaluation order."""
    return tuple(fn.__name__ for fn in _RULES)


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _classification(report: RunReport) -> dict[str, Any]:
    return _dict(report.decisions.get("classification"))


def _preflight(report: RunReport) -> dict[str, Any]:
    pre = report.preflight
    if not isinstance(pre, dict):
        pre = _dict(report.decisions.get("preflight"))
    if not pre:
        pre = _dict(report.config_resolved.get("preflight"))
    return pre


def _is_complex(report: RunReport) -> bool:
    return bool(_classification(report).get("is_complex")) or bool(report.children)


def _business_children(report: RunReport) -> list[dict[str, Any]]:
    return [c for c in report.children if str(c.get("id", "")) not in _AUX_CHILD_IDS]


def _terminated(report: RunReport, reason: TerminationReason) -> bool:
    return report.termination.reason == reason


def _termination_evidence(report: RunReport) -> dict[str, Any]:
    t = report.termination
    return {
        "termination.reason": t.reason.value,
        "termination.at_round": t.at_round,
        "termination.at_tool_call": t.at_tool_call,
        "termination.message": t.message[:160],
    }


# --------------------------------------------------------------------- #
# Loop / progress rules                                                  #
# --------------------------------------------------------------------- #


@rule
def loop_recall_deadlock(report: RunReport) -> Finding | None:
    c = report.counters
    if c.recall_errors >= 3 and c.dedup_banners >= 3 and c.emergency_count >= 1:
        return Finding(
            code="LOOP_RECALL_DEADLOCK",
            severity="critical",
            title=(
                "Model was told to recall evidence that emergency compaction "
                "had already evicted — the classic 0.7.13 infinite loop"
            ),
            evidence={
                "counters.recall_errors": c.recall_errors,
                "counters.dedup_banners": c.dedup_banners,
                "counters.emergency_count": c.emergency_count,
                "counters.tool_calls_context": c.tool_calls_context,
            },
            cause=(
                "Dedup banners pointed at recall refs that the emergency "
                "compactor had dropped; every recall failed and the model "
                "re-issued the same tool call."
            ),
            fix_now=(
                "Upgrade past 0.7.13. Until then: set an explicit "
                "context_window, do not mark every tool idempotent, and cap "
                "max_context_tool_calls."
            ),
            fix_release=(
                "Refs are pinned while a banner cites them; recall errors "
                "and context-tool calls are bounded (PR A / WS-6)."
            ),
        )
    return None


@rule
def loop_no_progress(report: RunReport) -> Finding | None:
    c = report.counters
    if _terminated(report, TerminationReason.NO_PROGRESS) or c.no_progress_rounds >= 3:
        return Finding(
            code="LOOP_NO_PROGRESS",
            severity="warning",
            title="Run stopped because consecutive rounds made no progress",
            evidence={
                **_termination_evidence(report),
                "counters.no_progress_rounds": c.no_progress_rounds,
                "counters.stalled_rounds": c.stalled_rounds,
                "counters.rounds": c.rounds,
            },
            cause=(
                "The model repeated identical tool calls (or empty turns) "
                "without new information — usually a tool set that is too "
                "wide, a vague objective, or truncated model output."
            ),
            fix_now=(
                "Shrink the tool set, tighten the objective, or raise "
                "llm_max_output_tokens if responses look cut off."
            ),
            fix_release="No-progress detector nudges once, then terminates cleanly.",
        )
    return None


@rule
def emergency_thrash(report: RunReport) -> Finding | None:
    c = report.counters
    if c.emergency_count >= 2:
        cfg = report.config_resolved
        return Finding(
            code="EMERGENCY_THRASH",
            severity="warning",
            title=f"Emergency compaction ran {c.emergency_count} times",
            evidence={
                "counters.emergency_count": c.emergency_count,
                "counters.compactions_by_strategy": dict(c.compactions_by_strategy),
                "config_resolved.context_window": cfg.get("context_window"),
                "config_resolved.optimal_budget": cfg.get("optimal_budget"),
                "counters.max_prompt_tokens_seen": c.max_prompt_tokens_seen,
            },
            cause=(
                "The working budget is too small for the tool results being "
                "returned; the compactor keeps evicting what the model needs."
            ),
            fix_now=(
                "Declare context_window explicitly, lower "
                "tool_result_per_call_max_chars, or give the run fewer tools."
            ),
            fix_release="Preflight fitness check and BudgetResolver (PR C / WS-1).",
        )
    return None


@rule
def evidence_evicted_unseen(report: RunReport) -> Finding | None:
    c = report.counters
    if c.unseen_evidence_evicted >= 1:
        cfg = report.config_resolved
        pre = _preflight(report) or {}
        return Finding(
            code="EVIDENCE_EVICTED_UNSEEN",
            severity="critical",
            title=(
                f"{c.unseen_evidence_evicted} tool result(s) were evicted before the "
                "model read them — the answer is not grounded in them"
            ),
            evidence={
                "counters.unseen_evidence_evicted": c.unseen_evidence_evicted,
                "counters.emergency_count": c.emergency_count,
                "counters.tool_calls_business": c.tool_calls_business,
                "config_resolved.context_window": cfg.get("context_window"),
                "config_resolved.optimal_budget": cfg.get("optimal_budget"),
                "preflight.working_tokens": pre.get("working_tokens"),
                "counters.max_prompt_tokens_seen": c.max_prompt_tokens_seen,
            },
            cause=(
                "One round of tool calls returned more than the working budget "
                "can hold, and the results had to be dropped before any "
                "assistant turn could read them. Whatever the output says about "
                "those resources came from the model's prior, not the evidence."
            ),
            fix_now=(
                "Treat the output as unverified for the affected resources. "
                "Use a larger context window, ask the model to process fewer "
                "resources per turn, or split the task so each child owns a "
                "subset that fits."
            ),
            fix_release=(
                "Adaptive Tier-1 offload keeps unseen results recallable "
                "instead of evicting them; the counter exists so the residual "
                "case is never silent."
            ),
        )
    return None


@rule
def critic_partial_view(report: RunReport) -> Finding | None:
    c = report.counters
    if c.critic_partial_views < 1:
        return None
    views = report.decisions.get("critic_views") or []
    last = views[-1] if isinstance(views, list) and views else {}
    cfg = report.config_resolved
    return Finding(
        code="CRITIC_PARTIAL_VIEW",
        severity="warning",
        title=(
            f"Critic verified against partial evidence in {c.critic_partial_views} "
            f"pass(es); {c.critic_fail_downgraded} FAIL verdict(s) were downgraded "
            "to UNCERTAIN"
        ),
        evidence={
            "counters.critic_partial_views": c.critic_partial_views,
            "counters.critic_fail_downgraded": c.critic_fail_downgraded,
            "counters.critic_verdicts": dict(c.critic_verdicts),
            "decisions.critic_views[-1]": last,
            "config_resolved.context_window": cfg.get("context_window"),
        },
        cause=(
            "The window could not hold every tool result whole for the Critic "
            "and the curated package had to drop or shorten items, so the "
            "verifier saw less than the generator. By invariant I-10 such a "
            "verifier may ask for another pass but may not condemn the answer "
            "as ungrounded on its own."
        ),
        fix_now=(
            "Read the downgraded feedback as points to re-check, not as proven "
            "errors. A larger context window or fewer/shorter tool results per "
            "run lets the Critic see everything and gives it full authority."
        ),
        fix_release=(
            "Package and per-result caps are window-derived; the residual case "
            "is flagged here and in the critique's `evidence_view`."
        ),
    )


@rule
def window_fallback(report: RunReport) -> Finding | None:
    cfg = report.config_resolved
    pre = _preflight(report)
    window = _int(cfg.get("context_window"), 0)
    is_fallback = bool(cfg.get("window_is_fallback")) or bool(
        pre.get("window_is_fallback")
    )
    if is_fallback:
        return Finding(
            code="WINDOW_FALLBACK",
            severity="warning",
            title=f"Context window {window or '?'} is a fallback, not a declared value",
            evidence={
                "config_resolved.context_window": cfg.get("context_window"),
                "config_resolved.window_is_fallback": cfg.get("window_is_fallback"),
                "preflight.window_is_fallback": pre.get("window_is_fallback"),
            },
            cause=(
                "Neither the builder nor the provider adapter declared the "
                "model's window, so the framework used a default."
            ),
            fix_now="Pass context_window= (and response_reserve=) explicitly.",
            fix_release="Preflight reports the fallback so it is visible.",
        )
    return None


# --------------------------------------------------------------------- #
# Decomposition / child rules                                            #
# --------------------------------------------------------------------- #


@rule
def child_window_mismatch(report: RunReport) -> Finding | None:
    parent_window = _int(report.config_resolved.get("context_window"), 0)
    if parent_window <= 0:
        return None
    offenders = [
        {"id": c.get("id"), "window": c.get("window")}
        for c in _business_children(report)
        if _int(c.get("window"), parent_window) < parent_window
    ]
    if offenders:
        return Finding(
            code="CHILD_WINDOW_MISMATCH",
            severity="warning",
            title="Sub-agents ran with a smaller context window than the parent",
            evidence={
                "config_resolved.context_window": parent_window,
                "children[].window": offenders,
            },
            cause="Context configuration was not inherited by the sub-agents.",
            fix_now="Set max_sub_agents=1 until upgraded.",
            fix_release="Sub-agents inherit the parent's resolved window (PR C).",
        )
    return None


@rule
def child_hidden_cap(report: RunReport) -> Finding | None:
    parent_cap = _int(report.config_resolved.get("max_tool_calls"), 0)
    if parent_cap <= 0:
        return None
    offenders = [
        {"id": c.get("id"), "max_tool_calls": c.get("max_tool_calls")}
        for c in _business_children(report)
        if 0 < _int(c.get("max_tool_calls"), parent_cap) < parent_cap
    ]
    if not offenders:
        return None
    legacy = all(o["max_tool_calls"] == _LEGACY_CHILD_CAP for o in offenders) and (
        parent_cap != _LEGACY_CHILD_CAP
    )
    return Finding(
        code="CHILD_HIDDEN_CAP",
        severity="warning" if legacy else "info",
        title=(
            "Sub-agents were capped at 15 tool calls by a hidden plugin"
            if legacy
            else "Sub-agents ran with a tighter tool-call cap than the parent"
        ),
        evidence={
            "config_resolved.max_tool_calls": parent_cap,
            "children[].max_tool_calls": offenders,
        },
        cause=(
            "0.7.13 attached a sub-agent plugin with a fixed cap of 15."
            if legacy
            else "The child cap was tightened deliberately or not inherited."
        ),
        fix_now="Set max_sub_agents=1, or raise max_tool_calls on the parent.",
        fix_release="Children inherit the parent's resolved cap (PR C).",
    )


@rule
def decomp_shared_source(report: RunReport) -> Finding | None:
    if not _is_complex(report):
        return None
    children = _business_children(report)
    sets = [
        (str(c.get("id")), {str(r) for r in _list(c.get("touched_resources"))})
        for c in children
    ]
    sets = [(cid, s) for cid, s in sets if s]
    overlaps: list[dict[str, Any]] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a_id, a = sets[i]
            b_id, b = sets[j]
            shared = len(a & b)
            ratio = shared / max(1, min(len(a), len(b)))
            if shared and ratio > 0.5:
                overlaps.append(
                    {
                        "children": [a_id, b_id],
                        "shared": shared,
                        "ratio": round(ratio, 2),
                    }
                )
    if overlaps:
        return Finding(
            code="DECOMP_SHARED_SOURCE",
            severity="warning",
            title="Sub-tasks re-read the same sources — the split was not grounded",
            evidence={"children[].touched_resources": overlaps},
            cause=(
                "The decomposer split the objective along topics rather than "
                "along inputs, so several children processed the same material."
            ),
            fix_now=(
                "Declare Task.resources so the split is per resource, or set "
                "enable_decomposition=False / max_sub_agents=1."
            ),
            fix_release="Gate 4 resource-grounded decomposition (PR D / WS-2).",
        )
    return None


@rule
def decomp_coverage_gap(report: RunReport) -> Finding | None:
    cov = _dict(report.coverage)
    unprocessed = _list(cov.get("unprocessed"))
    if not unprocessed:
        return None
    blocked = bool(cov.get("blocked"))
    followup = _dict(cov.get("followup"))
    return Finding(
        code="DECOMP_COVERAGE_GAP" if _is_complex(report) else "COVERAGE_GAP",
        severity="critical" if blocked else "warning",
        title=(
            f"{len(unprocessed)} declared resource(s) were never processed"
            + (" — result withheld (evidence gate)" if blocked else "")
        ),
        evidence={
            "coverage.resources": len(_list(cov.get("resources"))),
            "coverage.touched": len(_list(cov.get("touched"))),
            "coverage.unprocessed": unprocessed[:10],
            "coverage.acknowledged": _list(cov.get("acknowledged"))[:10],
            "coverage.followup": followup or None,
            "coverage.blocked": blocked,
        },
        cause=(
            "The run (or its sub-tasks) finished without touching every "
            "resource the task declared, and the one bounded follow-up "
            "either did not run or could not close the gap."
        ),
        fix_now=(
            "Re-run with max_sub_agents=1 (SIMPLE path) or raise "
            "max_tool_calls; set evidence_gate_enforce=True to make the "
            "result abstain instead of silently omitting inputs."
        ),
        fix_release="Coverage reconciliation + resource gate (PR F / WS-4).",
    )


@rule
def handoff_truncated(report: RunReport) -> Finding | None:
    handoff = _dict(report.decisions.get("synthesis_handoff"))
    truncated = _list(handoff.get("truncated"))
    if not truncated:
        return None
    return Finding(
        code="HANDOFF_TRUNCATED",
        severity="warning",
        title=(
            f"{len(truncated)} sub-task result(s) exceeded the synthesis "
            f"hand-off cap of {handoff.get('per_finding_chars')} chars"
        ),
        evidence={
            "decisions.synthesis_handoff": handoff,
            "children[].refs": {
                str(c.get("id")): c.get("refs") for c in _business_children(report)
            },
        },
        cause=(
            "Child output was longer than the share of the parent window "
            "reserved for each finding; the tail was cut before synthesis."
        ),
        fix_now=(
            "Ask children for compact findings, declare a larger "
            "context_window, or set max_sub_agents=1."
        ),
        fix_release=(
            "Children merge evidence into the parent store so synthesis can "
            "recall_tool_result the full material (PR E / WS-3)."
        ),
    )


@rule
def decomp_downgraded(report: RunReport) -> Finding | None:
    cls = _classification(report)
    reason = str(cls.get("downgrade_reason") or "")
    if not reason:
        return None
    return Finding(
        code="DECOMP_DOWNGRADED",
        severity="info",
        title="Classifier proposed COMPLEX but the run was downgraded to SIMPLE",
        evidence={
            "decisions.classification.downgrade_reason": reason[:200],
            "decisions.classification.gates": cls.get("gates"),
            "decisions.classification.sub_tasks": cls.get("sub_tasks"),
        },
        cause="One of the decomposition gates (coverage, resources, budget) failed.",
        fix_now=(
            "Declare Task.resources so sub-tasks can be grounded, or accept "
            "the SIMPLE path (it is the safer default)."
        ),
        fix_release="Gate 4 and coverage contract (PR D).",
    )


@rule
def children_failed(report: RunReport) -> Finding | None:
    c = report.counters
    if c.children_failed <= 0:
        return None
    failed = []
    for ch in report.children:
        if str(ch.get("status", "")) in ("", "success"):
            continue
        item = {"id": ch.get("id"), "termination_reason": ch.get("termination_reason")}
        if ch.get("error"):
            item["error"] = str(ch.get("error"))[:200]
        failed.append(item)
    return Finding(
        code="CHILDREN_FAILED",
        severity="warning",
        title=f"{c.children_failed} of {c.children_spawned} sub-agent(s) did not succeed",
        evidence={
            "counters.children_failed": c.children_failed,
            "counters.children_spawned": c.children_spawned,
            "children[].termination_reason": failed,
        },
        cause="Sub-agents hit their own tool/time budgets or errored.",
        fix_now="Inspect the child termination reasons; raise their budgets or split less.",
        fix_release="Child findings carry status so synthesis can flag gaps (PR E).",
    )


# --------------------------------------------------------------------- #
# Budget / preflight rules                                               #
# --------------------------------------------------------------------- #


@rule
def preflight_unfit(report: RunReport) -> Finding | None:
    pre = _preflight(report)
    fitness = str(pre.get("fitness") or "")
    if fitness not in ("unfit", "marginal"):
        return None
    unfit = fitness == "unfit"
    return Finding(
        code="PREFLIGHT_UNFIT" if unfit else "PREFLIGHT_MARGINAL",
        severity="warning" if unfit else "info",
        title=(
            f"Preflight: {pre.get('working_tokens')} working tokens is "
            + ("below the Autonomous floor" if unfit else "marginal for Autonomous")
            + (
                f" — {pre.get('action')}"
                if pre.get("action") not in (None, "none")
                else ""
            )
        ),
        evidence={
            "preflight.fitness": fitness,
            "preflight.action": pre.get("action"),
            "preflight.window": pre.get("window"),
            "preflight.response_reserve": pre.get("response_reserve"),
            "preflight.system_tokens": pre.get("system_tokens"),
            "preflight.tool_schema_tokens": pre.get("tool_schema_tokens"),
            "preflight.working_tokens": pre.get("working_tokens"),
        },
        cause=(
            "System prompt + tool schemas + response reserve leave too little "
            "of the window for evidence and the Critic/Refiner hand-offs."
        ),
        fix_now=(
            "Use a model with a bigger window, register fewer tools, shorten "
            "the system prompt, or lower response_reserve / llm_max_output_tokens."
        ),
        fix_release="Preflight fitness check with downgrade (PR C §6.7).",
    )


@rule
def reserve_lt_max_tokens(report: RunReport) -> Finding | None:
    cfg = report.config_resolved
    reserve = _int(cfg.get("response_reserve"), 0)
    max_out = _int(cfg.get("llm_max_output_tokens"), 0)
    if reserve > 0 and max_out > reserve:
        return Finding(
            code="RESERVE_LT_MAX_TOKENS",
            severity="warning",
            title="llm_max_output_tokens exceeds the response reserve",
            evidence={
                "config_resolved.llm_max_output_tokens": max_out,
                "config_resolved.response_reserve": reserve,
                "config_resolved.context_window": cfg.get("context_window"),
            },
            cause=(
                "On shared-window servers (vLLM, Ollama) prompt + max_tokens "
                "must fit the window; the compactor only protects the reserve."
            ),
            fix_now="Set response_reserve >= llm_max_output_tokens, or lower the latter.",
            fix_release="BudgetResolver aligns the two (PR C §6.6).",
        )
    return None


@rule
def context_overflow(report: RunReport) -> Finding | None:
    c = report.counters
    if not _terminated(report, TerminationReason.CONTEXT_OVERFLOW):
        return None
    return Finding(
        code="CONTEXT_OVERFLOW_400",
        severity="critical",
        title="Provider rejected the request: prompt + max_tokens exceeded the model window",
        evidence={
            **_termination_evidence(report),
            "config_resolved.context_window": report.config_resolved.get(
                "context_window"
            ),
            "config_resolved.llm_max_output_tokens": report.config_resolved.get(
                "llm_max_output_tokens"
            ),
            "counters.max_prompt_tokens_seen": c.max_prompt_tokens_seen,
        },
        cause="The declared window is larger than the server's max-model-len, or the reserve is too small.",
        fix_now="Set context_window to the server's real limit and lower llm_max_output_tokens.",
        fix_release="ContextLengthError recovery compacts once and retries (PR C §6.4).",
    )


@rule
def tool_budget_exhausted(report: RunReport) -> Finding | None:
    c = report.counters
    if _terminated(report, TerminationReason.TOOL_BUDGET):
        return Finding(
            code="TOOL_BUDGET_EXHAUSTED",
            severity="warning",
            title=f"Tool-call budget ({report.config_resolved.get('max_tool_calls')}) exhausted",
            evidence={
                **_termination_evidence(report),
                "counters.tool_calls_business": c.tool_calls_business,
                "config_resolved.max_tool_calls": report.config_resolved.get(
                    "max_tool_calls"
                ),
                "config_resolved.tool_count": report.config_resolved.get("tool_count"),
            },
            cause="The task needed more calls than allowed, or calls were wasted on repeats.",
            fix_now="Raise max_tool_calls, or give each task fewer documents/resources.",
            fix_release="Budget is resolved once per run and inherited by children (PR C).",
        )
    if _terminated(report, TerminationReason.CONTEXT_TOOL_BUDGET):
        return Finding(
            code="CONTEXT_TOOL_BUDGET_EXHAUSTED",
            severity="warning",
            title="Context-management tool budget exhausted (recall/search loop)",
            evidence={
                **_termination_evidence(report),
                "counters.tool_calls_context": c.tool_calls_context,
                "counters.recall_errors": c.recall_errors,
                "config_resolved.max_context_tool_calls": report.config_resolved.get(
                    "max_context_tool_calls"
                ),
            },
            cause="The model kept recalling/searching instead of answering.",
            fix_now="Lower tool_result_per_call_max_chars so less is offloaded, or raise max_context_tool_calls.",
            fix_release="Cap introduced in PR A to guarantee termination.",
        )
    return None


@rule
def deadline(report: RunReport) -> Finding | None:
    if _terminated(report, TerminationReason.DEADLINE):
        return Finding(
            code="DEADLINE",
            severity="warning",
            title="Wall-clock budget (max_execution_time) exhausted",
            evidence={
                **_termination_evidence(report),
                "config_resolved.max_execution_time": report.config_resolved.get(
                    "max_execution_time"
                ),
                "counters.llm_calls": report.counters.llm_calls,
            },
            cause="Too many LLM/tool rounds for the allotted time, or a slow provider.",
            fix_now="Raise max_execution_time or split the task upstream.",
            fix_release="Wall clock is enforced at every loop (PR C §6.3).",
        )
    if _terminated(report, TerminationReason.LLM_TIMEOUT):
        return Finding(
            code="LLM_TIMEOUT",
            severity="warning",
            title="A single LLM call exceeded llm_call_timeout",
            evidence={
                **_termination_evidence(report),
                "config_resolved.llm_call_timeout": report.config_resolved.get(
                    "llm_call_timeout"
                ),
            },
            cause="Provider latency or an oversized prompt.",
            fix_now="Raise llm_call_timeout or reduce prompt size (smaller tool results).",
            fix_release="",
        )
    return None


# --------------------------------------------------------------------- #
# Quality / structured-output rules                                      #
# --------------------------------------------------------------------- #


@rule
def critic_abstain(report: RunReport) -> Finding | None:
    if not _terminated(report, TerminationReason.CRITIC_ABSTAIN):
        return None
    c = report.counters
    return Finding(
        code="CRITIC_ABSTAIN",
        severity="info",
        title="Quality gate refused every attempt — the agent abstained",
        evidence={
            **_termination_evidence(report),
            "counters.critic_verdicts": dict(c.critic_verdicts),
            "counters.escalations": c.escalations,
            "config_resolved.max_retries": report.config_resolved.get("max_retries"),
        },
        cause="The Critic found unresolved issues on every retry.",
        fix_now=(
            "Inspect the critique in AgentResult.abstention_reason; for pure "
            "extraction jobs consider require_quality_check=False."
        ),
        fix_release="",
    )


@rule
def synth_vs_structured(report: RunReport) -> Finding | None:
    cfg = report.config_resolved
    if cfg.get("response_format_set") and report.counters.synthesis_runs > 0:
        return Finding(
            code="SYNTH_VS_STRUCTURED",
            severity="warning",
            title="Prose synthesis pass ran although response_format was set",
            evidence={
                "config_resolved.response_format_set": True,
                "counters.synthesis_runs": report.counters.synthesis_runs,
                "counters.finalizer_runs": report.counters.finalizer_runs,
            },
            cause="The tools-free synthesis nudge asks for a 'full-length deliverable' in prose.",
            fix_now="Upgrade; or set enable_synthesis=False when using response_format.",
            fix_release="Structured runs use the schema finalizer instead of synthesis (PR B / WS-8).",
        )
    return None


@rule
def schema_not_satisfied(report: RunReport) -> Finding | None:
    so = _dict(report.decisions.get("structured_output"))
    invalid = so and so.get("valid") is False
    if not invalid and not _terminated(report, TerminationReason.SCHEMA_INVALID):
        return None
    return Finding(
        code="SCHEMA_NOT_SATISFIED",
        severity="warning",
        title="Final output does not validate against response_format",
        evidence={
            **_termination_evidence(report),
            "decisions.structured_output": so or None,
            "counters.schema_validation_failures": report.counters.schema_validation_failures,
            "counters.finalizer_runs": report.counters.finalizer_runs,
        },
        cause="The model emitted prose or a partial object and the finalizer could not repair it.",
        fix_now=(
            "Simplify the schema (fewer required fields), raise "
            "llm_max_output_tokens, or use a provider with constrained decoding."
        ),
        fix_release="Schema-aware Critic/Refiner and finalizer (PR B).",
    )


@rule
def empty_responses(report: RunReport) -> Finding | None:
    c = report.counters
    if c.empty_responses >= 2 or _terminated(report, TerminationReason.EMPTY_RESPONSE):
        return Finding(
            code="EMPTY_RESPONSES",
            severity="info",
            title=f"Model returned {c.empty_responses} empty response(s)",
            evidence={
                "counters.empty_responses": c.empty_responses,
                "termination.reason": report.termination.reason.value,
            },
            cause="Provider filtered the output, or max_tokens was hit before any text.",
            fix_now="Check provider logs; raise llm_max_output_tokens.",
            fix_release="",
        )
    return None


@rule
def gather_first_skipped(report: RunReport) -> Finding | None:
    gf = _dict(report.decisions.get("gather_first"))
    if not gf or gf.get("ran") or not gf.get("skipped"):
        return None
    return Finding(
        code="GATHER_SKIPPED",
        severity="info",
        title="decomposition_gather_first was on but the gather phase did not run",
        evidence={"decisions.gather_first": gf},
        cause=str(gf.get("skipped"))[:160],
        fix_now="Declare Task.resources and register at least one idempotent read tool.",
        fix_release="",
    )


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #


def analyze(report: RunReport) -> list[Finding]:
    """Evaluate every rule; never raises.  Sorted critical → warning → info."""
    findings: list[tuple[int, int, Finding]] = []
    for order, fn in enumerate(_RULES):
        try:
            finding = fn(report)
        except Exception:
            finding = None
        if finding is not None:
            findings.append((_SEVERITY_RANK.get(finding.severity, 9), order, finding))
    findings.sort(key=lambda t: (t[0], t[1]))
    return [f for _, _, f in findings]


def recommendations(findings: Iterable[Finding]) -> list[str]:
    """De-duplicated ``fix_now`` lines, in finding order."""
    seen: set[str] = set()
    out: list[str] = []
    for f in findings:
        text = (f.fix_now or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(f"[{f.code}] {text}")
    return out


def attach_findings(report: RunReport) -> RunReport:
    """Return ``report`` with ``findings`` and ``recommendations`` filled in."""
    found = analyze(report)
    return report.model_copy(
        update={
            "findings": tuple(found),
            "recommendations": tuple(recommendations(found)),
        }
    )


def load_report(source: str | dict[str, Any]) -> RunReport:
    """Build a :class:`RunReport` from a JSON string, a file path, or a dict."""
    import json
    import os

    if isinstance(source, dict):
        data = source
    else:
        text = source
        if os.path.exists(source):
            with open(source, encoding="utf-8") as fh:
                text = fh.read()
        data = json.loads(text)
    if isinstance(data, dict) and "diagnostics" in data and "termination" not in data:
        data = data["diagnostics"]  # an AgentResult.summary() was pasted
    return RunReport.model_validate(data)


__all__ = [
    "Rule",
    "analyze",
    "attach_findings",
    "load_report",
    "recommendations",
    "registered_rules",
    "rule",
]
