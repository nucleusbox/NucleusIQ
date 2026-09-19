"""
Decomposer Component — Task Analysis, Sub-Agent Orchestration, and Synthesis.

Handles the "complex task" path in autonomous mode:
1. Analyze task complexity (SIMPLE vs COMPLEX)
2. Spawn parallel sub-agents for independent sub-tasks
3. Summarize and synthesize sub-agent findings

Design:
    - Uses the framework's own Agent class for sub-agents (proven pattern)
    - Sub-agents run in STANDARD mode with isolated context
    - Limits (window, ``max_tool_calls``) come from the parent Agent
    - Findings are summarized before synthesis (context engineering)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.config.agent_config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.task import Task


def _parent_resolved_window(parent: Agent) -> int | None:
    """Window the parent already sized — never re-default to 8192."""
    engine = getattr(parent, "_context_engine", None)
    resolved = getattr(engine, "resolved_max_tokens", None)
    if isinstance(resolved, int) and resolved > 0:
        return resolved
    llm = getattr(parent, "llm", None)
    getter = getattr(llm, "get_context_window", None)
    if not callable(getter):
        return None
    try:
        raw = getter()
    except Exception:
        return None
    if isinstance(raw, (int, float)) and int(raw) > 0:
        return int(raw)
    return None


def _stamp_inherited_window(ctx: ContextConfig | None, window: int) -> ContextConfig:
    """Put the parent's resolved window on the child context config."""
    if ctx is None:
        return ContextConfig(max_context_tokens=window)
    if ctx.max_context_tokens is not None:
        return ctx
    return ctx.model_copy(update={"max_context_tokens": window})


def _sub_agent_config(
    parent: Agent,
    *,
    max_tool_calls_override: int | None = None,
    enable_synthesis_override: bool | None = None,
) -> AgentConfig:
    """STANDARD config that keeps the parent's context budget and limits.

    Sub-agents must not recurse into Autonomous COMPLEX, but they
    share the parent's model — so they must share its window and the
    builder's ``max_tool_calls``.  Prefer ``sub_agent_context`` when
    the builder set a child-specific budget; otherwise copy ``context``.
    ``max_tool_calls_override`` tightens (never loosens) the tool budget
    — used by the gather-first child.
    """
    parent_cfg = getattr(parent, "config", None)
    sub_ctx = None
    respect = True
    llm_max_output_tokens = 2048
    enable_synthesis = True
    max_tool_calls = None
    max_retries = 3

    if isinstance(parent_cfg, AgentConfig):
        source = (
            parent_cfg.sub_agent_context
            if parent_cfg.sub_agent_context is not None
            else parent_cfg.context
        )
        if source is not None:
            sub_ctx = source.model_copy()
        respect = bool(parent_cfg.respect_context_window)
        llm_max_output_tokens = int(parent_cfg.llm_max_output_tokens)
        enable_synthesis = bool(parent_cfg.enable_synthesis)
        max_tool_calls = parent_cfg.max_tool_calls
        max_retries = int(parent_cfg.max_retries)

    if sub_ctx is None or sub_ctx.max_context_tokens is None:
        inherited_window = _parent_resolved_window(parent)
        if inherited_window is not None:
            sub_ctx = _stamp_inherited_window(sub_ctx, inherited_window)

    extra: dict[str, Any] = {}
    # Children get the parent's *remaining* wall clock, never a fresh hour.
    from nucleusiq.agents.modes.loop_guards import remaining_seconds

    left = remaining_seconds(parent)
    if left is not None:
        extra["max_execution_time"] = max(1, int(left))
    elif isinstance(parent_cfg, AgentConfig):
        extra["max_execution_time"] = int(parent_cfg.max_execution_time)
    if isinstance(parent_cfg, AgentConfig):
        explicit = getattr(parent_cfg, "model_fields_set", set())
        for field in ("llm_call_timeout", "step_timeout"):
            if field in explicit:
                extra[field] = getattr(parent_cfg, field)

    if max_tool_calls_override is not None:
        override = max(1, int(max_tool_calls_override))
        max_tool_calls = (
            override if max_tool_calls is None else min(int(max_tool_calls), override)
        )
    if enable_synthesis_override is not None:
        enable_synthesis = bool(enable_synthesis_override)

    return AgentConfig(
        execution_mode=ExecutionMode.STANDARD,
        respect_context_window=respect,
        context=sub_ctx,
        llm_max_output_tokens=llm_max_output_tokens,
        enable_synthesis=enable_synthesis,
        max_tool_calls=max_tool_calls,
        max_retries=max_retries,
        **extra,
    )


@dataclass
class TaskAnalysis:
    """Result of task complexity analysis."""

    is_complex: bool
    sub_tasks: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    #: Gate vector as the classifier answered it (``gate1``..``gate4``).
    gates: dict[str, bool] = field(default_factory=dict)
    #: Why a COMPLEX claim was overridden to SIMPLE (empty when not).
    downgrade_reason: str = ""
    #: Resources the classification was grounded on.
    resources: list[str] = field(default_factory=list)

    def to_decision(self) -> dict[str, Any]:
        """Summary for ``RunReport.decisions["classification"]`` (no task text)."""
        out: dict[str, Any] = {
            "is_complex": bool(self.is_complex),
            "classifier_called": True,
            "sub_tasks": len(self.sub_tasks),
            "reasoning": (self.reasoning or "")[:300],
        }
        if self.gates:
            out["gates"] = dict(self.gates)
        if self.downgrade_reason:
            out["downgrade_reason"] = self.downgrade_reason[:300]
        if self.resources:
            out["resources"] = len(self.resources)
            out["sub_task_resources"] = [
                len(st.get("resources") or []) for st in self.sub_tasks
            ]
        return out


@dataclass
class SubTaskFinding:
    """Structured hand-off from one child to the parent's synthesis (WS-3).

    ``result`` is the child's answer text; ``refs`` are the child's
    offloaded tool results (recallable by the parent after the merge);
    ``touched_resources`` feeds coverage reconciliation.  ``to_dict()``
    keeps the historical ``{"id", "objective", "result"}`` shape so every
    consumer of ``run_sub_tasks`` output keeps working.
    """

    id: str
    objective: str
    result: str = ""
    status: str = ""
    termination_reason: str = ""
    error: str = ""
    resources: list[str] = field(default_factory=list)
    touched_resources: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    merged: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "error" or self.result.startswith("Error:")

    def absorb_result(self, raw: Any) -> None:
        """Fill ``result`` / ``status`` / ``termination_reason`` / ``error``
        from a child's ``AgentResult`` (or any object the child returned).

        A child that ended in ``error`` has an empty ``output`` — ``str(raw)``
        would be ``""`` and the reason would be lost.  The error text is kept
        so synthesis can state the gap and the run report can explain why the
        child failed (``CHILDREN_FAILED`` evidence).
        """
        raw_status = getattr(raw, "status", None)
        if raw_status is not None:
            self.status = str(getattr(raw_status, "value", raw_status))
        self.termination_reason = str(getattr(raw, "termination_reason", "") or "")
        err = getattr(raw, "error", None)
        if err:
            self.error = str(err)
        text = str(raw)
        if not text.strip() and self.error:
            text = f"Error: {self.error}"
        self.result = text

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "objective": self.objective,
            "result": self.result,
        }
        if self.status:
            out["status"] = self.status
        if self.termination_reason:
            out["termination_reason"] = self.termination_reason
        if self.error:
            out["error"] = self.error
        if self.resources:
            out["resources"] = list(self.resources)
        if self.touched_resources:
            out["touched_resources"] = list(self.touched_resources)
        if self.refs:
            out["refs"] = list(self.refs)
        if self.merged:
            out["merged"] = dict(self.merged)
        return out


#: Resources listed in the classifier prompt; more are summarised as a count.
_MAX_GROUNDING_RESOURCES = 40
#: Corpus titles / tool names listed in the classifier prompt.
_MAX_GROUNDING_ITEMS = 25
#: Default owner cap for the coverage contract (``≤ k`` sub-tasks per resource).
DEFAULT_MAX_OWNERS_PER_RESOURCE = 1


def _norm_resource(value: Any) -> str:
    """Canonical form for resource matching (case/whitespace/slash-insensitive)."""
    text = str(value or "").strip().lower().replace("\\", "/")
    return text.rstrip("/")


def _resource_matches(claimed: str, declared: str) -> bool:
    """A sub-task claim covers a declared resource when it names it exactly
    or by its trailing path component (``report.pdf`` ↔ ``docs/report.pdf``)."""
    c, d = _norm_resource(claimed), _norm_resource(declared)
    if not c or not d:
        return False
    if c == d:
        return True
    return d.endswith("/" + c) or c.endswith("/" + d)


class Decomposer:
    """Analyzes task complexity and orchestrates parallel sub-agents.

    Responsibilities (SRP):
    - Classify tasks as SIMPLE or COMPLEX via a single LLM call
    - Create isolated sub-agents using the framework's Agent class
    - Run sub-agents in parallel and collect summarized findings
    - Synthesize findings into a single coherent result

    The Decomposer does NOT handle verification or revision — that
    remains the Critic/Refiner's job in AutonomousMode.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._sub_agent_results: list[Any] = []

    # ------------------------------------------------------------------ #
    # Task Analysis                                                        #
    # ------------------------------------------------------------------ #

    async def analyze(self, agent: Agent, task: Task) -> TaskAnalysis:
        """Classify a task as SIMPLE or COMPLEX via one LLM call.

        Uses a 4-gate checklist: the task is COMPLEX only when ALL
        gates evaluate to true.  Gate 4 (separable sources / not a
        single record) is the grounding gate: with ``Task.resources``
        present the classifier sees the real resource list and every
        sub-task must claim a slice; the coverage contract in
        :meth:`_parse_analysis` then verifies the split.  Falls back
        to SIMPLE on any error (safe default).

        Records the LLM call on the agent's tracer (this method calls
        ``agent.llm.call()`` directly, bypassing ``call_llm()``).
        """
        import time as _time

        resources = task.effective_resources()
        prompt = self.build_classifier_prompt(
            task.objective,
            resources=resources,
            corpus_titles=self._corpus_titles(agent),
            tool_names=self._tool_names(agent),
        )
        assert agent.llm is not None, "agent.llm must be set for Decomposer"
        model_name = getattr(agent.llm, "model_name", "default")
        token_budget = agent.config.llm_max_output_tokens
        t0 = _time.perf_counter()
        try:
            response = await agent.llm.call(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_output_tokens=token_budget,
            )
            self._record_llm_call(agent, t0, response, purpose="decomposer")
            return self._parse_analysis(
                response,
                resources=resources,
                max_owners=self._max_owners(agent),
            )
        except Exception as e:
            self._logger.warning(
                "Task analysis failed: %s — defaulting to SIMPLE",
                e,
            )
            self._record_llm_call(agent, t0, None, purpose="decomposer")
            return TaskAnalysis(
                is_complex=False,
                reasoning=f"Analysis error: {e}",
                resources=resources,
            )

    # ------------------------------------------------------------------ #
    # Grounding inputs                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _corpus_titles(agent: Agent) -> list[str]:
        corpus = getattr(agent, "_document_corpus", None)
        lister = getattr(corpus, "list_documents", None)
        if not callable(lister):
            return []
        try:
            docs = lister()
        except Exception:
            return []
        titles: list[str] = []
        for doc in docs:
            title = getattr(doc, "title", None) or getattr(doc, "id", None)
            if isinstance(title, str) and title.strip():
                titles.append(title.strip())
        return titles

    @staticmethod
    def _tool_names(agent: Agent) -> list[str]:
        from nucleusiq.agents.context.workspace_tools import (
            is_context_management_tool_name,
        )

        names: list[str] = []
        for tool in getattr(agent, "tools", None) or []:
            name = getattr(tool, "name", None)
            if (
                isinstance(name, str)
                and name
                and not is_context_management_tool_name(name)
            ):
                names.append(name)
        return names

    @staticmethod
    def _max_owners(agent: Agent) -> int:
        value = getattr(
            getattr(agent, "config", None),
            "decomposition_max_owners_per_resource",
            DEFAULT_MAX_OWNERS_PER_RESOURCE,
        )
        return max(1, int(value)) if isinstance(value, int) else 1

    @staticmethod
    def build_classifier_prompt(
        task_objective: str,
        *,
        resources: list[str] | None = None,
        corpus_titles: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> str:
        """The grounded 4-gate classifier prompt (pure; testable)."""
        resources = list(resources or [])
        corpus_titles = list(corpus_titles or [])
        tool_names = list(tool_names or [])

        grounding = ""
        if resources:
            shown = resources[:_MAX_GROUNDING_RESOURCES]
            items = "\n".join(f"- {r}" for r in shown)
            if len(resources) > len(shown):
                items += f"\n- … and {len(resources) - len(shown)} more"
            grounding += (
                f"## RESOURCES THE TASK MUST COVER ({len(resources)})\n{items}\n\n"
            )
        if corpus_titles:
            shown = corpus_titles[:_MAX_GROUNDING_ITEMS]
            grounding += (
                "## DOCUMENTS ALREADY INDEXED\n"
                + "\n".join(f"- {t}" for t in shown)
                + "\n\n"
            )
        if tool_names:
            shown = tool_names[:_MAX_GROUNDING_ITEMS]
            grounding += "## AVAILABLE TOOLS\n" + ", ".join(shown) + "\n\n"

        if resources:
            sub_task_shape = (
                '{"id": "sub1", "objective": "...", '
                '"resources": ["<exact resource ids from the list>"]}'
            )
            resource_rule = (
                "- With resources present, EVERY sub-task MUST list the "
                'resources it owns under "resources" (exact ids from the '
                "list above). Every resource must be owned by exactly one "
                "sub-task; a sub-task with no resources is invalid.\n"
            )
        else:
            sub_task_shape = '{"id": "sub1", "objective": "..."}'
            resource_rule = ""

        return (
            "You are a task classifier. Evaluate the following task "
            "against four gate conditions to determine whether it "
            "should be handled by a single agent (SIMPLE) or split "
            "into parallel sub-agents (COMPLEX).\n\n"
            f"Task: {task_objective}\n\n"
            f"{grounding}"
            "## FOUR-GATE CHECKLIST\n\n"
            "Answer each gate true or false:\n\n"
            "GATE 1 — MULTIPLE SUB-TOPICS: Does the task contain "
            "two or more distinct sub-topics or entities to "
            "investigate?\n\n"
            "GATE 2 — INDEPENDENCE: Can each sub-topic be fully "
            "completed WITHOUT needing results from the others? "
            "(If step B requires output of step A, answer false.)\n\n"
            "GATE 3 — MERGE-SAFE: Can the sub-topic results be "
            "combined at the end to produce the final answer "
            "without losing accuracy?\n\n"
            "GATE 4 — SEPARABLE SOURCES, SEPARATE OUTPUTS: Can the work be "
            "split so that each source (document, file, record, dataset) "
            "is read by exactly ONE sub-task, AND the deliverable is not a "
            "single record / single document that every sub-task would "
            "have to fill in together? If every sub-task would need the "
            "same sources, or the output is one record, answer false.\n\n"
            "## DECISION RULE\n\n"
            "- If ALL four gates are true → COMPLEX.\n"
            "- If ANY gate is false → SIMPLE.\n"
            f"{resource_rule}\n"
            "Respond with ONLY this JSON:\n"
            '{"gate1": true/false, "gate2": true/false, '
            '"gate3": true/false, "gate4": true/false, "complexity": "simple"}\n'
            "or\n"
            '{"gate1": true, "gate2": true, "gate3": true, "gate4": true, '
            f'"complexity": "complex", "sub_tasks": [{sub_task_shape}, ...]}}\n'
        )

    @staticmethod
    def _record_llm_call(
        agent: Agent,
        t0: float,
        response: Any,
        purpose: str = "decomposer",
    ) -> None:
        """Record this direct LLM call on the agent's tracer."""
        import time as _time

        from nucleusiq.agents.agent_result import LLMCallRecord

        tracer = getattr(agent, "_tracer", None)
        if tracer is None:
            return
        try:
            total_tokens = 0
            if response is not None:
                usage = getattr(response, "usage_metadata", None) or getattr(
                    response, "usage", None
                )
                if usage:
                    total_tokens = getattr(usage, "total_token_count", 0) or getattr(
                        usage, "total_tokens", 0
                    )
            tracer.record_llm_call(
                LLMCallRecord(
                    round=len(tracer.llm_calls) + 1,
                    model=getattr(agent.llm, "model_name", "unknown"),
                    purpose=purpose,
                    total_tokens=total_tokens,
                    duration_ms=(_time.perf_counter() - t0) * 1000,
                )
            )
        except Exception:
            pass

    def _parse_analysis(
        self,
        response: Any,
        *,
        resources: list[str] | None = None,
        max_owners: int = DEFAULT_MAX_OWNERS_PER_RESOURCE,
    ) -> TaskAnalysis:
        """Parse the LLM's complexity classification and enforce the contract.

        SIMPLE is always a correct answer; COMPLEX is an optimisation
        that must earn its keep: all gates true, at least two
        well-formed sub-tasks, and — when the task names resources —
        a coverage contract: every declared resource is owned by at
        least one sub-task, no sub-task is resource-less, and no
        resource has more than ``max_owners`` owners.  Any violation
        downgrades to SIMPLE with the reason recorded.
        """
        resources = list(resources or [])
        content = ""
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            content = (
                msg.get("content", "")
                if isinstance(msg, dict)
                else getattr(msg, "content", "") or ""
            )

        try:
            json_match = json.loads(
                content[content.index("{") : content.rindex("}") + 1]
            )
        except (json.JSONDecodeError, ValueError):
            return TaskAnalysis(
                is_complex=False, reasoning="Could not parse", resources=resources
            )
        if not isinstance(json_match, dict):
            return TaskAnalysis(
                is_complex=False, reasoning="Could not parse", resources=resources
            )

        gates = {
            "gate1": bool(json_match.get("gate1", False)),
            "gate2": bool(json_match.get("gate2", False)),
            "gate3": bool(json_match.get("gate3", False)),
            # Gate 4 was added later; a reply that omits it is judged on
            # the original three gates so older-style answers still work.
            "gate4": bool(json_match.get("gate4", True)),
        }
        all_gates_pass = all(gates.values())

        claimed_complex = json_match.get("complexity", "simple") == "complex"
        raw_sub_tasks = json_match.get("sub_tasks", [])
        sub_tasks = self._normalize_sub_tasks(raw_sub_tasks)
        reasoning = str(json_match.get("reasoning", "") or "")

        downgrade = ""
        if claimed_complex and not all_gates_pass:
            failed = ", ".join(k for k, v in gates.items() if not v)
            downgrade = f"gates failed: {failed}"
        elif claimed_complex and len(sub_tasks) < 2:
            downgrade = f"only {len(sub_tasks)} well-formed sub-task(s)"
        elif claimed_complex and resources:
            downgrade = self._check_coverage_contract(
                sub_tasks, resources, max_owners=max_owners
            )

        is_complex = claimed_complex and not downgrade

        if downgrade:
            self._logger.info(
                "LLM claimed COMPLEX but %s — overriding to SIMPLE", downgrade
            )

        return TaskAnalysis(
            is_complex=is_complex,
            sub_tasks=sub_tasks if is_complex else [],
            reasoning=reasoning,
            gates=gates,
            downgrade_reason=downgrade,
            resources=resources,
        )

    @staticmethod
    def _normalize_sub_tasks(raw: Any) -> list[dict[str, Any]]:
        """Keep only well-formed ``{"id", "objective"[, "resources"]}`` entries."""
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            objective = str(item.get("objective", "") or "").strip()
            if not objective:
                continue
            entry: dict[str, Any] = {
                "id": str(item.get("id") or f"sub{index + 1}"),
                "objective": objective,
            }
            claimed = item.get("resources")
            if isinstance(claimed, str):
                claimed = [claimed]
            if isinstance(claimed, list):
                entry["resources"] = [
                    str(r).strip() for r in claimed if str(r or "").strip()
                ]
            out.append(entry)
        return out

    @staticmethod
    def _check_coverage_contract(
        sub_tasks: list[dict[str, Any]],
        resources: list[str],
        *,
        max_owners: int = DEFAULT_MAX_OWNERS_PER_RESOURCE,
    ) -> str:
        """Return a violation description, or ``""`` when the split is sound.

        Rules (design WS-2 item 5): ``∪ sub.resources ⊇ resources``, no
        sub-task without resources, each resource owned by ≤ ``max_owners``.
        """
        owners: dict[str, list[str]] = {r: [] for r in resources}
        for st in sub_tasks:
            claimed = st.get("resources") or []
            if not claimed:
                return f"sub-task {st.get('id')!r} claims no resources"
            for claim in claimed:
                for declared in resources:
                    if _resource_matches(claim, declared):
                        owners[declared].append(str(st.get("id")))
        missing = [r for r, o in owners.items() if not o]
        if missing:
            shown = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            return (
                f"coverage contract: {len(missing)} resource(s) unowned: {shown}{more}"
            )
        shared = {r: o for r, o in owners.items() if len(set(o)) > max(1, max_owners)}
        if shared:
            first, its_owners = next(iter(shared.items()))
            return (
                f"coverage contract: {len(shared)} resource(s) shared by several "
                f"sub-tasks (e.g. {first!r} → {sorted(set(its_owners))}); each "
                f"resource may have at most {max_owners} owner(s)"
            )
        return ""

    # ------------------------------------------------------------------ #
    # Sub-Agent Creation                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def create_sub_agent(
        parent: Agent,
        sub_task_id: str,
        sub_task_objective: str,
        *,
        parent_view: Any = None,
        tools: list[Any] | None = None,
        max_tool_calls: int | None = None,
        system_suffix: str = "",
        enable_synthesis: bool | None = None,
    ) -> Agent | None:
        """Create an isolated sub-agent using the framework's Agent class.

        Shares the parent's LLM and tools but has its own:
        - Config (STANDARD mode, inherited context budget and limits)
        - Isolated plugin state
        - No memory (isolated context, no cross-contamination)
        - Local evidence stores that *read through* to ``parent_view``
          (the parent's store / corpus / dossier) so a child never
          re-fetches what the parent or a finished sibling already has.

        The context window and ``max_tool_calls`` are inherited from
        the parent (or ``AgentConfig.sub_agent_context`` for the
        window). A fresh ``AgentConfig(STANDARD)`` plus a hardcoded
        15-call plugin used to drop both, so a 65K / 40-call parent
        spawned an 8K / 15-call child.

        Returns None if creation fails (graceful degradation).
        """
        from nucleusiq.agents.agent import Agent

        try:
            from nucleusiq.prompts import PromptFactory, PromptTechnique

            system = (
                f"You are a focused sub-analyst for {parent.name}. "
                f"Role: {parent.role}. Complete only your assigned sub-task."
            )
            hint_fn = getattr(parent_view, "system_hint", None)
            hint = hint_fn() if callable(hint_fn) else ""
            if hint:
                system = f"{system} {hint}"
            if system_suffix:
                system = f"{system} {system_suffix.strip()}"
            sub_prompt = PromptFactory.create_prompt(
                technique=PromptTechnique.ZERO_SHOT
            ).configure(system=system)
            sub = Agent(
                name=f"{parent.name}-sub-{sub_task_id}",
                role=parent.role,
                objective=sub_task_objective,
                prompt=sub_prompt,
                llm=parent.llm,
                tools=list(tools if tools is not None else parent.tools),
                memory=None,
                config=_sub_agent_config(
                    parent,
                    max_tool_calls_override=max_tool_calls,
                    enable_synthesis_override=enable_synthesis,
                ),
            )
            if parent_view is not None:
                sub._parent_evidence = parent_view
            await sub.initialize()
            return sub
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Sub-agent creation failed for %s: %s",
                sub_task_id,
                e,
            )
            return None

    # ------------------------------------------------------------------ #
    # Parallel Execution                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_child_task(
        parent_task: Task | None,
        sub_task: dict[str, Any],
    ) -> Task:
        """The ``Task`` a child executes.

        Children see what the user attached to the parent task —
        ``attachments``, ``context`` (minus the resources key) and
        ``metadata`` — plus their own ``resources`` slice.  A child with
        no declared slice inherits the parent's full resource list so
        an ungrounded split still tells the child what must be covered.
        """
        sub_id = str(sub_task.get("id") or "sub")
        objective = str(sub_task.get("objective") or "")
        claimed = [
            str(r).strip() for r in (sub_task.get("resources") or []) if str(r).strip()
        ]
        if parent_task is None:
            return Task(id=sub_id, objective=objective, resources=claimed or None)

        resources = claimed or parent_task.effective_resources()
        context = parent_task.context_without_resources() or None
        return Task(
            id=sub_id,
            objective=objective,
            context=context,
            metadata=dict(parent_task.metadata) if parent_task.metadata else None,
            attachments=list(parent_task.attachments)
            if parent_task.attachments
            else None,
            resources=resources or None,
        )

    async def run_sub_tasks(
        self,
        parent: Agent,
        sub_tasks: list[dict[str, Any]],
        max_sub_agents: int = 5,
        *,
        parent_task: Task | None = None,
    ) -> list[dict[str, Any]]:
        """Run sub-tasks in parallel via isolated sub-agents.

        Each sub-agent runs independently with its own context.
        Results are collected and returned as a list of findings.
        Failed sub-agents are logged and skipped (partial results
        are better than total failure).

        Args:
            parent: The parent agent (provides LLM, tools, role).
            sub_tasks: List of {"id": ..., "objective": ...[, "resources"]} dicts.
            max_sub_agents: Cap on parallel sub-agents.
            parent_task: The user's task; its attachments / context /
                metadata and each sub-task's resources slice are handed
                to the children (see :meth:`build_child_task`).

        Returns:
            List of {"id": ..., "objective": ..., "result": ...[, "resources"]} dicts.
        """
        capped = sub_tasks[:max_sub_agents]
        self._logger.info(
            "Decomposing into %d sub-tasks (cap=%d)",
            len(capped),
            max_sub_agents,
        )

        parent_view = self._parent_view(parent)
        agents_and_tasks = []
        for st in capped:
            sub_agent = await self.create_sub_agent(
                parent,
                st["id"],
                st["objective"],
                parent_view=parent_view,
            )
            if sub_agent:
                agents_and_tasks.append((sub_agent, st))
            else:
                self._logger.warning(
                    "Skipping sub-task %s (agent creation failed)", st["id"]
                )

        if not agents_and_tasks:
            return []

        collected_results: list[Any] = []

        async def _execute_one(agent: Agent, st: dict[str, Any]) -> dict[str, Any]:
            child_task = self.build_child_task(parent_task, st)
            finding = SubTaskFinding(
                id=str(st["id"]),
                objective=str(st["objective"]),
                resources=list(child_task.resources or []),
            )
            try:
                raw = await agent.execute(child_task)
                collected_results.append(raw)
                finding.absorb_result(raw)
            except Exception as e:
                self._logger.warning("Sub-task %s failed: %s", st["id"], e)
                finding.result = f"Error: {e}"
                finding.error = str(e)
                finding.status = "error"
            if finding.failed:
                self._logger.warning(
                    "Sub-task %s ended %s: %s",
                    st["id"],
                    finding.termination_reason or finding.status,
                    finding.error or finding.result[:200],
                )
            self._absorb_child(parent, agent, finding)
            return finding.to_dict()

        findings = await asyncio.gather(
            *[_execute_one(agent, st) for agent, st in agents_and_tasks]
        )
        self._sub_agent_results = collected_results
        return list(findings)

    # ------------------------------------------------------------------ #
    # Gather-first (WS-3, opt-in)                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_gather_objective(resources: list[str]) -> str:
        """Objective for the read-only gather child."""
        listed = "\n".join(f"- {r}" for r in resources[:_MAX_GROUNDING_RESOURCES])
        more = (
            f"\n- … and {len(resources) - _MAX_GROUNDING_RESOURCES} more "
            "(see the Resources list in the task context)"
            if len(resources) > _MAX_GROUNDING_RESOURCES
            else ""
        )
        return (
            "GATHER PHASE. Read every resource listed below exactly once using "
            "the available read/fetch/search tools. Do NOT analyse, summarise, "
            "compare or answer anything — the analysis happens later with the "
            "material you collect. When every resource has been read (or a read "
            "failed after one retry), reply with one line per resource: "
            "'<resource>: read' or '<resource>: failed (<reason>)'.\n\n"
            f"Resources:\n{listed}{more}"
        )

    @staticmethod
    def gather_tools(parent: Agent) -> list[Any]:
        """Tools the gather child may use: the parent's idempotent ones."""
        return [
            t
            for t in (getattr(parent, "tools", None) or [])
            if bool(getattr(t, "idempotent", False))
        ]

    async def run_gather_phase(
        self,
        parent: Agent,
        parent_task: Task | None,
    ) -> dict[str, Any]:
        """Opt-in pre-pass: one child reads every resource into the parent's
        stores so the analysis children start from shared evidence instead
        of each re-fetching the same documents.

        Bounded by construction: idempotent tools only, at most
        ``2 × len(resources)`` tool calls (never more than the parent's own
        budget), and the parent's remaining wall clock.  Returns a summary
        dict for the run report; never raises.
        """
        resources = parent_task.effective_resources() if parent_task else []
        summary: dict[str, Any] = {
            "ran": False,
            "resources": len(resources),
            "touched": [],
            "unprocessed": list(resources),
        }
        if not resources:
            summary["skipped"] = "no resources declared"
            return summary
        tools = self.gather_tools(parent)
        if not tools:
            summary["skipped"] = "no idempotent tools available"
            return summary

        cap = 2 * len(resources)
        summary["max_tool_calls"] = cap
        finding = await self._run_aux_child(
            parent,
            parent_task,
            child_id="gather",
            objective=self.build_gather_objective(resources),
            resources=resources,
            tools=tools,
            max_tool_calls=cap,
            system_suffix="You only collect material; never analyse or conclude.",
            # A collector has nothing to synthesise — skip the extra pass.
            enable_synthesis=False,
        )
        if finding is None:
            summary["skipped"] = "gather agent creation failed"
            return summary

        tracker = getattr(parent, "_resource_tracker", None)
        summary.update(
            {
                "ran": True,
                "status": finding.status,
                "termination_reason": finding.termination_reason,
                "touched": list(getattr(tracker, "touched", finding.touched_resources)),
                "unprocessed": list(
                    getattr(tracker, "unprocessed", [])
                    if tracker is not None
                    else [r for r in resources if r not in finding.touched_resources]
                ),
                "refs": len(finding.refs),
                "merged": finding.merged,
            }
        )
        return summary

    # ------------------------------------------------------------------ #
    # Coverage follow-up (WS-4)                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_followup_objective(parent_objective: str, unprocessed: list[str]) -> str:
        """Objective for the bounded coverage follow-up child."""
        listed = "\n".join(f"- {r}" for r in unprocessed[:_MAX_GROUNDING_RESOURCES])
        more = (
            f"\n- … and {len(unprocessed) - _MAX_GROUNDING_RESOURCES} more"
            if len(unprocessed) > _MAX_GROUNDING_RESOURCES
            else ""
        )
        return (
            "COVERAGE FOLLOW-UP. The resources listed below were declared for "
            "the task but were never processed by the earlier sub-tasks. Process "
            "ONLY these resources, fully, in the spirit of the original "
            f"objective:\n\n{parent_objective.strip()}\n\n"
            "For each resource report what you found, or "
            "'<resource>: could not be processed (<reason>)'. Do not redo work "
            "on resources that are not listed.\n\n"
            f"Unprocessed resources:\n{listed}{more}"
        )

    async def run_coverage_followup(
        self,
        parent: Agent,
        parent_task: Task,
        unprocessed: list[str],
    ) -> dict[str, Any] | None:
        """One bounded child that processes only the resources nobody touched.

        Same shape as any other finding so synthesis renders it; capped at
        ``2 × len(unprocessed)`` tool calls (never above the parent's budget).
        Returns ``None`` when the child could not be created.
        """
        pending = [str(r).strip() for r in unprocessed if str(r).strip()]
        if not pending:
            return None
        finding = await self._run_aux_child(
            parent,
            parent_task,
            child_id="coverage-followup",
            objective=self.build_followup_objective(parent_task.objective, pending),
            resources=pending,
            tools=list(getattr(parent, "tools", None) or []),
            max_tool_calls=2 * len(pending),
            system_suffix=(
                "You cover only the resources you were handed; be explicit about "
                "any you could not process."
            ),
        )
        return finding.to_dict() if finding is not None else None

    @staticmethod
    def _aux_tool_budget(tools: list[Any], max_tool_calls: int) -> int:
        """Tool-call budget for an auxiliary child that its own preflight
        will accept.

        ``Agent.execute`` treats ``max_tool_calls`` as an upper bound on the
        number of *user* tools an agent may carry (see the tool-count
        validation there).  A follow-up over four resources gets a budget of
        eight calls — but if the parent hands it twenty business tools the
        child dies before its first LLM call with "has 20 tools but STANDARD
        mode allows max 8".  The budget must never be smaller than the tool
        list the child was given; the context-management tools the child
        adds itself are exempt from that count.
        """
        try:
            from nucleusiq.agents.context.workspace_tools import (
                is_context_management_tool_name,
            )

            user_tools = sum(
                1
                for t in tools
                if not is_context_management_tool_name(getattr(t, "name", None))
            )
        except Exception:
            user_tools = len(tools)
        return max(1, max_tool_calls, user_tools)

    async def _run_aux_child(
        self,
        parent: Agent,
        parent_task: Task | None,
        *,
        child_id: str,
        objective: str,
        resources: list[str],
        tools: list[Any],
        max_tool_calls: int,
        system_suffix: str = "",
        enable_synthesis: bool | None = None,
    ) -> SubTaskFinding | None:
        """Spawn one auxiliary child (gather / follow-up), run it on a Task
        derived from the parent's, merge its evidence back and return the
        finding.  Never raises; ``None`` only when creation failed."""
        child = await self.create_sub_agent(
            parent,
            child_id,
            objective,
            parent_view=self._parent_view(parent),
            tools=tools,
            max_tool_calls=self._aux_tool_budget(tools, max_tool_calls),
            system_suffix=system_suffix,
            enable_synthesis=enable_synthesis,
        )
        if child is None:
            return None

        child_task = Task(
            id=f"{parent_task.id}-{child_id}" if parent_task else child_id,
            objective=objective,
            context=parent_task.context_without_resources() or None
            if parent_task
            else None,
            metadata=dict(parent_task.metadata)
            if parent_task and parent_task.metadata
            else None,
            attachments=list(parent_task.attachments)
            if parent_task and parent_task.attachments
            else None,
            resources=list(resources) or None,
        )
        finding = SubTaskFinding(
            id=child_id, objective=objective, resources=list(resources)
        )
        try:
            raw = await child.execute(child_task)
            finding.absorb_result(raw)
            self._sub_agent_results.append(raw)
        except Exception as exc:
            self._logger.warning("Auxiliary child %s failed: %s", child_id, exc)
            finding.result = f"Error: {exc}"
            finding.error = str(exc)
            finding.status = "error"
        if finding.failed:
            self._logger.warning(
                "Child %s ended %s: %s",
                child_id,
                finding.termination_reason or finding.status,
                finding.error or finding.result[:200],
            )
        self._absorb_child(parent, child, finding)
        return finding

    # ------------------------------------------------------------------ #
    # Shared evidence (WS-3)                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parent_view(parent: Agent) -> Any:
        """Read-only view of the parent's evidence for its children."""
        try:
            from nucleusiq.agents.context.shared_evidence import ParentEvidenceView

            tracker = getattr(parent, "_resource_tracker", None)
            prefetched = list(getattr(tracker, "touched", []) or [])
            return ParentEvidenceView.from_agent(parent, prefetched=prefetched)
        except Exception:
            return None

    def _absorb_child(
        self, parent: Agent, child: Agent, finding: SubTaskFinding
    ) -> None:
        """Merge a finished child's evidence into the parent and fill the
        finding's ``refs`` / ``touched_resources`` from the child's stores."""
        try:
            from nucleusiq.agents.context.shared_evidence import (
                LayeredContentStore,
                merge_child_evidence,
            )

            store = getattr(getattr(child, "_context_engine", None), "store", None)
            if isinstance(store, LayeredContentStore):
                finding.refs = store.local_keys()
            elif store is not None and hasattr(store, "keys"):
                finding.refs = list(store.keys())

            tracker = getattr(child, "_resource_tracker", None)
            touched = list(getattr(tracker, "touched", []) or [])
            finding.touched_resources = touched

            report = merge_child_evidence(parent, child, label=finding.id)
            if report.store_entries or report.documents or report.evidence_items:
                finding.merged = report.to_dict()

            parent_tracker = getattr(parent, "_resource_tracker", None)
            if parent_tracker is not None and touched:
                parent_tracker.mark(touched, via=f"child:{finding.id}")
        except Exception as exc:
            self._logger.debug(
                "Child evidence merge skipped for %s: %s", finding.id, exc
            )
        self._record_child(parent, child, finding)

    @staticmethod
    def _record_child(parent: Agent, child: Agent, finding: SubTaskFinding) -> None:
        """Put the child's resolved config + counters in the parent's report
        so the analyzer can spot window / tool-budget mismatches (WS-6)."""
        recorder = getattr(parent, "_run_recorder", None)
        if recorder is None or not hasattr(recorder, "record_child"):
            return
        try:
            info: dict[str, Any] = {
                "id": finding.id,
                "name": str(getattr(child, "name", "")),
                "status": finding.status,
                "termination_reason": finding.termination_reason,
                "resources": list(finding.resources),
                "touched_resources": list(finding.touched_resources),
                "refs": len(finding.refs),
                "result_chars": len(finding.result),
                "merged": dict(finding.merged),
            }
            if finding.error:
                info["error"] = finding.error[:400]
            cfg = getattr(child, "config", None)
            if cfg is not None:
                info["max_tool_calls"] = getattr(cfg, "max_tool_calls", None)
                ctx = getattr(cfg, "context", None)
                info["max_context_tokens"] = getattr(ctx, "max_context_tokens", None)
            report = getattr(child, "_last_run_report", None)
            if report is not None:
                counters = getattr(report, "counters", None)
                if counters is not None:
                    info["tool_calls_business"] = getattr(
                        counters, "tool_calls_business", None
                    )
                    info["tool_calls_context"] = getattr(
                        counters, "tool_calls_context", None
                    )
                    info["llm_calls"] = getattr(counters, "llm_calls", None)
                    info["emergency_count"] = getattr(counters, "emergency_count", None)
                resolved = getattr(report, "config_resolved", None)
                if isinstance(resolved, dict):
                    info["window"] = resolved.get("context_window")
                    info["response_reserve"] = resolved.get("response_reserve")
            recorder.record_child(info, failed=finding.failed)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Synthesis                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_findings_section(
        findings: list[dict[str, Any]],
        *,
        per_finding_chars: int = 2_000,
    ) -> str:
        """The sub-agent findings exactly as the synthesizer reads them.

        Shared by :meth:`build_synthesis_prompt` and the Critic (I-10): a
        verifier of a COMPLEX synthesis must see the same hand-off the
        synthesizer saw, under the same per-finding cap.
        """
        cap = max(0, int(per_finding_chars))
        findings_text = ""
        for f in findings:
            body = str(f.get("result", ""))
            if len(body) > cap:
                body = body[: max(0, cap - 24)].rstrip() + "\n[... finding truncated]"
            meta_lines: list[str] = []
            status = f.get("status")
            if status and status != "success":
                meta_lines.append(f"status: {status}")
            touched = f.get("touched_resources") or []
            if touched:
                meta_lines.append(
                    "resources covered: " + ", ".join(map(str, touched[:20]))
                )
            refs = f.get("refs") or []
            if refs:
                shown = ", ".join(map(str, refs[:8]))
                more = f" (+{len(refs) - 8} more)" if len(refs) > 8 else ""
                meta_lines.append(f"evidence refs (recall_tool_result): {shown}{more}")
            meta = (
                ("\n".join(f"_{line}_" for line in meta_lines) + "\n")
                if meta_lines
                else ""
            )
            findings_text += f"\n### Sub-task: {f['objective']}\n{meta}{body}\n"
        return findings_text

    @staticmethod
    def build_synthesis_prompt(
        task_objective: str,
        findings: list[dict[str, Any]],
        *,
        per_finding_chars: int = 2_000,
    ) -> str:
        """Build a prompt to synthesize sub-agent findings.

        Args:
            task_objective: The original task.
            findings: List of sub-agent results.

        Returns:
            Prompt string for the synthesis LLM call.
        """
        findings_text = Decomposer.build_findings_section(
            findings, per_finding_chars=per_finding_chars
        )

        return (
            f"## ORIGINAL TASK\n{task_objective}\n\n"
            f"## SUB-AGENT FINDINGS\n{findings_text}\n\n"
            "## SYNTHESIS INSTRUCTIONS\n\n"
            "Follow these steps IN ORDER:\n\n"
            "1. **RE-READ the original task** — identify exactly what "
            "output is required (a number, a ranking, a comparison, "
            "a report, etc.).\n\n"
            "2. **EXTRACT relevant data** from each sub-agent's "
            "findings. If a sub-agent returned intermediate values, "
            "use them as inputs. If a sub-agent failed, note the gap.\n\n"
            "3. **COMPUTE the final answer** — if the task requires a "
            "calculation that combines sub-agent results (e.g., "
            "ranking, aggregation, comparison), perform that "
            "calculation explicitly using the available tools.\n\n"
            "4. **STATE the final answer clearly** — begin your "
            "conclusion with 'FINAL ANSWER: <value>' so the answer "
            "is unambiguous.\n\n"
            "IMPORTANT: Do NOT simply summarize the findings. You "
            "must produce the EXACT output the original task asks "
            "for. If the task asks for a single number, return that "
            "number. If it asks for a ranking, return the ranked list. "
            "Use tools to verify any computation.\n"
        )
