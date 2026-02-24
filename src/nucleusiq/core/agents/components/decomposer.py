"""
Decomposer Component — Task Analysis, Sub-Agent Orchestration, and Synthesis.

Handles the "complex task" path in autonomous mode:
1. Analyze task complexity (SIMPLE vs COMPLEX)
2. Spawn parallel sub-agents for independent sub-tasks
3. Summarize and synthesize sub-agent findings

Design:
    - Uses the framework's own Agent class for sub-agents (proven pattern)
    - Sub-agents run in STANDARD mode with isolated context
    - Plugin-governed limits prevent runaway costs
    - Findings are summarized before synthesis (context engineering)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.config.agent_config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.plugins.builtin.model_call_limit import ModelCallLimitPlugin

_SUB_AGENT_MAX_LLM_CALLS = 15


@dataclass
class TaskAnalysis:
    """Result of task complexity analysis."""

    is_complex: bool
    sub_tasks: List[Dict[str, str]] = field(default_factory=list)
    reasoning: str = ""


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

    # ------------------------------------------------------------------ #
    # Task Analysis                                                        #
    # ------------------------------------------------------------------ #

    async def analyze(self, agent: Agent, task: Task) -> TaskAnalysis:
        """Classify a task as SIMPLE or COMPLEX via one LLM call.

        Uses a 3-gate checklist: the task is COMPLEX only when ALL
        three gates evaluate to true.  If any gate is false the task
        is SIMPLE.  Falls back to SIMPLE on any error (safe default).
        """
        prompt = (
            "You are a task classifier. Evaluate the following task "
            "against three gate conditions to determine whether it "
            "should be handled by a single agent (SIMPLE) or split "
            "into parallel sub-agents (COMPLEX).\n\n"
            f"Task: {task.objective}\n\n"
            "## THREE-GATE CHECKLIST\n\n"
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
            "## DECISION RULE\n\n"
            "- If ALL three gates are true → COMPLEX.\n"
            "- If ANY gate is false → SIMPLE.\n\n"
            "Respond with ONLY this JSON:\n"
            '{"gate1": true/false, "gate2": true/false, '
            '"gate3": true/false, "complexity": "simple"}\n'
            "or\n"
            '{"gate1": true, "gate2": true, "gate3": true, '
            '"complexity": "complex", "sub_tasks": ['
            '{"id": "sub1", "objective": "..."}, ...]}\n'
        )
        try:
            response = await agent.llm.call(
                model=getattr(agent.llm, "model_name", "default"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return self._parse_analysis(response)
        except Exception as e:
            self._logger.warning(
                "Task analysis failed: %s — defaulting to SIMPLE",
                e,
            )
            return TaskAnalysis(is_complex=False, reasoning=f"Analysis error: {e}")

    def _parse_analysis(self, response: Any) -> TaskAnalysis:
        """Parse the LLM's complexity classification."""
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
            return TaskAnalysis(is_complex=False, reasoning="Could not parse")

        gate1 = json_match.get("gate1", False)
        gate2 = json_match.get("gate2", False)
        gate3 = json_match.get("gate3", False)
        all_gates_pass = gate1 and gate2 and gate3

        claimed_complex = json_match.get("complexity", "simple") == "complex"
        sub_tasks = json_match.get("sub_tasks", [])

        is_complex = claimed_complex and all_gates_pass and len(sub_tasks) >= 2

        if claimed_complex and not all_gates_pass:
            self._logger.info(
                "LLM claimed COMPLEX but gates failed "
                "(gate1=%s, gate2=%s, gate3=%s) — overriding to SIMPLE",
                gate1,
                gate2,
                gate3,
            )

        if claimed_complex and all_gates_pass and len(sub_tasks) < 2:
            self._logger.info(
                "LLM claimed COMPLEX but returned %d sub-tasks — overriding to SIMPLE",
                len(sub_tasks),
            )

        return TaskAnalysis(
            is_complex=is_complex,
            sub_tasks=sub_tasks if is_complex else [],
            reasoning=json_match.get("reasoning", ""),
        )

    # ------------------------------------------------------------------ #
    # Sub-Agent Creation                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def create_sub_agent(
        parent: Agent,
        sub_task_id: str,
        sub_task_objective: str,
    ) -> Agent | None:
        """Create an isolated sub-agent using the framework's Agent class.

        Shares the parent's LLM and tools but has its own:
        - Config (STANDARD mode)
        - Plugin state (isolated counters with call limit)
        - No memory (isolated context, no cross-contamination)

        Returns None if creation fails (graceful degradation).
        """
        from nucleusiq.agents.agent import Agent

        try:
            sub = Agent(
                name=f"{parent.name}-sub-{sub_task_id}",
                role=parent.role,
                objective=sub_task_objective,
                llm=parent.llm,
                tools=list(parent.tools),
                memory=None,
                config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
                plugins=[ModelCallLimitPlugin(max_calls=_SUB_AGENT_MAX_LLM_CALLS)],
            )
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

    async def run_sub_tasks(
        self,
        parent: Agent,
        sub_tasks: List[Dict[str, str]],
        max_sub_agents: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run sub-tasks in parallel via isolated sub-agents.

        Each sub-agent runs independently with its own context.
        Results are collected and returned as a list of findings.
        Failed sub-agents are logged and skipped (partial results
        are better than total failure).

        Args:
            parent: The parent agent (provides LLM, tools, role).
            sub_tasks: List of {"id": ..., "objective": ...} dicts.
            max_sub_agents: Cap on parallel sub-agents.

        Returns:
            List of {"id": ..., "objective": ..., "result": ...} dicts.
        """
        capped = sub_tasks[:max_sub_agents]
        self._logger.info(
            "Decomposing into %d sub-tasks (cap=%d)",
            len(capped),
            max_sub_agents,
        )

        agents_and_tasks = []
        for st in capped:
            sub_agent = await self.create_sub_agent(
                parent,
                st["id"],
                st["objective"],
            )
            if sub_agent:
                agents_and_tasks.append((sub_agent, st))
            else:
                self._logger.warning(
                    "Skipping sub-task %s (agent creation failed)", st["id"]
                )

        if not agents_and_tasks:
            return []

        async def _execute_one(agent: Agent, st: Dict[str, str]) -> Dict[str, Any]:
            try:
                raw = await agent.execute(Task(id=st["id"], objective=st["objective"]))
                return {
                    "id": st["id"],
                    "objective": st["objective"],
                    "result": str(raw),
                }
            except Exception as e:
                self._logger.warning("Sub-task %s failed: %s", st["id"], e)
                return {
                    "id": st["id"],
                    "objective": st["objective"],
                    "result": f"Error: {e}",
                }

        findings = await asyncio.gather(
            *[_execute_one(agent, st) for agent, st in agents_and_tasks]
        )
        return list(findings)

    # ------------------------------------------------------------------ #
    # Synthesis                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_synthesis_prompt(
        task_objective: str,
        findings: List[Dict[str, Any]],
    ) -> str:
        """Build a prompt to synthesize sub-agent findings.

        Args:
            task_objective: The original task.
            findings: List of sub-agent results.

        Returns:
            Prompt string for the synthesis LLM call.
        """
        findings_text = ""
        for f in findings:
            findings_text += f"\n### Sub-task: {f['objective']}\n{f['result'][:2000]}\n"

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
