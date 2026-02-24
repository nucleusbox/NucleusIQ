"""
Critic Component — Prompt Builder + Result Parser for the Verifier SubAgent.

The Critic does NOT execute LLM calls itself.  It builds verification
prompts and parses the result into structured CritiqueResult objects.
Execution is handled by the orchestrator (AutonomousMode) through the
framework's StandardMode tool loop — the same engine that powers the
Generator and Reviser subagents.

Architecture:
    Generator SubAgent (StandardMode tool loop)
        ↓  result
    Critic (builds prompt) → StandardMode tool loop → Critic (parses result)
        ↓  CritiqueResult
    Refiner SubAgent (components/refiner.py)
        ↓  corrected result
    (loop until Critic passes or max rounds reached)

Design Principles:
    1. Separation of concerns — builds prompts and parses results only
    2. Execution delegated — orchestrator runs via StandardMode tool loop
    3. Fresh context — Verifier gets its own conversation, no generator bias
    4. Three-way verdict — PASS / FAIL / UNCERTAIN
    5. Structured feedback — specific issues + actionable suggestions
    6. No raw loops — the framework's tool loop + plugin system handles limits
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.plan import Plan, PlanStep

# ------------------------------------------------------------------ #
# Data models                                                         #
# ------------------------------------------------------------------ #


class Verdict(str, Enum):
    """Three-way outcome of a critique review."""

    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class CritiqueResult(BaseModel):
    """Structured output from the Critic's review."""

    verdict: Verdict = Field(
        description="Overall verdict: pass, fail, or uncertain",
    )
    score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 = terrible, 1.0 = perfect)",
    )
    feedback: str = Field(
        default="",
        description="Human-readable overall assessment",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific problems found in the result",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable improvements for the Refiner",
    )
    verifier_answer: str | None = Field(
        default=None,
        description=(
            "The Verifier's independently computed answer. "
            "Used for programmatic cross-check against the Generator's answer."
        ),
    )


# ------------------------------------------------------------------ #
# Critic component                                                    #
# ------------------------------------------------------------------ #


class Critic:
    """Prompt builder and result parser for the Verifier SubAgent.

    The Critic does NOT execute LLM calls or manage tool loops.  It:
    1. Builds verification prompts (system + user messages)
    2. Parses the Verifier's text output into ``CritiqueResult``

    Execution is delegated to the orchestrator (AutonomousMode) which
    runs the verification through ``StandardMode._tool_call_loop`` —
    the same engine that powers the Generator and Reviser subagents.
    This means:
    - No raw loops or hardcoded iteration limits inside the Critic
    - The framework's plugin system (ToolCallLimitPlugin, etc.) governs
      all execution limits
    - The Verifier gets full tool access through the framework's tool loop

    For backward compatibility, ``review_step`` and ``review_final`` still
    make single LLM calls directly.  The new ``build_verification_messages``
    + ``parse_result_text`` API is the preferred path for AutonomousMode.

    Usage (preferred — via AutonomousMode)::

        critic = Critic()
        messages = critic.build_verification_messages(objective, result, conv)
        # AutonomousMode runs these through std_mode._tool_call_loop
        verifier_output = await std_mode._tool_call_loop(...)
        critique = critic.parse_result_text(verifier_output)

    Usage (legacy — single LLM call)::

        critique = await critic.review_final(agent, objective, final_result=r)

    When no LLM is available the Critic returns ``Verdict.PASS`` so the
    pipeline degrades gracefully (execute → done, no verification).
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # Public API — New (used by AutonomousMode)                           #
    # ------------------------------------------------------------------ #

    def build_verification_prompt(
        self,
        task_objective: str,
        final_result: Any,
        generator_messages: List[Any] | None = None,
    ) -> str:
        """Build an adaptive verification prompt for the Verifier Agent.

        Automatically detects whether the Generator used tools and
        dispatches to the appropriate verification strategy:

        - **Tool verification**: When tools were called, the Verifier
          spot-checks the critical step by calling 1-3 tools itself.
        - **Reasoning verification**: When no tools were used, the
          Verifier reviews logical consistency, completeness, and
          whether the answer addresses all parts of the task.

        Both strategies share the same JSON response format so the
        ``parse_result_text`` parser works uniformly.

        Args:
            task_objective: The user's original task text.
            final_result: The Generator's answer to verify.
            generator_messages: The Generator's conversation (optional).

        Returns:
            Prompt string to use as the Verifier Agent's task objective.
        """
        trace = self._extract_reasoning_trace(generator_messages)
        used_tools = bool(trace and "[Tool Call]" in trace)

        if used_tools:
            return self._build_tool_verification(
                task_objective,
                final_result,
                trace,
            )
        return self._build_reasoning_verification(
            task_objective,
            final_result,
            trace,
        )

    # ------------------------------------------------------------------ #
    # Verification strategies (Open/Closed — add new strategies here)     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_tool_verification(
        task_objective: str,
        final_result: Any,
        trace: str,
    ) -> str:
        """Verification strategy for tool-based tasks.

        The Verifier independently re-derives the final answer from
        the intermediate values in the trace, then compares with the
        Generator's claimed answer. This catches assembly errors that
        random spot-checks miss.
        """
        return (
            "VERIFY whether the following answer is correct.\n\n"
            f"## ORIGINAL TASK\n{task_objective}\n\n"
            f"## GENERATOR'S EXECUTION TRACE\n{trace}\n\n"
            f"## CLAIMED ANSWER\n{_truncate(str(final_result), 3000)}\n\n"
            "## YOUR JOB — MANDATORY STEPS IN ORDER\n"
            "1. READ the execution trace to understand the intermediate "
            "values and the approach used.\n"
            "2. **RE-DERIVE THE FINAL ANSWER YOURSELF**: Using the "
            "intermediate data from the trace, call the tools needed to "
            "independently compute the FINAL answer from scratch. "
            "This is the MOST IMPORTANT step — you MUST do this.\n"
            "3. If you suspect an intermediate value is wrong, "
            "spot-check that value too.\n"
            "4. Compare YOUR independently computed answer with the "
            "CLAIMED ANSWER.\n\n"
            "## RULES\n"
            "- You MUST call tools to independently derive the final "
            "answer — do NOT just review the trace and approve.\n"
            "- Report YOUR computed answer in the "
            '"verifier_answer" field.\n'
            "- If your answer matches the claimed answer (within "
            'reasonable rounding) → verdict "pass".\n'
            "- If your answer DIFFERS from the claimed answer → "
            'verdict "fail" with the discrepancy.\n'
            '- If you cannot compute the answer → verdict "uncertain".\n\n'
            + _VERDICT_FORMAT
        )

    @staticmethod
    def _build_reasoning_verification(
        task_objective: str,
        final_result: Any,
        trace: str,
    ) -> str:
        """Verification strategy for reasoning-only tasks (no tools).

        The Verifier reviews the output for logical consistency,
        completeness, factual plausibility, and task alignment.
        No tool calls required.
        """
        return (
            "VERIFY whether the following answer is correct and complete.\n\n"
            f"## ORIGINAL TASK\n{task_objective}\n\n"
            + (f"## GENERATOR'S RESPONSE\n{trace}\n\n" if trace else "")
            + f"## CLAIMED ANSWER\n{_truncate(str(final_result), 3000)}\n\n"
            "## YOUR JOB\n"
            "Review the answer for quality across these dimensions:\n\n"
            "1. **Logical Consistency** — Does the reasoning chain make sense? "
            "Are there contradictions or unsupported leaps?\n"
            "2. **Completeness** — Does the answer address ALL parts of the "
            "task? If the task asks for multiple things, are all covered?\n"
            "3. **Factual Plausibility** — Are claims reasonable and "
            "consistent with general knowledge? Any obvious errors?\n"
            "4. **Task Alignment** — Does the answer actually answer what "
            "was asked? Is the format/unit correct?\n\n"
            "## RULES\n"
            "- Do NOT fail for style or formatting preferences.\n"
            "- Do NOT fail because a different approach could also work.\n"
            "- Only FAIL when you can point to a SPECIFIC, CONCRETE error "
            "or missing requirement.\n"
            '- Major error (wrong answer, missed requirement) → "fail", '
            "score < 0.3.\n"
            '- Minor issue (could be better) → "pass", score 0.6-0.8.\n'
            '- Cannot determine quality → "uncertain".\n\n' + _VERDICT_FORMAT
        )

    @staticmethod
    def _extract_reasoning_trace(messages: List[Any] | None) -> str:
        """Extract the Generator's execution trace from its conversation.

        Captures the full sequence of what the Generator did:
        - Tool calls (function name + arguments)
        - Tool results (return values, truncated for size)
        - Assistant text (explanations, if any — many models return
          None for content during tool-calling turns)

        Skips system and user messages (the Verifier already has the task).
        The Verifier uses this trace to understand WHAT was done and
        WHERE to focus its spot-check.
        """
        if not messages:
            return ""
        lines: List[str] = []
        for msg in messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "?")
            content = (
                msg.content if hasattr(msg, "content") else msg.get("content", "")
            ) or ""
            tool_calls = (
                getattr(msg, "tool_calls", None)
                if hasattr(msg, "tool_calls")
                else msg.get("tool_calls")
            )

            if role in ("system", "user"):
                continue

            if role == "assistant":
                if content.strip():
                    lines.append(f"[Assistant] {_truncate(content, 500)}")
                if tool_calls:
                    for tc in tool_calls:
                        name = (
                            tc.name
                            if hasattr(tc, "name")
                            else tc.get("function", {}).get("name", "?")
                        )
                        args = (
                            tc.arguments
                            if hasattr(tc, "arguments")
                            else tc.get("function", {}).get("arguments", "")
                        )
                        lines.append(f"[Tool Call] {name}({_truncate(str(args), 200)})")
            elif role == "tool":
                lines.append(f"[Tool Result] {_truncate(content, 300)}")

        if len(lines) > 50:
            lines = lines[:10] + ["  ... (middle steps omitted) ..."] + lines[-30:]
        return "\n".join(lines)

    def parse_result_text(self, text: str) -> CritiqueResult:
        """Parse the Verifier's text output into a structured CritiqueResult.

        Handles:
        - Clean JSON
        - JSON inside markdown code fences
        - Plain text (falls back to keyword inference)
        - Empty / None (returns UNCERTAIN)

        Args:
            text: The raw text output from the Verifier subagent.

        Returns:
            CritiqueResult with verdict, score, feedback, issues, suggestions.
        """
        if not text or not text.strip():
            return CritiqueResult(
                verdict=Verdict.UNCERTAIN,
                score=0.5,
                feedback="Verifier returned empty response",
            )

        json_str = self._extract_json(text)
        if not json_str:
            return self._infer_from_text(text)

        try:
            data = json.loads(json_str)
            raw_va = data.get("verifier_answer")
            return CritiqueResult(
                verdict=Verdict(data.get("verdict", "uncertain")),
                score=float(data.get("score", 0.5)),
                feedback=str(data.get("feedback", "")),
                issues=list(data.get("issues", [])),
                suggestions=list(data.get("suggestions", [])),
                verifier_answer=str(raw_va) if raw_va is not None else None,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._logger.debug(
                "Failed to parse verifier JSON: %s. Inferring from text.",
                e,
            )
            return self._infer_from_text(text)

    # ------------------------------------------------------------------ #
    # Public API — Legacy (single LLM call, backward compatible)          #
    # ------------------------------------------------------------------ #

    async def review_step(
        self,
        agent: Agent,
        task_objective: str,
        step: PlanStep,
        result: Any,
        context: Dict[str, Any],
    ) -> CritiqueResult:
        """Review a single step result (single LLM call, no tool loop).

        Args:
            agent: Agent instance (provides LLM access)
            task_objective: The user's original task text
            step: The PlanStep that was executed
            result: The raw result produced by the Generator
            context: Accumulated results from prior steps

        Returns:
            CritiqueResult with verdict, feedback, issues, suggestions
        """
        if not agent.llm:
            return self._auto_pass("No LLM available — skipping critique")

        prompt = self._build_step_review_prompt(
            task_objective,
            step,
            result,
            context,
        )
        return await self._run_single_call(agent, prompt)

    async def review_final(
        self,
        agent: Agent,
        task_objective: str,
        plan: Plan | None = None,
        results: List[Any] | None = None,
        final_result: Any = None,
        messages: List[Any] | None = None,
    ) -> CritiqueResult:
        """Review the final result (single LLM call, backward compatible).

        For the full Verifier subagent with tool access, use
        ``build_verification_messages`` + ``parse_result_text`` instead.

        Args:
            agent: Agent instance (provides LLM access)
            task_objective: The user's original task text
            plan: The execution plan (optional, legacy support)
            results: All step results (optional, legacy support)
            final_result: The final result to be returned to the user
            messages: Full conversation messages including tool calls

        Returns:
            CritiqueResult for the overall output
        """
        if not agent.llm:
            return self._auto_pass("No LLM available — skipping final review")

        if messages:
            prompt = self._build_conversation_review_prompt(
                task_objective,
                final_result,
                messages,
            )
        else:
            prompt = self._build_final_review_prompt(
                task_objective,
                plan,
                results or [],
                final_result,
            )
        return await self._run_single_call(agent, prompt)

    # ------------------------------------------------------------------ #
    # Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_step_review_prompt(
        task_objective: str,
        step: PlanStep,
        result: Any,
        context: Dict[str, Any],
    ) -> str:
        step_details = f"\nStep Details: {step.details}" if step.details else ""
        ctx_summary = ""
        if context:
            ctx_lines = []
            for k, v in context.items():
                if k.endswith("_action"):
                    continue
                ctx_lines.append(f"  - {k}: {_truncate(str(v), 300)}")
            if ctx_lines:
                ctx_summary = "\n\nPrevious Steps Results:\n" + "\n".join(ctx_lines)

        return (
            "You are a quality reviewer for an AI agent. "
            "Your job is to verify whether a task step was completed "
            "correctly and completely.\n\n"
            f"## Original Task\n{task_objective}\n\n"
            f"## Step Being Reviewed\n"
            f"Step {step.step}: {step.action}{step_details}\n\n"
            f"## Step Result\n{_truncate(str(result), 2000)}\n"
            f"{ctx_summary}\n\n"
            "## Review Instructions\n"
            "Evaluate the step result. Consider:\n"
            "1. **Correctness**: Is the result factually accurate?\n"
            "2. **Completeness**: Does it fully address the step's purpose?\n"
            "3. **Relevance**: Is it useful for the overall task?\n"
            "4. **Quality**: Is it well-formed and actionable?\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "verdict": "pass" | "fail" | "uncertain",\n'
            '  "score": 0.0 to 1.0,\n'
            '  "feedback": "overall assessment",\n'
            '  "issues": ["specific problem 1", ...],\n'
            '  "suggestions": ["specific improvement 1", ...]\n'
            "}\n\n"
            "Rules:\n"
            '- "pass": result is correct, complete, and useful\n'
            '- "fail": result has clear errors or is incomplete\n'
            '- "uncertain": you cannot determine the quality\n'
            "- Be specific in issues and suggestions\n"
        )

    @staticmethod
    def _build_conversation_review_prompt(
        task_objective: str,
        final_result: Any,
        messages: List[Any],
    ) -> str:
        """Build a multi-dimensional verification prompt.

        The Critic independently re-reasons about the task, verifying:
        1. Accuracy — does the answer match the evidence?
        2. Completeness — did it address ALL parts of the task?
        3. Reasoning validity — is the logic sound?
        4. Self-consistency — no internal contradictions?
        """
        tool_evidence: List[str] = []
        assistant_reasoning: List[str] = []
        for msg in messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "?")
            content = (
                msg.content if hasattr(msg, "content") else msg.get("content", "")
            ) or ""

            tool_calls = (
                getattr(msg, "tool_calls", None)
                if hasattr(msg, "tool_calls")
                else msg.get("tool_calls")
            )

            if role == "assistant" and tool_calls:
                for tc in tool_calls:
                    name = (
                        tc.name
                        if hasattr(tc, "name")
                        else tc.get("function", {}).get("name", "?")
                    )
                    args = (
                        tc.arguments
                        if hasattr(tc, "arguments")
                        else tc.get("function", {}).get("arguments", "")
                    )
                    tool_evidence.append(f"  CALL: {name}({_truncate(str(args), 400)})")
            elif role == "tool":
                tool_evidence.append(f"  RESULT: {_truncate(str(content), 600)}")
            elif role == "assistant" and content:
                assistant_reasoning.append(_truncate(content, 500))

        evidence = "\n".join(tool_evidence) if tool_evidence else "(no tools called)"
        reasoning = (
            "\n---\n".join(assistant_reasoning[-3:])
            if assistant_reasoning
            else "(none)"
        )

        return (
            "You are an independent verifier with access to the same tools "
            "the agent used. Your job: independently verify whether the "
            "agent's answer is correct.\n\n"
            "## HOW TO VERIFY\n"
            "You can call the available tools yourself to independently "
            "check the agent's work. For example:\n"
            "- Re-call a tool with the same arguments to confirm a result.\n"
            "- Call a tool with different arguments to cross-check.\n"
            "- Spot-check key intermediate values.\n"
            "You do NOT need to re-do everything — focus on the parts that "
            "look suspicious or critical.\n\n"
            "## VERIFICATION DIMENSIONS\n"
            "Check ALL of the following:\n\n"
            "### 1. Accuracy\n"
            "- Do the agent's conclusions match the tool outputs?\n"
            "- If the task involves calculations, are they correct?\n"
            "- If the task involves facts or data, are they accurately used?\n"
            "- FAIL if the answer clearly contradicts the evidence.\n\n"
            "### 2. Task Completeness\n"
            "- Did the agent address ALL parts of the task?\n"
            "- If the task asks for multiple things, are all answered?\n"
            "- FAIL if the agent skipped a required part.\n\n"
            "### 3. Reasoning Validity\n"
            "- Does the reasoning chain make logical sense?\n"
            "- Did the agent use the right approach and right tools?\n"
            "- FAIL if the agent used a wrong method or skipped a "
            "critical step.\n\n"
            "### 4. Self-Consistency\n"
            "- Does the final answer match what the agent derived?\n"
            "- FAIL if there are internal contradictions.\n\n"
            "## SAFEGUARDS\n"
            "- Do NOT fail for style, phrasing, or formatting preferences.\n"
            "- Do NOT fail because a different approach could also work.\n"
            "- Only FAIL when you can point to a SPECIFIC, CONCRETE error.\n"
            "- If the error is minor (rounding, formatting), use score > 0.5.\n"
            "- If the error is major (wrong answer, missed requirement), "
            "use score < 0.3.\n\n"
            f"## Task\n{task_objective}\n\n"
            f"## What the Agent Did (tool calls and results)\n"
            f"{_truncate(evidence, 4000)}\n\n"
            f"## Agent's Reasoning (last steps)\n"
            f"{_truncate(reasoning, 2000)}\n\n"
            f"## Agent's Final Answer\n"
            f"{_truncate(str(final_result), 2000)}\n\n"
            "## Your Verification\n"
            "1. Re-read the task. What EXACTLY is being asked?\n"
            "2. If anything looks suspicious, call the tools to verify.\n"
            "3. Trace the agent's reasoning. Is each step valid?\n"
            "4. Check the final answer against the evidence.\n"
            "5. Give your verdict as a JSON object:\n\n"
            "{\n"
            '  "verdict": "pass" | "fail" | "uncertain",\n'
            '  "score": 0.0 to 1.0,\n'
            '  "feedback": "what you verified and your conclusion",\n'
            '  "issues": ["specific error with evidence, if any"],\n'
            '  "suggestions": ["specific fix needed, if any"]\n'
            "}\n"
        )

    @staticmethod
    def _build_final_review_prompt(
        task_objective: str,
        plan: Plan,
        results: List[Any],
        final_result: Any,
    ) -> str:
        plan_lines = []
        for s in plan.steps:
            detail = f" — {s.details}" if s.details else ""
            plan_lines.append(f"  Step {s.step}: {s.action}{detail}")
        plan_summary = "\n".join(plan_lines)

        results_lines = []
        for i, r in enumerate(results, 1):
            results_lines.append(f"  Step {i} result: {_truncate(str(r), 500)}")
        results_summary = "\n".join(results_lines)

        return (
            "You are a quality reviewer for an AI agent. "
            "Your job is to verify whether the agent's complete output "
            "satisfies the original task.\n\n"
            f"## Original Task\n{task_objective}\n\n"
            f"## Execution Plan\n{plan_summary}\n\n"
            f"## Step Results\n{results_summary}\n\n"
            f"## Final Result\n{_truncate(str(final_result), 2000)}\n\n"
            "## Review Instructions\n"
            "Evaluate whether the final result satisfies the original task:\n"
            "1. **Task Completion**: Does the result answer the task?\n"
            "2. **Accuracy**: Are conclusions and facts correct?\n"
            "3. **Coherence**: Does the result make sense as a whole?\n"
            "4. **Quality**: Would a user be satisfied?\n\n"
            "Respond with ONLY a JSON object:\n"
            "{\n"
            '  "verdict": "pass" | "fail" | "uncertain",\n'
            '  "score": 0.0 to 1.0,\n'
            '  "feedback": "overall assessment",\n'
            '  "issues": ["specific problem 1", ...],\n'
            '  "suggestions": ["specific improvement 1", ...]\n'
            "}\n"
        )

    # ------------------------------------------------------------------ #
    # Internal — single LLM call (legacy path)                            #
    # ------------------------------------------------------------------ #

    async def _run_single_call(
        self,
        agent: Agent,
        prompt: str,
    ) -> CritiqueResult:
        """Single LLM call without tools (text-only verification).

        Used by the legacy ``review_step`` / ``review_final`` API.
        For the full Verifier subagent path, AutonomousMode runs
        ``build_verification_messages`` through ``StandardMode._tool_call_loop``.
        """
        try:
            call_kwargs = {
                "model": getattr(agent.llm, "model_name", "default"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
            }
            call_kwargs.update(getattr(agent, "_current_llm_overrides", {}))
            response = await agent.llm.call(**call_kwargs)
            return self._parse_response(response)
        except Exception as e:
            self._logger.warning("Critic LLM call failed: %s", e)
            return CritiqueResult(
                verdict=Verdict.UNCERTAIN,
                score=0.5,
                feedback=f"Critique failed: {e}",
            )

    def _parse_response(self, response: Any) -> CritiqueResult:
        """Extract CritiqueResult from an LLM response object.

        Used by the legacy single-call path.  The new path uses
        ``parse_result_text`` which takes raw text directly.
        """
        content = self._extract_content(response)
        if not content:
            return CritiqueResult(
                verdict=Verdict.UNCERTAIN,
                score=0.5,
                feedback="Critic returned empty response",
            )
        return self.parse_result_text(content)

    # ------------------------------------------------------------------ #
    # Parsing helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_content(response: Any) -> str | None:
        """Pull text content from an LLM response object."""
        if not response or not hasattr(response, "choices") or not response.choices:
            return None
        msg = response.choices[0].message
        if isinstance(msg, dict):
            return msg.get("content")
        return getattr(msg, "content", None)

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON from text, handling markdown fences."""
        fenced = re.search(
            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
            text,
            re.DOTALL,
        )
        if fenced:
            return fenced.group(1)

        bare = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
        if bare:
            return bare.group(0)
        return None

    @staticmethod
    def _infer_from_text(content: str) -> CritiqueResult:
        """Best-effort verdict when the LLM didn't produce JSON.

        Only assigns PASS or FAIL when the signal is unambiguous.
        If both keywords appear (e.g. an echoed prompt that contains
        the instruction text), returns UNCERTAIN so the pipeline
        proceeds without unnecessary revision.
        """
        lower = content.lower()
        has_pass = "pass" in lower
        has_fail = "fail" in lower

        if has_pass and not has_fail:
            return CritiqueResult(
                verdict=Verdict.PASS,
                score=0.7,
                feedback=content[:500],
            )
        if has_fail and not has_pass:
            return CritiqueResult(
                verdict=Verdict.FAIL,
                score=0.3,
                feedback=content[:500],
            )
        return CritiqueResult(
            verdict=Verdict.UNCERTAIN,
            score=0.5,
            feedback=content[:500],
        )

    @staticmethod
    def _auto_pass(reason: str) -> CritiqueResult:
        """Return a PASS when critique cannot be performed."""
        return CritiqueResult(
            verdict=Verdict.PASS,
            score=1.0,
            feedback=reason,
        )


# ------------------------------------------------------------------ #
# Module-level helpers                                                #
# ------------------------------------------------------------------ #


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for prompt inclusion."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


_VERDICT_FORMAT = (
    "## RESPONSE FORMAT\n"
    "Respond with ONLY this JSON:\n"
    "{\n"
    '  "verdict": "pass" | "fail" | "uncertain",\n'
    '  "score": 0.0 to 1.0,\n'
    '  "verifier_answer": "YOUR independently computed answer (number, text, etc.)",\n'
    '  "feedback": "what you checked and your conclusion",\n'
    '  "issues": ["specific problem if any"],\n'
    '  "suggestions": ["specific fix: replace X with Y"]\n'
    "}\n"
)
