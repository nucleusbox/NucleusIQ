"""
Strategy for planning prompt generation.

Defines the contract (``PlanPromptStrategy``) and a sensible default
(``DefaultPlanPromptStrategy``) that produces JSON-structured planning
prompts.

To customise planning prompts, implement the protocol and inject via
``Planner(agent, prompt_strategy=MyStrategy())``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class PlanPromptStrategy(Protocol):
    """Contract for planning prompt generation."""

    def build_planning_prompt(
        self,
        task_objective: str,
        tool_names: List[str],
        role: str,
        objective: str,
    ) -> str:
        """Full structured-JSON planning prompt (content fallback)."""
        ...

    def build_tool_call_prompt(
        self,
        task_objective: str,
        tool_param_lines: List[str],
    ) -> str:
        """Concise prompt for tool-calling-based plan generation."""
        ...

    def build_step_inference_prompt(
        self,
        action: str,
        required_keys: List[str],
        task_objective: str,
        context: Dict[str, Any],
        step_num: int,
        step_details: str,
    ) -> str:
        """Prompt that asks the LLM to infer tool arguments for a step."""
        ...


class DefaultPlanPromptStrategy:
    """Default JSON-based planning prompts.  Works out of the box."""

    def build_planning_prompt(
        self,
        task_objective: str,
        tool_names: List[str],
        role: str,
        objective: str,
    ) -> str:
        tools_str = ", ".join(tool_names) or "None"

        if role and objective:
            base = (
                f"As {role} with objective '{objective}',\n"
                f"create a plan to accomplish the following task:\n"
                f"{task_objective}"
            )
        else:
            base = f"Create a plan for the following task:\n{task_objective}"

        return base + self._json_instructions(tools_str)

    def build_tool_call_prompt(
        self,
        task_objective: str,
        tool_param_lines: List[str],
    ) -> str:
        tools_block = "\n".join(tool_param_lines) if tool_param_lines else "- None"
        return (
            "Create a step-by-step execution plan for the task below. "
            "You MUST call the create_plan tool with the plan.\n\n"
            f"Task: {task_objective}\n\n"
            f"Available tools:\n{tools_block}\n\n"
            "CRITICAL RULES:\n"
            "1. EVERY tool step MUST include 'args' with ALL required "
            "parameters filled in.\n"
            "2. For the first steps, extract CONCRETE VALUES from the "
            "task (e.g., numbers, strings).\n"
            "3. For later steps that need results from previous steps, "
            'use "$step_N" references.\n\n'
            "Now create the plan for the given task. Extract actual "
            "values from the task text.\n"
        )

    def build_step_inference_prompt(
        self,
        action: str,
        required_keys: List[str],
        task_objective: str,
        context: Dict[str, Any],
        step_num: int,
        step_details: str,
    ) -> str:
        return (
            "You must call the tool below with valid JSON arguments.\n\n"
            f"Tool: {action}\n"
            f"Required args: {required_keys}\n\n"
            f"Overall task: {task_objective}\n"
            f"Current context (use these concrete values): "
            f"{json.dumps(context)}\n"
            f"Step {step_num} details: {step_details}\n\n"
            "Call the tool now."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _json_instructions(tools_str: str) -> str:
        return (
            "\n\nIMPORTANT: You must respond with a valid JSON object "
            "following this exact structure:\n"
            "{\n"
            '    "steps": [\n'
            "        {\n"
            '            "step": 1,\n'
            '            "action": "execute",\n'
            '            "args": {},\n'
            '            "details": "Description of this step"\n'
            "        },\n"
            "        {\n"
            '            "step": 2,\n'
            '            "action": "tool_name",\n'
            '            "args": {"param1": "value1"},\n'
            '            "details": "Description of this step"\n'
            "        }\n"
            "    ]\n"
            "}\n\n"
            "Requirements:\n"
            '- "steps" must be an array of step objects\n'
            '- Each step must have "step" (integer, 1-indexed) '
            'and "action" (string)\n'
            '- "args" (object) and "details" (string) are optional\n'
            '- Action can be "execute" for direct execution or a '
            f"tool name from: [{tools_str}]\n"
            "- Return ONLY valid JSON, no additional text\n\n"
            f"Available tools: {tools_str}\n\n"
            "Create a step-by-step plan to accomplish this task. "
            "Return the plan as a JSON object."
        )
