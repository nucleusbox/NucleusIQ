"""
PlanParser — parses LLM planning responses into structured Plan objects.

Extracted from ``Agent._parse_plan_response()``.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from nucleusiq.agents.plan import PlanResponse, PlanStepResponse


class PlanParser:
    """Parses LLM responses into PlanResponse models."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)

    def parse(self, response: str) -> PlanResponse:
        """
        Parse the LLM's planning response into a structured PlanResponse model.

        Supports multiple formats:
        1. Function call response (preferred — structured)
        2. JSON object in response text
        3. Fallback to text parsing (backward compatibility)

        Returns PlanResponse model instance.
        """

        # Helper function to extract JSON with balanced brackets
        def _extract_balanced_json(text: str, start_pos: int) -> Optional[str]:
            """Extract JSON object using balanced bracket matching to handle nested structures."""
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(text[start_pos:], start=start_pos):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return text[start_pos : i + 1]
            return None

        # Method 1: Try to extract JSON from markdown code blocks
        json_str: Optional[str] = None
        code_block_match = re.search(
            r"```(?:json)?\s*(\{)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if code_block_match:
            json_str = _extract_balanced_json(response, code_block_match.start(1))

        # Method 2: Try to find JSON object directly in text
        if not json_str:
            first_brace = response.find("{")
            if first_brace != -1:
                json_str = _extract_balanced_json(response, first_brace)

        # Method 3: Fallback to entire response
        if not json_str:
            json_str = response.strip()

        # Try to parse as JSON and validate with Pydantic model
        try:
            plan_data = json.loads(json_str)
            if isinstance(plan_data, dict) and "steps" in plan_data:
                # Use Pydantic model for validation and parsing
                plan_response = PlanResponse.from_dict(plan_data)
                self._logger.debug(
                    "Successfully parsed %d steps from JSON using PlanResponse model",
                    len(plan_response.steps),
                )
                return plan_response
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self._logger.warning(
                "Failed to parse JSON from response: %s. Trying fallback parsing.", e
            )

        # Method 2: Fallback to text parsing (backward compatibility)
        self._logger.warning(
            "Using fallback text parsing. Consider using structured JSON output "
            "for better reliability."
        )
        step_responses: List[PlanStepResponse] = []
        current_step: Optional[Dict[str, Any]] = None

        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Look for step markers
            step_match = re.match(r"Step\s+(\d+)[:.]?\s*(.*)", line, re.IGNORECASE)
            if step_match:
                if current_step:
                    step_responses.append(PlanStepResponse(**current_step))
                step_num = int(step_match.group(1))
                action = step_match.group(2).strip()
                current_step = {
                    "step": step_num,
                    "action": action or "execute",
                    "args": {},
                    "details": "",
                }
            elif current_step:
                # Check if line contains action information
                if "action" in line.lower() and ":" in line:
                    action_match = re.search(
                        r"action[:\s]+(.+)", line, re.IGNORECASE
                    )
                    if action_match and (
                        not current_step.get("action")
                        or current_step.get("action") == "execute"
                    ):
                        current_step["action"] = action_match.group(1).strip()
                else:
                    # Add to details
                    if current_step["details"]:
                        current_step["details"] += "\n" + line
                    else:
                        current_step["details"] = line

        if current_step:
            step_responses.append(PlanStepResponse(**current_step))

        # If no steps found, create a default single-step plan
        if not step_responses:
            self._logger.warning(
                "No steps parsed from response. Creating default single-step plan."
            )
            step_responses = [
                PlanStepResponse(
                    step=1, action="execute", args={}, details="Execute the task"
                )
            ]

        return PlanResponse(steps=step_responses)
