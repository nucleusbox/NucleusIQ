"""
Plan schema definitions — shared by PlanCreator and Planner.

Extracted here to break the circular dependency where PlanCreator
imported Planner just to access ``get_plan_function_spec()``.
"""

from typing import Any, Dict


def get_plan_schema() -> Dict[str, Any]:
    """Minimal, OpenAI-friendly JSON schema for plan output."""
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "integer",
                            "description": "Step number (1-indexed)",
                        },
                        "action": {
                            "type": "string",
                            "description": ("Action/tool name or 'execute'"),
                        },
                        "args": {
                            "type": "object",
                            "description": ("Arguments for the action/tool"),
                        },
                        "details": {
                            "type": "string",
                            "description": ("Human-readable description"),
                        },
                    },
                    "required": ["step", "action", "args"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }


def get_plan_function_spec() -> Dict[str, Any]:
    """Function specification for structured plan generation via
    function calling."""
    return {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": (
                "Create a structured execution plan with step-by-step actions"
            ),
            "parameters": get_plan_schema(),
        },
    }
