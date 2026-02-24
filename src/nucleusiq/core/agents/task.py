# src/nucleusiq/agents/task.py
"""
Task class for representing user requests to agents.

Task = WHAT the user wants done (specific request)
This is different from Agent.objective which is the agent's general purpose.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    Represents a task to be executed by an agent.

    Task represents WHAT the user wants done (specific request).
    This is different from Agent.objective which represents the agent's
    general purpose (WHO the agent is).

    Example:
        ```python
        task = Task(id="task1", objective="What is 5 + 3?")

        # Or from dictionary (backward compatible)
        task = Task.from_dict({"id": "task1", "objective": "What is 5 + 3?"})
        ```
    """

    id: str = Field(..., description="Unique task identifier")
    objective: str = Field(
        ..., description="What the user wants done (specific request)"
    )
    context: Dict[str, Any] | None = Field(
        default=None, description="Additional context for the task"
    )
    metadata: Dict[str, Any] | None = Field(default=None, description="Task metadata")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """
        Create Task from dictionary (backward compatibility).

        Args:
            data: Dictionary with 'id' and 'objective' keys

        Returns:
            Task instance
        """
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Task to dictionary.

        Returns:
            Dictionary representation of the task
        """
        return self.model_dump()
