# src/nucleusiq/agents/task.py
"""
Task class for representing user requests to agents.

Task = WHAT the user wants done (specific request)
This is different from Agent.objective which is the agent's general purpose.
"""

from __future__ import annotations

from typing import Any

from nucleusiq.agents.attachments import Attachment
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

        # With attachments (v0.4.0+)
        task = Task(
            id="task2",
            objective="What is in this image?",
            attachments=[
                Attachment(type="image_url", data="https://example.com/img.png"),
            ],
        )

        # Grounded (Autonomous harness hardening): the documents / records the
        # task is about.  Drives the decomposition coverage contract and the
        # post-run coverage reconciliation.
        task = Task(
            id="task3",
            objective="Extract the invoice fields from every document",
            resources=["invoices/2026-01.pdf", "invoices/2026-02.pdf"],
            context={"customer": "ACME", "currency": "EUR"},
        )
        ```
    """

    id: str = Field(..., description="Unique task identifier")
    objective: str = Field(
        ..., description="What the user wants done (specific request)"
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional context for the task. Rendered as a bounded "
            "'Task Context' block ahead of the objective in the user "
            "message, so it does not need to be pasted into ``objective``."
        ),
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Task metadata")
    attachments: list[Attachment] | None = Field(
        default=None,
        description="Files, images, or other media attached to this task",
    )
    resources: list[str] | None = Field(
        default=None,
        description=(
            "Identifiers of the documents / files / records this task must "
            "cover (paths, URLs, doc ids). In Autonomous mode the Decomposer "
            "grounds its classification on them, every sub-task must claim a "
            "slice, and unprocessed resources are reported (and, when "
            "possible, followed up). ``context['resources']`` is accepted as "
            "a fallback; the typed field wins when both are present."
        ),
    )

    def effective_resources(self) -> list[str]:
        """Resources for this task — typed field first, ``context['resources']`` second.

        Always a de-duplicated list of non-empty strings (order kept), so
        callers never branch on ``None`` or on a mistyped payload.
        """
        raw: Any = self.resources
        if raw is None and isinstance(self.context, dict):
            raw = self.context.get("resources")
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple, set)):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for item in raw:
            text = str(item).strip() if item is not None else ""
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out

    def context_without_resources(self) -> dict[str, Any]:
        """``context`` minus the ``resources`` fallback key (rendered separately)."""
        if not isinstance(self.context, dict):
            return {}
        return {k: v for k, v in self.context.items() if k != "resources"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """
        Create Task from dictionary (backward compatibility).

        Args:
            data: Dictionary with 'id' and 'objective' keys

        Returns:
            Task instance
        """
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert Task to dictionary.

        Returns:
            Dictionary representation of the task
        """
        return self.model_dump()
