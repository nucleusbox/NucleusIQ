"""
ProgressTracker — Step-by-step execution tracking for Autonomous Mode.

Records each step's status, result, and validation outcome.
Provides structured context for retries and observability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


StepStatus = Literal["pending", "executing", "completed", "failed", "skipped"]


@dataclass
class StepRecord:
    """One step in an autonomous execution."""

    step_id: str
    objective: str
    status: StepStatus = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def mark_executing(self) -> None:
        self.status = "executing"
        self.attempts += 1
        self.started_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self, result: str) -> None:
        self.status = "completed"
        self.result = result
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionProgress:
    """Tracks all steps of an autonomous execution."""

    task_id: str
    steps: List[StepRecord] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def completed_steps(self) -> List[StepRecord]:
        return [s for s in self.steps if s.status == "completed"]

    @property
    def failed_steps(self) -> List[StepRecord]:
        return [s for s in self.steps if s.status == "failed"]

    @property
    def is_complete(self) -> bool:
        return all(s.status in ("completed", "skipped") for s in self.steps)

    def add_step(self, step_id: str, objective: str) -> StepRecord:
        step = StepRecord(step_id=step_id, objective=objective)
        self.steps.append(step)
        return step

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_steps": len(self.steps),
            "completed": len(self.completed_steps),
            "failed": len(self.failed_steps),
            "started_at": self.started_at,
        }
