# src/nucleusiq/agents/plan.py
"""
Plan classes for representing execution plans.

Plan = HOW to break down a task into steps
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from nucleusiq.agents.task import Task


class PlanStep(BaseModel):
    """
    Represents a single step in an execution plan.
    
    Example:
        ```python
        step = PlanStep(
            step=1,
            action="execute",
            task=task,
            details="Execute the main task"
        )
        ```
    """
    
    step: int = Field(..., description="Step number (1-indexed)")
    action: str = Field(..., description="Action to take (e.g., 'execute', tool name)")
    task: Optional[Union[Task, Dict[str, Any]]] = Field(default=None, description="Task for this step")
    args: Optional[Dict[str, Any]] = Field(default=None, description="Action arguments")
    details: Optional[str] = Field(default=None, description="Step details or description")


class Plan(BaseModel):
    """
    Represents an execution plan with multiple steps.
    
    Plan breaks down a complex task into smaller, manageable steps.
    
    Example:
        ```python
        plan = Plan(
            steps=[
                PlanStep(step=1, action="execute", task=task),
                PlanStep(step=2, action="execute", task=task),
            ],
            task=task
        )
        
        # Or from list (backward compatible)
        plan = Plan.from_list(
            steps=[{"step": 1, "action": "execute", "task": task}],
            task=task
        )
        ```
    """
    
    steps: List[PlanStep] = Field(..., description="List of plan steps")
    task: Union[Task, Dict[str, Any]] = Field(..., description="Original task")
    
    @classmethod
    def from_list(cls, steps: List[Dict[str, Any]], task: Union[Task, Dict[str, Any]]) -> 'Plan':
        """
        Create Plan from list of step dictionaries (backward compatibility).
        
        Args:
            steps: List of step dictionaries
            task: Task instance or dictionary
            
        Returns:
            Plan instance
        """
        if isinstance(task, dict):
            task = Task.from_dict(task)
        plan_steps = [PlanStep(**step) for step in steps]
        return cls(steps=plan_steps, task=task)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert Plan to list of dictionaries (backward compatibility).
        
        Returns:
            List of step dictionaries
        """
        return [step.model_dump() for step in self.steps]
    
    def __len__(self) -> int:
        """Return number of steps in the plan."""
        return len(self.steps)
    
    def __getitem__(self, index: int) -> PlanStep:
        """Allow indexing into plan steps."""
        return self.steps[index]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the Plan object to a dictionary."""
        return self.model_dump(mode='json')

