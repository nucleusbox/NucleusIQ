# src/nucleusiq/agents/plan.py
"""
Plan classes for representing execution plans.

Plan = HOW to break down a task into steps
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from nucleusiq.agents.task import Task


class PlanStepResponse(BaseModel):
    """
    Represents a single step in a plan response from LLM.
    
    This is the structured format that LLMs return when generating plans.
    It's then converted to PlanStep (which includes the task reference).
    
    Example:
        ```python
        step_response = PlanStepResponse(
            step=1,
            action="execute",
            args={"param": "value"},
            details="Execute the main task"
        )
        ```
    """
    
    step: int = Field(..., description="Step number (1-indexed)")
    action: str = Field(..., description="Action to take (e.g., 'execute', tool name)")
    args: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Action arguments")
    details: Optional[str] = Field(default=None, description="Step details or description")
    
    def to_plan_step(self, task: Optional[Union[Task, Dict[str, Any]]] = None) -> 'PlanStep':
        """
        Convert PlanStepResponse to PlanStep.
        
        Args:
            task: Optional task to associate with this step
            
        Returns:
            PlanStep instance
        """
        return PlanStep(
            step=self.step,
            action=self.action,
            task=task,
            args=self.args,
            details=self.details
        )


class PlanResponse(BaseModel):
    """
    Represents a structured plan response from LLM.
    
    This is the format that LLMs return when generating execution plans.
    It's then converted to a Plan (which includes the task reference).
    
    Example:
        ```python
        plan_response = PlanResponse(
            steps=[
                PlanStepResponse(step=1, action="execute", details="Step 1"),
                PlanStepResponse(step=2, action="tool_name", args={"param": "value"})
            ]
        )
        ```
    """
    
    steps: List[PlanStepResponse] = Field(..., description="List of plan steps")
    
    def to_plan(self, task: Union[Task, Dict[str, Any]]) -> 'Plan':
        """
        Convert PlanResponse to Plan.
        
        Args:
            task: Task to associate with the plan
            
        Returns:
            Plan instance
        """
        plan_steps = [step.to_plan_step(task=task) for step in self.steps]
        return Plan(steps=plan_steps, task=task)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanResponse':
        """
        Create PlanResponse from dictionary.
        
        Args:
            data: Dictionary with 'steps' key containing list of step dicts
            
        Returns:
            PlanResponse instance
        """
        if "steps" not in data:
            raise ValueError("PlanResponse must have 'steps' key")
        
        steps = [PlanStepResponse(**step) if isinstance(step, dict) else step for step in data["steps"]]
        return cls(steps=steps)


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

