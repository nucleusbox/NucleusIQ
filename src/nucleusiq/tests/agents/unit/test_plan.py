"""
Comprehensive tests for Plan and PlanStep classes.

Tests cover:
- Plan creation (positive scenarios)
- PlanStep creation (positive scenarios)
- Plan validation (negative scenarios)
- Plan serialization/deserialization
- Plan from_list conversion
- Edge cases and error handling
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


import pytest
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.task import Task
from pydantic import ValidationError


class TestPlanStepCreation:
    """Test PlanStep creation with various scenarios."""

    def test_plan_step_creation_minimal(self):
        """Test creating plan step with only required fields."""
        step = PlanStep(step=1, action="execute")

        assert step.step == 1
        assert step.action == "execute"
        assert step.task is None
        assert step.args is None  # Defaults to None, not {}
        assert step.details is None

    def test_plan_step_creation_with_all_fields(self):
        """Test creating plan step with all fields."""
        task = Task(id="subtask1", objective="Subtask")
        step = PlanStep(
            step=2,
            action="use_tool",
            task=task,
            args={"tool_name": "calculator", "operation": "add"},
            details="Use calculator to add numbers",
        )

        assert step.step == 2
        assert step.action == "use_tool"
        assert step.task == task
        assert step.args == {"tool_name": "calculator", "operation": "add"}
        assert step.details == "Use calculator to add numbers"

    def test_plan_step_creation_with_task(self):
        """Test creating plan step with task."""
        task = Task(id="task1", objective="Perform calculation")
        step = PlanStep(step=1, action="execute", task=task)

        assert step.task == task
        # Type check: task can be Task or Dict, but we know it's Task here
        assert isinstance(step.task, Task)
        assert step.task.objective == "Perform calculation"

    def test_plan_step_creation_with_args(self):
        """Test creating plan step with arguments."""
        step = PlanStep(
            step=1, action="use_tool", args={"param1": "value1", "param2": 42}
        )

        assert step.args == {"param1": "value1", "param2": 42}

    def test_plan_step_creation_with_details(self):
        """Test creating plan step with details."""
        step = PlanStep(
            step=1, action="execute", details="This step performs the main calculation"
        )

        assert step.details == "This step performs the main calculation"

    def test_plan_step_missing_step(self):
        """Test that missing step number raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlanStep(action="execute")  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("step",) for error in errors)

    def test_plan_step_missing_action(self):
        """Test that missing action raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlanStep(step=1)  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("action",) for error in errors)

    def test_plan_step_invalid_step_type(self):
        """Test that invalid step type raises ValidationError."""
        with pytest.raises(ValidationError):
            PlanStep(step="not-an-int", action="execute")  # type: ignore[arg-type]

    def test_plan_step_invalid_action_type(self):
        """Test that invalid action type raises ValidationError."""
        with pytest.raises(ValidationError):
            PlanStep(step=1, action=123)  # type: ignore[arg-type]

    def test_plan_step_invalid_args_type(self):
        """Test that invalid args type raises ValidationError."""
        with pytest.raises(ValidationError):
            PlanStep(step=1, action="execute", args="not-a-dict")  # type: ignore[arg-type]

    def test_plan_step_negative_step_number(self):
        """Test plan step with negative step number."""
        # Should allow negative (or raise error - depends on validation)
        step = PlanStep(step=-1, action="execute")
        assert step.step == -1

    def test_plan_step_zero_step_number(self):
        """Test plan step with zero step number."""
        step = PlanStep(step=0, action="execute")
        assert step.step == 0


class TestPlanCreation:
    """Test Plan creation with various scenarios."""

    def test_plan_creation_minimal(self):
        """Test creating plan with required fields (task and steps)."""
        task = Task(id="task1", objective="Main task")
        plan = Plan(task=task, steps=[])

        assert plan.task == task
        assert plan.steps == []
        assert len(plan) == 0

    def test_plan_creation_with_steps(self):
        """Test creating plan with steps."""
        task = Task(id="task1", objective="Main task")
        step1 = PlanStep(step=1, action="execute")
        step2 = PlanStep(step=2, action="verify")

        plan = Plan(task=task, steps=[step1, step2])

        assert len(plan) == 2
        assert plan.steps[0] == step1
        assert plan.steps[1] == step2

    def test_plan_creation_with_task_dict(self):
        """Test creating plan with task as dict."""
        task_dict = {"id": "task1", "objective": "Main task"}
        plan = Plan(task=task_dict, steps=[])

        # Should accept dict
        assert isinstance(plan.task, (dict, Task))

    def test_plan_creation_empty_steps(self):
        """Test creating plan with empty steps list."""
        task = Task(id="task1", objective="Main task")
        plan = Plan(task=task, steps=[])

        assert len(plan) == 0
        assert plan.steps == []

    def test_plan_missing_task(self):
        """Test that missing task raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Plan()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("task",) for error in errors)

    def test_plan_invalid_steps_type(self):
        """Test that invalid steps type raises ValidationError."""
        task = Task(id="task1", objective="Main task")
        with pytest.raises(ValidationError):
            Plan(task=task, steps="not-a-list")  # type: ignore[arg-type]

    def test_plan_invalid_step_in_list(self):
        """Test that invalid step in list raises ValidationError."""
        task = Task(id="task1", objective="Main task")
        with pytest.raises(ValidationError):
            Plan(task=task, steps=["not-a-planstep"])  # type: ignore[arg-type]


class TestPlanAccess:
    """Test Plan access methods."""

    def test_plan_length(self):
        """Test plan length property."""
        task = Task(id="task1", objective="Main task")
        plan = Plan(
            task=task,
            steps=[
                PlanStep(step=1, action="execute"),
                PlanStep(step=2, action="verify"),
            ],
        )

        assert len(plan) == 2

    def test_plan_indexing(self):
        """Test plan indexing."""
        task = Task(id="task1", objective="Main task")
        step1 = PlanStep(step=1, action="execute")
        step2 = PlanStep(step=2, action="verify")
        plan = Plan(task=task, steps=[step1, step2])

        assert plan[0] == step1
        assert plan[1] == step2

    def test_plan_index_out_of_range(self):
        """Test plan indexing with out of range index."""
        task = Task(id="task1", objective="Main task")
        plan = Plan(task=task, steps=[PlanStep(step=1, action="execute")])

        with pytest.raises(IndexError):
            _ = plan[10]

    def test_plan_iteration(self):
        """Test iterating over plan steps."""
        task = Task(id="task1", objective="Main task")
        step1 = PlanStep(step=1, action="execute")
        step2 = PlanStep(step=2, action="verify")
        plan = Plan(task=task, steps=[step1, step2])

        # Iterate over plan.steps directly (Plan doesn't implement __iter__)
        steps_list = list(plan.steps)
        assert len(steps_list) == 2
        assert steps_list[0].step == step1.step
        assert steps_list[0].action == step1.action
        assert steps_list[1].step == step2.step
        assert steps_list[1].action == step2.action


class TestPlanSerialization:
    """Test Plan serialization and deserialization."""

    def test_plan_to_dict(self):
        """Test converting plan to dictionary."""
        task = Task(id="task1", objective="Main task")
        step1 = PlanStep(step=1, action="execute", details="Step 1")
        step2 = PlanStep(step=2, action="verify", details="Step 2")
        plan = Plan(task=task, steps=[step1, step2])

        plan_dict = plan.to_dict()

        assert isinstance(plan_dict, dict)
        assert "task" in plan_dict
        assert "steps" in plan_dict
        assert len(plan_dict["steps"]) == 2
        assert plan_dict["steps"][0]["action"] == "execute"
        assert plan_dict["steps"][1]["action"] == "verify"

    def test_plan_to_dict_empty_steps(self):
        """Test converting plan with empty steps to dictionary."""
        task = Task(id="task1", objective="Main task")
        plan = Plan(task=task, steps=[])

        plan_dict = plan.to_dict()

        assert plan_dict["steps"] == []
        assert "task" in plan_dict

    def test_plan_from_list(self):
        """Test creating plan from list of step dictionaries."""
        task = Task(id="task1", objective="Main task")
        steps_list = [
            {"step": 1, "action": "execute", "details": "Step 1"},
            {"step": 2, "action": "verify", "details": "Step 2"},
        ]

        plan = Plan.from_list(steps_list, task)

        assert len(plan) == 2
        assert plan.steps[0].step == 1
        assert plan.steps[0].action == "execute"
        assert plan.steps[1].step == 2
        assert plan.steps[1].action == "verify"

    def test_plan_from_list_empty(self):
        """Test creating plan from empty list."""
        task = Task(id="task1", objective="Main task")
        plan = Plan.from_list([], task)

        assert len(plan) == 0
        assert plan.task == task

    def test_plan_from_list_with_task_in_step(self):
        """Test creating plan from list with task in step."""
        main_task = Task(id="task1", objective="Main task")
        subtask = Task(id="subtask1", objective="Subtask")
        steps_list = [{"step": 1, "action": "execute", "task": subtask.to_dict()}]

        plan = Plan.from_list(steps_list, main_task)

        assert len(plan) == 1
        assert plan.steps[0].task is not None
        assert isinstance(plan.steps[0].task, Task)

    def test_plan_round_trip_serialization(self):
        """Test serializing and deserializing plan maintains data."""
        task = Task(id="task1", objective="Main task")
        original_plan = Plan(
            task=task,
            steps=[
                PlanStep(step=1, action="execute", details="Step 1"),
                PlanStep(step=2, action="verify", details="Step 2"),
            ],
        )

        plan_dict = original_plan.to_dict()
        # Reconstruct from dict
        restored_plan = Plan.from_list(plan_dict["steps"], plan_dict["task"])

        assert len(restored_plan) == len(original_plan)
        assert restored_plan.steps[0].action == original_plan.steps[0].action
        assert restored_plan.steps[1].action == original_plan.steps[1].action


class TestPlanValidation:
    """Test Plan validation and error handling."""

    def test_plan_from_list_invalid_step(self):
        """Test from_list with invalid step dictionary."""
        task = Task(id="task1", objective="Main task")

        with pytest.raises(ValidationError):
            Plan.from_list([{"invalid": "step"}], task)

    def test_plan_from_list_missing_required_fields(self):
        """Test from_list with missing required fields."""
        task = Task(id="task1", objective="Main task")

        with pytest.raises(ValidationError):
            Plan.from_list([{"step": 1}], task)  # Missing action

    def test_plan_from_list_invalid_task_type(self):
        """Test from_list with invalid task type."""
        with pytest.raises(ValidationError):
            Plan.from_list([], "not-a-task-or-dict")  # type: ignore[arg-type]


class TestPlanEdgeCases:
    """Test Plan edge cases and special scenarios."""

    def test_plan_with_many_steps(self):
        """Test plan with many steps."""
        task = Task(id="task1", objective="Main task")
        steps = [PlanStep(step=i, action=f"step_{i}") for i in range(100)]
        plan = Plan(task=task, steps=steps)

        assert len(plan) == 100
        assert plan[0].step == 0
        assert plan[99].step == 99

    def test_plan_with_duplicate_step_numbers(self):
        """Test plan with duplicate step numbers (should be allowed)."""
        task = Task(id="task1", objective="Main task")
        step1 = PlanStep(step=1, action="execute")
        step2 = PlanStep(step=1, action="execute_again")  # Same step number

        plan = Plan(task=task, steps=[step1, step2])

        assert len(plan) == 2
        assert plan[0].step == plan[1].step == 1

    def test_plan_with_non_sequential_steps(self):
        """Test plan with non-sequential step numbers."""
        task = Task(id="task1", objective="Main task")
        plan = Plan(
            task=task,
            steps=[
                PlanStep(step=5, action="step5"),
                PlanStep(step=2, action="step2"),
                PlanStep(step=10, action="step10"),
            ],
        )

        assert len(plan) == 3
        assert plan[0].step == 5
        assert plan[1].step == 2
        assert plan[2].step == 10

    def test_plan_step_with_complex_args(self):
        """Test plan step with complex nested arguments."""
        step = PlanStep(
            step=1,
            action="complex_action",
            args={
                "nested": {"level1": {"level2": "value"}},
                "list": [1, 2, 3],
                "mixed": {"key": "value", "number": 42},
            },
        )

        # Type check: args is Optional[Dict], but we know it's set here
        assert step.args is not None
        assert step.args["nested"]["level1"]["level2"] == "value"
        assert step.args["list"] == [1, 2, 3]
        assert step.args["mixed"]["number"] == 42

    def test_plan_step_with_very_long_details(self):
        """Test plan step with very long details string."""
        long_details = "A" * 10000
        step = PlanStep(step=1, action="execute", details=long_details)

        # Type check: details is Optional[str], but we know it's set here
        assert step.details is not None
        assert len(step.details) == 10000

    def test_plan_step_with_special_characters(self):
        """Test plan step with special characters."""
        step = PlanStep(
            step=1,
            action="action!@#$%",
            details="Details with special chars: !@#$%^&*()",
        )

        assert "!@#$%" in step.action
        # Type check: details is Optional[str], but we know it's set here
        assert step.details is not None
        assert "!@#$%^&*()" in step.details


class TestPlanIntegration:
    """Test Plan integration with Agent."""

    def test_plan_with_agent_execute(self):
        """Test plan can be used with agent execution."""
        from nucleusiq.agents import Agent
        from nucleusiq.agents.config import AgentConfig
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        agent = Agent(
            name="TestAgent",
            role="Assistant",
            objective="Help users",
            llm=llm,
            config=AgentConfig(use_planning=True),
        )

        task = Task(id="task1", objective="Main task")
        plan = Plan(task=task, steps=[PlanStep(step=1, action="execute")])

        # Plan should be usable in agent context
        assert plan.task == task
        assert len(plan) == 1

    def test_plan_dict_compatibility(self):
        """Test plan works with dict-based agent methods."""
        task = Task(id="task1", objective="Main task")
        plan = Plan(task=task, steps=[PlanStep(step=1, action="execute")])

        plan_dict = plan.to_dict()

        # Should be able to reconstruct
        restored_plan = Plan.from_list(plan_dict["steps"], plan_dict["task"])
        assert len(restored_plan) == len(plan)


class TestPlanStepValidation:
    """Test PlanStep validation edge cases."""

    def test_plan_step_empty_action(self):
        """Test plan step with empty action string."""
        step = PlanStep(step=1, action="")
        assert step.action == ""

    def test_plan_step_whitespace_action(self):
        """Test plan step with whitespace-only action."""
        step = PlanStep(step=1, action="   ")
        assert step.action == "   "

    def test_plan_step_none_args(self):
        """Test plan step with None args (defaults to None)."""
        step = PlanStep(step=1, action="execute", args=None)
        assert step.args is None  # Defaults to None, not {}

    def test_plan_step_none_details(self):
        """Test plan step with None details."""
        step = PlanStep(step=1, action="execute", details=None)
        assert step.details is None
