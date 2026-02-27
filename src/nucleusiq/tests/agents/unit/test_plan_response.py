"""Additional tests for PlanResponse/PlanStepResponse conversions."""

from __future__ import annotations

import pytest
from nucleusiq.agents.plan import PlanResponse, PlanStepResponse
from nucleusiq.agents.task import Task


def test_plan_step_response_to_plan_step():
    step = PlanStepResponse(
        step=1,
        action="execute",
        args={"x": 1},
        details="detail",
    )
    out = step.to_plan_step(task={"id": "t", "objective": "obj"})
    assert out.step == 1
    assert out.action == "execute"
    assert out.args == {"x": 1}
    assert out.details == "detail"


def test_plan_response_to_plan():
    pr = PlanResponse(
        steps=[
            PlanStepResponse(step=1, action="execute"),
            PlanStepResponse(step=2, action="verify", details="check"),
        ]
    )
    task = Task(id="t1", objective="do")
    plan = pr.to_plan(task)
    assert len(plan.steps) == 2
    assert plan.task == task


def test_plan_response_from_dict_missing_steps_raises():
    with pytest.raises(ValueError, match="must have 'steps'"):
        PlanResponse.from_dict({"foo": []})


def test_plan_response_from_dict_supports_mixed_step_types():
    s = PlanStepResponse(step=2, action="verify")
    pr = PlanResponse.from_dict(
        {
            "steps": [
                {"step": 1, "action": "execute", "args": {"k": "v"}},
                s,
            ]
        }
    )
    assert len(pr.steps) == 2
    assert pr.steps[0].action == "execute"
    assert pr.steps[1].action == "verify"
