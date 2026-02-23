"""
Tests for ProgressTracker — step-by-step execution tracking.

Covers:
- StepRecord lifecycle (pending → executing → completed/failed)
- ExecutionProgress aggregation and properties
- Summary generation
"""

from nucleusiq.agents.components.progress import (
    StepRecord,
    ExecutionProgress,
)


class TestStepRecord:

    def test_initial_state(self):
        step = StepRecord(step_id="s1", objective="Do something")
        assert step.status == "pending"
        assert step.attempts == 0
        assert step.result is None
        assert step.error is None

    def test_mark_executing(self):
        step = StepRecord(step_id="s1", objective="X")
        step.mark_executing()
        assert step.status == "executing"
        assert step.attempts == 1
        assert step.started_at is not None

    def test_mark_completed(self):
        step = StepRecord(step_id="s1", objective="X")
        step.mark_executing()
        step.mark_completed("result 42")
        assert step.status == "completed"
        assert step.result == "result 42"
        assert step.completed_at is not None

    def test_mark_failed(self):
        step = StepRecord(step_id="s1", objective="X")
        step.mark_executing()
        step.mark_failed("timeout")
        assert step.status == "failed"
        assert step.error == "timeout"

    def test_multiple_attempts(self):
        step = StepRecord(step_id="s1", objective="X")
        step.mark_executing()
        step.mark_failed("error 1")
        step.mark_executing()
        step.mark_completed("ok")
        assert step.attempts == 2
        assert step.status == "completed"


class TestExecutionProgress:

    def test_add_step(self):
        progress = ExecutionProgress(task_id="t1")
        step = progress.add_step("s1", "Step 1")
        assert len(progress.steps) == 1
        assert step.step_id == "s1"
        assert step.status == "pending"

    def test_completed_steps(self):
        progress = ExecutionProgress(task_id="t1")
        s1 = progress.add_step("s1", "A")
        s2 = progress.add_step("s2", "B")
        s1.mark_executing()
        s1.mark_completed("done")
        assert len(progress.completed_steps) == 1
        assert len(progress.failed_steps) == 0

    def test_failed_steps(self):
        progress = ExecutionProgress(task_id="t1")
        s1 = progress.add_step("s1", "A")
        s1.mark_executing()
        s1.mark_failed("error")
        assert len(progress.failed_steps) == 1

    def test_is_complete(self):
        progress = ExecutionProgress(task_id="t1")
        s1 = progress.add_step("s1", "A")
        s2 = progress.add_step("s2", "B")
        assert not progress.is_complete
        s1.mark_executing()
        s1.mark_completed("ok")
        s2.status = "skipped"
        assert progress.is_complete

    def test_summary(self):
        progress = ExecutionProgress(task_id="t1")
        s1 = progress.add_step("s1", "A")
        s2 = progress.add_step("s2", "B")
        s1.mark_executing()
        s1.mark_completed("ok")
        s2.mark_executing()
        s2.mark_failed("err")

        summary = progress.summary()
        assert summary["task_id"] == "t1"
        assert summary["total_steps"] == 2
        assert summary["completed"] == 1
        assert summary["failed"] == 1
