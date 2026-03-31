"""Tests for AgentResult model (v0.7.2).

Verifies:
- Immutability (frozen=True)
- Backward compatibility (__str__, __bool__)
- All sub-models
- Convenience properties
- Serialization
"""

import pytest
from nucleusiq.agents.agent_result import (
    AgentResult,
    AutonomousDetail,
    LLMCallRecord,
    MemorySnapshot,
    PluginEvent,
    ResultStatus,
    ToolCallRecord,
    ValidationRecord,
)
from pydantic import ValidationError


class TestResultStatus:
    def test_values(self):
        assert ResultStatus.SUCCESS == "success"
        assert ResultStatus.ERROR == "error"
        assert ResultStatus.HALTED == "halted"

    def test_all_values_present(self):
        assert len(ResultStatus) == 3


class TestAgentResultCreation:
    def test_minimal(self):
        r = AgentResult(agent_id="a1", agent_name="Bot", task_id="t1", mode="direct")
        assert r.agent_id == "a1"
        assert r.status == ResultStatus.SUCCESS
        assert r.output is None
        assert r.tool_calls == ()
        assert r.llm_calls == ()
        assert r.warnings == ()
        assert r.metadata == {}

    def test_with_output(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="standard",
            output="Hello, world!",
        )
        assert r.output == "Hello, world!"

    def test_error_result(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="standard",
            status=ResultStatus.ERROR,
            error="Something broke",
            error_type="RuntimeError",
        )
        assert r.is_error
        assert not r.is_halted
        assert r.error == "Something broke"
        assert r.error_type == "RuntimeError"

    def test_halted_result(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            status=ResultStatus.HALTED,
            output="Plugin stopped me",
        )
        assert r.is_halted
        assert not r.is_error
        assert str(r) == "Plugin stopped me"

    def test_created_at_auto_populated(self):
        r = AgentResult(agent_id="a1", agent_name="Bot", task_id="t1", mode="direct")
        assert r.created_at != ""
        assert "T" in r.created_at


class TestAgentResultImmutability:
    def test_frozen(self):
        r = AgentResult(agent_id="a1", agent_name="Bot", task_id="t1", mode="direct")
        with pytest.raises(ValidationError):
            r.status = ResultStatus.ERROR  # type: ignore

    def test_tuple_sequences_are_immutable(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            tool_calls=(ToolCallRecord(tool_name="calc"),),
        )
        assert isinstance(r.tool_calls, tuple)
        assert len(r.tool_calls) == 1


class TestBackwardCompatibility:
    def test_str_returns_output(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            output="Echo: Hello",
        )
        assert str(r) == "Echo: Hello"

    def test_str_returns_empty_for_none(self):
        r = AgentResult(agent_id="a1", agent_name="Bot", task_id="t1", mode="direct")
        assert str(r) == ""

    def test_bool_true_on_success(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            status=ResultStatus.SUCCESS,
        )
        assert bool(r) is True

    def test_bool_false_on_error(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            status=ResultStatus.ERROR,
        )
        assert bool(r) is False

    def test_bool_false_on_halted(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            status=ResultStatus.HALTED,
        )
        assert bool(r) is False

    def test_string_containment_via_str(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            output="Echo: Hello World",
        )
        assert "Echo:" in str(r)
        assert "Hello" in str(r)


class TestConvenienceProperties:
    def test_tool_call_count(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="standard",
            tool_calls=(
                ToolCallRecord(tool_name="a"),
                ToolCallRecord(tool_name="b"),
            ),
        )
        assert r.tool_call_count == 2

    def test_failed_tool_calls(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="standard",
            tool_calls=(
                ToolCallRecord(tool_name="ok", success=True),
                ToolCallRecord(tool_name="bad", success=False, error="timeout"),
                ToolCallRecord(tool_name="ok2", success=True),
            ),
        )
        failed = r.failed_tool_calls
        assert len(failed) == 1
        assert failed[0].tool_name == "bad"


class TestSubModels:
    def test_tool_call_record(self):
        r = ToolCallRecord(
            tool_name="calc",
            tool_call_id="call_1",
            args={"a": 1, "b": 2},
            result=3,
            success=True,
            duration_ms=12.5,
            round=1,
        )
        assert r.tool_name == "calc"
        assert r.success is True

    def test_llm_call_record(self):
        r = LLMCallRecord(
            round=1,
            purpose="main",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert r.total_tokens == 150

    def test_plugin_event(self):
        e = PluginEvent(
            plugin_name="ModelCallLimit",
            hook="before_model",
            action="halted",
            detail="Call count exceeded 10",
        )
        assert e.action == "halted"

    def test_memory_snapshot(self):
        m = MemorySnapshot(
            strategy="sliding_window",
            message_count=5,
            messages=({"role": "user", "content": "hi"},),
        )
        assert m.message_count == 5
        assert len(m.messages) == 1

    def test_validation_record(self):
        v = ValidationRecord(
            attempt=1, valid=True, layer="deterministic", reason="pass"
        )
        assert v.valid is True

    def test_autonomous_detail(self):
        a = AutonomousDetail(
            attempts=3,
            max_attempts=5,
            complexity="COMPLEX",
            sub_tasks=("step1", "step2"),
            validations=(ValidationRecord(attempt=1, valid=False, reason="bad"),),
            refined=True,
        )
        assert a.attempts == 3
        assert len(a.sub_tasks) == 2
        assert a.refined is True


class TestSerialization:
    def test_summary(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
            output="Hello",
        )
        d = r.summary()
        assert d["agent_id"] == "a1"
        assert d["output"] == "Hello"
        assert "memory_snapshot" not in d

    def test_model_dump_json(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="direct",
        )
        json_str = r.model_dump_json()
        assert '"agent_id":"a1"' in json_str

    def test_display(self):
        r = AgentResult(
            agent_id="a1",
            agent_name="Bot",
            task_id="t1",
            mode="standard",
            output="42",
            duration_ms=150.5,
        )
        text = r.display()
        assert "Bot" in text
        assert "150.5ms" in text
        assert "42" in text
