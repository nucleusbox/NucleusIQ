"""Unit tests for execution_tracer helpers and tracer implementations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from nucleusiq.agents.agent_result import LLMCallRecord, PluginEvent, ValidationRecord
from nucleusiq.agents.chat_models import ToolCallRequest
from nucleusiq.agents.observability import (
    DefaultExecutionTracer,
    NoOpTracer,
    build_llm_call_record,
    build_llm_call_record_from_stream,
    build_tool_call_record,
)
from nucleusiq.agents.observability._response_parser import extract_tool_calls


class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, choices, usage=None, model=None):
        self.choices = choices
        self.usage = usage
        self.model = model


def test_safe_int_via_build_llm_call_record():
    usage = {"prompt_tokens": "10", "completion_tokens": None, "total_tokens": "bad"}
    r = _Resp([_Choice(_Msg(content="hi"))], usage=usage, model="m1")
    rec = build_llm_call_record(
        r, call_round=1, purpose="main", duration_ms=5.0, model="override"
    )
    assert rec.model == "override"
    assert rec.prompt_tokens == 10
    assert rec.completion_tokens == 0
    assert rec.total_tokens == 10


def test_extract_tool_calls_openai_style():
    tc = {
        "id": "1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a": 1}'},
    }
    r = _Resp([_Choice(_Msg(content=None, tool_calls=[tc]))])
    raw = extract_tool_calls(r)
    assert len(raw) == 1
    assert raw[0]["function"]["name"] == "add"


def test_extract_tool_calls_function_call_legacy():
    r = _Resp(
        [
            _Choice(
                _Msg(
                    content=None,
                    function_call={"name": "legacy", "arguments": "{}"},
                )
            )
        ]
    )
    raw = extract_tool_calls(r)
    assert len(raw) == 1
    assert raw[0]["function"]["name"] == "legacy"


def test_extract_tool_calls_gemini_parts():
    r = _Resp(
        [
            _Choice(
                _Msg(
                    content=[
                        {
                            "function_call": {
                                "name": "fn",
                                "args": {"x": 1},
                                "id": "g1",
                            }
                        }
                    ]
                )
            )
        ]
    )
    raw = extract_tool_calls(r)
    assert len(raw) == 1
    assert raw[0]["function"]["name"] == "fn"
    assert json.loads(raw[0]["function"]["arguments"]) == {"x": 1}


def test_build_llm_call_record_from_stream():
    meta = {
        "tool_calls": [{"id": "a"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    rec = build_llm_call_record_from_stream(
        meta, call_round=2, purpose="tool_loop", duration_ms=9.0, model="stream-m"
    )
    assert rec.round == 2
    assert rec.purpose == "tool_loop"
    assert rec.tool_call_count == 1
    assert rec.has_tool_calls is True
    assert rec.duration_ms == 9.0
    assert rec.model == "stream-m"


def test_build_tool_call_record_parses_args():
    tc = ToolCallRequest(id="x", name="t", arguments='{"k": "v"}')
    rec = build_tool_call_record(tc, success=True, duration_ms=1.0, round=3)
    assert rec.args == {"k": "v"}
    assert rec.round == 3


def test_default_tracer_and_reset():
    t = DefaultExecutionTracer()
    tc = ToolCallRequest(name="n", arguments="{}")
    t.record_tool_call(build_tool_call_record(tc, success=True, duration_ms=1.0))
    t.record_llm_call(
        LLMCallRecord(
            round=1,
            purpose="main",
            duration_ms=2.0,
        )
    )
    t.record_warning("w")
    t.record_plugin_event(
        PluginEvent(plugin_name="p", hook="h", action="executed", duration_ms=0.0)
    )
    t.record_validation(
        ValidationRecord(attempt=1, valid=True, layer="critic", reason="ok")
    )
    t.set_autonomous_detail(foo=1)
    assert len(t.tool_calls) == 1
    assert len(t.llm_calls) == 1
    assert t.warnings == ("w",)
    assert len(t.plugin_events) == 1
    assert len(t.validations) == 1
    assert t.autonomous_detail == {"foo": 1}
    t.reset()
    assert t.tool_calls == ()
    assert t.autonomous_detail is None


def test_noop_tracer():
    n = NoOpTracer()
    tc = ToolCallRequest(name="n", arguments="{}")
    n.record_tool_call(build_tool_call_record(tc, success=True, duration_ms=1.0))
    assert n.tool_calls == ()
    assert n.llm_calls == ()
