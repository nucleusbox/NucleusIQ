"""Tests for PluginManager → ExecutionTracer wiring (PluginEvent recording)."""

from __future__ import annotations

from typing import Any

import pytest
from nucleusiq.agents.agent_result import PluginEvent
from nucleusiq.agents.observability import DefaultExecutionTracer
from nucleusiq.plugins.base import AgentContext, BasePlugin, ModelRequest
from nucleusiq.plugins.manager import PluginManager


class _SpyPlugin(BasePlugin):
    """Plugin that records which hooks were called."""

    def __init__(self):
        self.calls: list[str] = []

    async def before_agent(self, ctx: AgentContext) -> AgentContext | None:
        self.calls.append("before_agent")
        return ctx

    async def after_agent(self, ctx: AgentContext, result: Any) -> Any:
        self.calls.append("after_agent")
        return result

    async def before_model(self, request: ModelRequest) -> ModelRequest | None:
        self.calls.append("before_model")
        return request

    async def after_model(self, request: ModelRequest, response: Any) -> Any:
        self.calls.append("after_model")
        return response


def _make_agent_ctx() -> AgentContext:
    return AgentContext(
        agent_name="test-agent",
        task=None,
        state=None,
        config=None,
        memory=None,
    )


def _make_model_request() -> ModelRequest:
    return ModelRequest(
        model="test-model",
        messages=[],
        tools=None,
        max_output_tokens=1024,
        call_count=1,
        agent_name="test-agent",
    )


class TestPluginEventTracing:
    @pytest.mark.asyncio
    async def test_before_agent_records_event(self):
        tracer = DefaultExecutionTracer()
        pm = PluginManager([_SpyPlugin()])
        pm._tracer = tracer

        ctx = _make_agent_ctx()
        await pm.run_before_agent(ctx)

        assert len(tracer.plugin_events) == 1
        event = tracer.plugin_events[0]
        assert event.plugin_name == "_SpyPlugin"
        assert event.hook == "before_agent"
        assert event.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_after_agent_records_event(self):
        tracer = DefaultExecutionTracer()
        pm = PluginManager([_SpyPlugin()])
        pm._tracer = tracer

        ctx = _make_agent_ctx()
        await pm.run_after_agent(ctx, "result")

        assert len(tracer.plugin_events) == 1
        assert tracer.plugin_events[0].hook == "after_agent"

    @pytest.mark.asyncio
    async def test_before_model_records_event(self):
        tracer = DefaultExecutionTracer()
        pm = PluginManager([_SpyPlugin()])
        pm._tracer = tracer

        req = _make_model_request()
        await pm.run_before_model(req)

        assert len(tracer.plugin_events) == 1
        assert tracer.plugin_events[0].hook == "before_model"
        assert tracer.plugin_events[0].action == "modified"

    @pytest.mark.asyncio
    async def test_after_model_records_event(self):
        tracer = DefaultExecutionTracer()
        pm = PluginManager([_SpyPlugin()])
        pm._tracer = tracer

        req = _make_model_request()
        await pm.run_after_model(req, "response")

        assert len(tracer.plugin_events) == 1
        assert tracer.plugin_events[0].hook == "after_model"

    @pytest.mark.asyncio
    async def test_multiple_plugins_record_multiple_events(self):
        tracer = DefaultExecutionTracer()
        pm = PluginManager([_SpyPlugin(), _SpyPlugin()])
        pm._tracer = tracer

        ctx = _make_agent_ctx()
        await pm.run_before_agent(ctx)

        assert len(tracer.plugin_events) == 2

    @pytest.mark.asyncio
    async def test_no_tracer_no_error(self):
        pm = PluginManager([_SpyPlugin()])

        ctx = _make_agent_ctx()
        result = await pm.run_before_agent(ctx)
        assert result is not None

    @pytest.mark.asyncio
    async def test_event_has_correct_fields(self):
        tracer = DefaultExecutionTracer()
        pm = PluginManager([_SpyPlugin()])
        pm._tracer = tracer

        ctx = _make_agent_ctx()
        await pm.run_before_agent(ctx)

        event = tracer.plugin_events[0]
        assert isinstance(event, PluginEvent)
        assert isinstance(event.duration_ms, float)
        assert event.plugin_name == "_SpyPlugin"
