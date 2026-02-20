"""Unit tests for BasePlugin and request models."""

import pytest
from typing import Optional, Any

from nucleusiq.plugins.base import (
    BasePlugin,
    AgentContext,
    ModelRequest,
    ToolRequest,
)


# ------------------------------------------------------------------ #
# AgentContext                                                         #
# ------------------------------------------------------------------ #


class TestAgentContext:
    def test_minimal_creation(self):
        ctx = AgentContext(agent_name="test", task="do something", state="idle", config={})
        assert ctx.agent_name == "test"
        assert ctx.metadata == {}
        assert ctx.memory is None

    def test_metadata_mutable(self):
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        ctx.metadata["key"] = "value"
        assert ctx.metadata["key"] == "value"


# ------------------------------------------------------------------ #
# ModelRequest                                                         #
# ------------------------------------------------------------------ #


class TestModelRequest:
    def test_defaults(self):
        req = ModelRequest(agent_name="a")
        assert req.model == "default"
        assert req.messages == []
        assert req.tools is None
        assert req.max_tokens == 1024
        assert req.call_count == 0
        assert req.metadata == {}
        assert req.extra_kwargs == {}

    def test_with_values(self):
        req = ModelRequest(
            agent_name="bot",
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            call_count=5,
            max_tokens=2048,
        )
        assert req.call_count == 5
        assert req.model == "gpt-4"
        assert req.max_tokens == 2048

    def test_with_creates_copy(self):
        original = ModelRequest(agent_name="a", model="gpt-4", call_count=3)
        modified = original.with_(model="gpt-4o-mini", max_tokens=512)

        assert modified.model == "gpt-4o-mini"
        assert modified.max_tokens == 512
        assert modified.call_count == 3  # unchanged
        assert original.model == "gpt-4"  # original unchanged

    def test_with_preserves_extra_kwargs(self):
        req = ModelRequest(agent_name="a", extra_kwargs={"temperature": 0.7})
        modified = req.with_(model="gpt-4o")
        assert modified.extra_kwargs["temperature"] == 0.7

    def test_to_call_kwargs(self):
        req = ModelRequest(
            agent_name="a",
            model="gpt-4",
            messages=[],
            tools=None,
            max_tokens=1024,
            extra_kwargs={"temperature": 0.5},
        )
        kwargs = req.to_call_kwargs()
        assert kwargs["model"] == "gpt-4"
        assert kwargs["messages"] == []
        assert kwargs["tools"] is None
        assert kwargs["max_tokens"] == 1024
        assert kwargs["temperature"] == 0.5


# ------------------------------------------------------------------ #
# ToolRequest                                                          #
# ------------------------------------------------------------------ #


class TestToolRequest:
    def test_minimal(self):
        req = ToolRequest(tool_name="calc", agent_name="a")
        assert req.tool_args == {}
        assert req.tool_call_id is None
        assert req.call_count == 0

    def test_with_args(self):
        req = ToolRequest(
            agent_name="bot",
            tool_name="search",
            tool_args={"query": "test"},
            tool_call_id="tc_123",
            call_count=3,
        )
        assert req.tool_args["query"] == "test"
        assert req.tool_call_id == "tc_123"

    def test_with_creates_copy(self):
        original = ToolRequest(agent_name="a", tool_name="calc", tool_args={"x": 1})
        modified = original.with_(tool_args={"x": 99})
        assert modified.tool_args["x"] == 99
        assert original.tool_args["x"] == 1

    def test_to_tool_call_request(self):
        req = ToolRequest(
            agent_name="a",
            tool_name="calc",
            tool_args={"x": 1},
            tool_call_id="tc_1",
        )
        tc = req.to_tool_call_request()
        assert tc.name == "calc"
        assert tc.id == "tc_1"
        assert '"x": 1' in tc.arguments


# ------------------------------------------------------------------ #
# BasePlugin                                                           #
# ------------------------------------------------------------------ #


class ConcretePlugin(BasePlugin):
    pass


class NamedPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "custom_name"


class TestBasePlugin:
    def test_default_name_is_class_name(self):
        p = ConcretePlugin()
        assert p.name == "ConcretePlugin"

    def test_custom_name(self):
        p = NamedPlugin()
        assert p.name == "custom_name"

    def test_repr(self):
        p = NamedPlugin()
        assert "NamedPlugin" in repr(p)
        assert "custom_name" in repr(p)

    @pytest.mark.asyncio
    async def test_default_before_agent_returns_none(self):
        p = ConcretePlugin()
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await p.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_default_after_agent_passthrough(self):
        p = ConcretePlugin()
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await p.after_agent(ctx, "hello")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_default_before_model_returns_none(self):
        p = ConcretePlugin()
        req = ModelRequest(agent_name="a")
        result = await p.before_model(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_default_after_model_passthrough(self):
        p = ConcretePlugin()
        req = ModelRequest(agent_name="a")
        response = {"choices": []}
        result = await p.after_model(req, response)
        assert result is response

    @pytest.mark.asyncio
    async def test_default_wrap_model_call(self):
        p = ConcretePlugin()
        req = ModelRequest(agent_name="a")
        called = False

        async def fake_handler(r):
            nonlocal called
            called = True
            return "llm_result"

        result = await p.wrap_model_call(req, fake_handler)
        assert called
        assert result == "llm_result"

    @pytest.mark.asyncio
    async def test_default_wrap_tool_call(self):
        p = ConcretePlugin()
        req = ToolRequest(agent_name="a", tool_name="t")
        called = False

        async def fake_handler(r):
            nonlocal called
            called = True
            return "tool_result"

        result = await p.wrap_tool_call(req, fake_handler)
        assert called
        assert result == "tool_result"
