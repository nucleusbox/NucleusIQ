"""Integration tests: Agent + Plugin pipeline end-to-end using MockLLM."""

import pytest
from typing import Any, Optional
from unittest.mock import AsyncMock

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.task import Task
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.base_tool import BaseTool

from nucleusiq.plugins.base import (
    BasePlugin,
    AgentContext,
    ModelRequest,
    ToolRequest,
)
from nucleusiq.plugins.decorators import (
    before_agent,
    after_agent,
    before_model,
    after_model,
    wrap_model_call,
    wrap_tool_call,
)
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.plugins.builtin.model_call_limit import ModelCallLimitPlugin
from nucleusiq.plugins.builtin.tool_call_limit import ToolCallLimitPlugin


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def make_agent(mode="direct", plugins=None, tools=None):
    return Agent(
        name="TestBot",
        role="Assistant",
        objective="Help with tests",
        llm=MockLLM(),
        config=AgentConfig(execution_mode=mode, verbose=False),
        plugins=plugins or [],
        tools=tools or [],
    )


def make_task(objective="Hello"):
    return Task.from_dict({"id": "t1", "objective": objective})


def add_tool():
    def add(a: int, b: int) -> int:
        return a + b
    return BaseTool.from_function(add, name="add", description="Add two numbers")


# ------------------------------------------------------------------ #
# Direct Mode + Plugins                                                #
# ------------------------------------------------------------------ #


class TestDirectModeWithPlugins:
    @pytest.mark.asyncio
    async def test_no_plugins_works(self):
        agent = make_agent()
        result = await agent.execute(make_task("Hello"))
        assert "Echo" in result or "Hello" in result

    @pytest.mark.asyncio
    async def test_before_agent_runs(self):
        calls = []

        @before_agent
        async def tracker(ctx: AgentContext):
            calls.append("before_agent")
            return ctx

        agent = make_agent(plugins=[tracker])
        await agent.execute(make_task("Hi"))
        assert "before_agent" in calls

    @pytest.mark.asyncio
    async def test_after_agent_modifies_result(self):
        @after_agent
        async def add_suffix(ctx: AgentContext, result: Any):
            return f"{result} [processed]"

        agent = make_agent(plugins=[add_suffix])
        result = await agent.execute(make_task("Hi"))
        assert "[processed]" in result

    @pytest.mark.asyncio
    async def test_before_agent_halt(self):
        @before_agent
        async def halt(ctx):
            raise PluginHalt("blocked")

        agent = make_agent(plugins=[halt])
        result = await agent.execute(make_task("Hi"))
        assert result == "blocked"

    @pytest.mark.asyncio
    async def test_before_model_runs(self):
        calls = []

        @before_model
        async def tracker(request: ModelRequest):
            calls.append(f"before_model_{request.call_count}")
            return None  # no change

        agent = make_agent(plugins=[tracker])
        await agent.execute(make_task("Hi"))
        assert len(calls) == 1
        assert "before_model_1" in calls

    @pytest.mark.asyncio
    async def test_before_model_observe_returns_none(self):
        """Returning None from before_model should not break execution."""
        @before_model
        def just_log(request: ModelRequest) -> None:
            pass  # pure observer

        agent = make_agent(plugins=[just_log])
        result = await agent.execute(make_task("Hi"))
        assert result  # got some response

    @pytest.mark.asyncio
    async def test_after_model_sees_response(self):
        responses = []

        @after_model
        async def capture(request: ModelRequest, response: Any):
            responses.append(response)
            return response

        agent = make_agent(plugins=[capture])
        await agent.execute(make_task("Hi"))
        assert len(responses) == 1
        assert hasattr(responses[0], "choices")

    @pytest.mark.asyncio
    async def test_wrap_model_call_intercepts(self):
        @wrap_model_call
        async def cache(request: ModelRequest, handler):
            return MockLLM.LLMResponse([
                MockLLM.Choice(MockLLM.Message(content="from_cache"))
            ])

        agent = make_agent(plugins=[cache])
        result = await agent.execute(make_task("Hi"))
        assert result == "from_cache"

    @pytest.mark.asyncio
    async def test_wrap_model_call_with_request_override(self):
        """Handler receives the modified request via .with_()."""
        captured_models = []

        @wrap_model_call
        async def downgrade(request: ModelRequest, handler):
            modified = request.with_(model="gpt-4o-mini")
            captured_models.append(modified.model)
            return await handler(modified)

        agent = make_agent(plugins=[downgrade])
        await agent.execute(make_task("Hi"))
        assert "gpt-4o-mini" in captured_models


# ------------------------------------------------------------------ #
# Standard Mode + Plugins (with tools)                                 #
# ------------------------------------------------------------------ #


class TestStandardModeWithPlugins:
    @pytest.mark.asyncio
    async def test_tool_call_with_no_plugins(self):
        agent = make_agent(mode="standard", tools=[add_tool()])
        result = await agent.execute(make_task("add 3 and 5"))
        assert "8" in str(result)

    @pytest.mark.asyncio
    async def test_wrap_tool_call_intercepts(self):
        @wrap_tool_call
        async def block_all(request: ToolRequest, handler):
            return "tool_blocked"

        agent = make_agent(mode="standard", plugins=[block_all], tools=[add_tool()])
        result = await agent.execute(make_task("add 3 and 5"))
        assert "tool_blocked" in str(result)

    @pytest.mark.asyncio
    async def test_before_and_after_agent_with_tools(self):
        calls = []

        @before_agent
        async def before(ctx):
            calls.append("before")
            return ctx

        @after_agent
        async def after(ctx, result):
            calls.append("after")
            return result

        agent = make_agent(
            mode="standard",
            plugins=[before, after],
            tools=[add_tool()],
        )
        await agent.execute(make_task("add 3 and 5"))
        assert calls == ["before", "after"]


# ------------------------------------------------------------------ #
# Multiple Plugins + Ordering                                          #
# ------------------------------------------------------------------ #


class TestMultiplePlugins:
    @pytest.mark.asyncio
    async def test_execution_order(self):
        order = []

        @before_agent
        async def first(ctx):
            order.append("first")
            return ctx

        @before_agent
        async def second(ctx):
            order.append("second")
            return ctx

        agent = make_agent(plugins=[first, second])
        await agent.execute(make_task("Hi"))
        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_after_agent_chain(self):
        @after_agent
        async def add_a(ctx, result):
            return f"{result}_A"

        @after_agent
        async def add_b(ctx, result):
            return f"{result}_B"

        agent = make_agent(plugins=[add_a, add_b])
        result = await agent.execute(make_task("Hi"))
        assert result.endswith("_A_B")


# ------------------------------------------------------------------ #
# Builtin Plugins with Agent                                           #
# ------------------------------------------------------------------ #


class TestBuiltinPluginsWithAgent:
    @pytest.mark.asyncio
    async def test_model_call_limit_with_direct_mode(self):
        """ModelCallLimitPlugin with max_calls=0 should halt immediately."""
        agent = make_agent(plugins=[ModelCallLimitPlugin(max_calls=0)])
        result = await agent.execute(make_task("Hi"))
        assert "limit exceeded" in str(result).lower()


# ------------------------------------------------------------------ #
# Class-based Plugin with Agent                                        #
# ------------------------------------------------------------------ #


class TestClassBasedPlugin:
    @pytest.mark.asyncio
    async def test_multi_hook_plugin(self):
        class AuditPlugin(BasePlugin):
            def __init__(self):
                self.events = []

            @property
            def name(self):
                return "audit"

            async def before_agent(self, ctx):
                self.events.append(f"agent_start:{ctx.agent_name}")
                return ctx

            async def after_agent(self, ctx, result):
                self.events.append(f"agent_end:{ctx.agent_name}")
                return result

            async def before_model(self, request):
                self.events.append(f"model_call:{request.call_count}")
                return None  # observe only

        audit = AuditPlugin()
        agent = make_agent(plugins=[audit])
        await agent.execute(make_task("Hi"))

        assert "agent_start:TestBot" in audit.events
        assert "agent_end:TestBot" in audit.events
        assert any(e.startswith("model_call:") for e in audit.events)

    @pytest.mark.asyncio
    async def test_default_name_from_class(self):
        class MyCustomPlugin(BasePlugin):
            async def before_model(self, request):
                return None

        p = MyCustomPlugin()
        assert p.name == "MyCustomPlugin"


# ------------------------------------------------------------------ #
# Counter Reset Between Executions                                     #
# ------------------------------------------------------------------ #


class TestCounterReset:
    @pytest.mark.asyncio
    async def test_counters_reset_between_executions(self):
        call_counts = []

        @before_model
        async def track(request: ModelRequest):
            call_counts.append(request.call_count)
            return None

        agent = make_agent(plugins=[track])
        await agent.execute(make_task("First"))
        await agent.execute(make_task("Second"))

        assert call_counts == [1, 1]
