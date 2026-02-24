"""Unit tests for plugin decorator API."""

from typing import Any

import pytest
from nucleusiq.plugins.base import (
    AgentContext,
    BasePlugin,
    ModelRequest,
    ToolRequest,
)
from nucleusiq.plugins.decorators import (
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from nucleusiq.plugins.errors import PluginHalt


class TestBeforeAgentDecorator:
    def test_creates_plugin_instance(self):
        @before_agent
        async def my_hook(ctx):
            return ctx

        assert isinstance(my_hook, BasePlugin)
        assert my_hook.name == "my_hook"

    @pytest.mark.asyncio
    async def test_async_function(self):
        @before_agent
        async def add_meta(ctx: AgentContext):
            ctx.metadata["touched"] = True
            return ctx

        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await add_meta.before_agent(ctx)
        assert result.metadata["touched"]

    @pytest.mark.asyncio
    async def test_sync_function_auto_wrapped(self):
        @before_agent
        def sync_hook(ctx: AgentContext):
            ctx.metadata["sync"] = True
            return ctx

        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await sync_hook.before_agent(ctx)
        assert result.metadata["sync"]

    @pytest.mark.asyncio
    async def test_return_none_means_no_change(self):
        @before_agent
        def observe(ctx: AgentContext) -> None:
            pass  # pure observer

        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await observe.before_agent(ctx)
        assert result is None

    @pytest.mark.asyncio
    async def test_halt(self):
        @before_agent
        async def halt_hook(ctx):
            raise PluginHalt("stopped")

        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        with pytest.raises(PluginHalt):
            await halt_hook.before_agent(ctx)


class TestAfterAgentDecorator:
    @pytest.mark.asyncio
    async def test_modifies_result(self):
        @after_agent
        async def append_suffix(ctx: AgentContext, result: Any):
            return f"{result}_done"

        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await append_suffix.after_agent(ctx, "hello")
        assert result == "hello_done"


class TestBeforeModelDecorator:
    def test_creates_plugin(self):
        @before_model
        async def check(request):
            return None

        assert isinstance(check, BasePlugin)

    @pytest.mark.asyncio
    async def test_return_none_for_observe(self):
        @before_model
        def log(request: ModelRequest) -> None:
            pass  # just observing

        req = ModelRequest(agent_name="a")
        result = await log.before_model(req)
        assert result is None

    @pytest.mark.asyncio
    async def test_return_modified_request(self):
        @before_model
        def downgrade(request: ModelRequest) -> ModelRequest:
            return request.with_(model="gpt-4o-mini")

        req = ModelRequest(agent_name="a", model="gpt-4")
        result = await downgrade.before_model(req)
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_can_halt(self):
        @before_model
        async def limit(request: ModelRequest):
            if request.call_count > 5:
                raise PluginHalt("limit reached")

        req = ModelRequest(agent_name="a", call_count=10)
        with pytest.raises(PluginHalt):
            await limit.before_model(req)


class TestAfterModelDecorator:
    @pytest.mark.asyncio
    async def test_modifies_response(self):
        @after_model
        async def tag_response(request: ModelRequest, response: Any):
            return {"tagged": True, "original": response}

        req = ModelRequest(agent_name="a")
        result = await tag_response.after_model(req, "raw")
        assert result["tagged"]
        assert result["original"] == "raw"


class TestWrapModelCallDecorator:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        @wrap_model_call
        async def passthrough(request, handler):
            return await handler(request)

        req = ModelRequest(agent_name="a", model="t")

        async def fake_handler(r):
            return "llm_result"

        result = await passthrough.wrap_model_call(req, fake_handler)
        assert result == "llm_result"

    @pytest.mark.asyncio
    async def test_short_circuit(self):
        @wrap_model_call
        async def cached(request, handler):
            return "from_cache"

        req = ModelRequest(agent_name="a")
        result = await cached.wrap_model_call(req, lambda r: None)
        assert result == "from_cache"

    @pytest.mark.asyncio
    async def test_modify_request_before_handler(self):
        captured = {}

        @wrap_model_call
        async def downgrade(request: ModelRequest, handler):
            modified = request.with_(model="gpt-4o-mini")
            return await handler(modified)

        req = ModelRequest(agent_name="a", model="gpt-4")

        async def fake_handler(r: ModelRequest):
            captured["model"] = r.model
            return "ok"

        await downgrade.wrap_model_call(req, fake_handler)
        assert captured["model"] == "gpt-4o-mini"


class TestWrapToolCallDecorator:
    @pytest.mark.asyncio
    async def test_passthrough(self):
        @wrap_tool_call
        async def passthrough(request, handler):
            return await handler(request)

        req = ToolRequest(agent_name="a", tool_name="t")

        async def fake_handler(r):
            return "tool_result"

        result = await passthrough.wrap_tool_call(req, fake_handler)
        assert result == "tool_result"

    @pytest.mark.asyncio
    async def test_blocks_tool(self):
        @wrap_tool_call
        async def block_dangerous(request, handler):
            if request.tool_name == "rm_rf":
                return "blocked"
            return await handler(request)

        req = ToolRequest(agent_name="a", tool_name="rm_rf")
        result = await block_dangerous.wrap_tool_call(req, lambda r: None)
        assert result == "blocked"

    @pytest.mark.asyncio
    async def test_sync_wrapper(self):
        @wrap_tool_call
        def sync_wrapper(request, handler):
            return "sync_ok"

        req = ToolRequest(agent_name="a", tool_name="t")
        result = await sync_wrapper.wrap_tool_call(req, lambda r: None)
        assert result == "sync_ok"

    @pytest.mark.asyncio
    async def test_modify_tool_args(self):
        captured = {}

        @wrap_tool_call
        async def modify_args(request: ToolRequest, handler):
            modified = request.with_(tool_args={"x": 99})
            return await handler(modified)

        req = ToolRequest(agent_name="a", tool_name="calc", tool_args={"x": 1})

        async def fake_handler(r: ToolRequest):
            captured["args"] = r.tool_args
            return "ok"

        await modify_args.wrap_tool_call(req, fake_handler)
        assert captured["args"]["x"] == 99


class TestDecoratorOtherHooksPassthrough:
    """Decorated plugins should have default pass-through for non-overridden hooks."""

    @pytest.mark.asyncio
    async def test_before_agent_other_hooks_untouched(self):
        @before_agent
        async def hook(ctx):
            return ctx

        req = ModelRequest(agent_name="a")
        result = await hook.before_model(req)
        assert result is None  # default returns None now

    @pytest.mark.asyncio
    async def test_wrap_tool_other_hooks_untouched(self):
        @wrap_tool_call
        async def hook(request, handler):
            return await handler(request)

        agent_ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await hook.before_agent(agent_ctx)
        assert result is None  # default returns None now
