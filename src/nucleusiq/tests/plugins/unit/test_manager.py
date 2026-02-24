"""Unit tests for PluginManager."""

from unittest.mock import AsyncMock

import pytest
from nucleusiq.plugins.base import (
    AgentContext,
    BasePlugin,
    ModelRequest,
    ToolRequest,
)
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.plugins.manager import PluginManager

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


class TrackingPlugin(BasePlugin):
    """Plugin that records which hooks were called."""

    def __init__(self, plugin_name: str = "tracker") -> None:
        self._name = plugin_name
        self.calls: list = []

    @property
    def name(self) -> str:
        return self._name

    async def before_agent(self, ctx):
        self.calls.append("before_agent")
        return ctx

    async def after_agent(self, ctx, result):
        self.calls.append("after_agent")
        return result

    async def before_model(self, request):
        self.calls.append("before_model")
        return request

    async def after_model(self, request, response):
        self.calls.append("after_model")
        return response

    async def wrap_model_call(self, request, handler):
        self.calls.append("wrap_model_call_enter")
        result = await handler(request)
        self.calls.append("wrap_model_call_exit")
        return result

    async def wrap_tool_call(self, request, handler):
        self.calls.append("wrap_tool_call_enter")
        result = await handler(request)
        self.calls.append("wrap_tool_call_exit")
        return result


class HaltPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "halter"

    async def before_agent(self, ctx):
        raise PluginHalt("agent_halted")

    async def before_model(self, request):
        raise PluginHalt("model_halted")


class ModifyingPlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "modifier"

    async def before_agent(self, ctx):
        ctx.metadata["modified_by"] = "modifier"
        return ctx

    async def after_agent(self, ctx, result):
        return f"modified: {result}"

    async def before_model(self, request):
        return request.with_(metadata={**request.metadata, "model_modified": True})

    async def after_model(self, request, response):
        return {"wrapped": response}


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #


class TestPluginManagerInit:
    def test_empty(self):
        pm = PluginManager()
        assert pm.plugins == []
        assert not pm.has_plugins()

    def test_with_plugins(self):
        p = TrackingPlugin()
        pm = PluginManager([p])
        assert pm.has_plugins()
        assert len(pm.plugins) == 1

    def test_counters_start_zero(self):
        pm = PluginManager()
        assert pm.model_call_count == 0
        assert pm.tool_call_count == 0


class TestCounters:
    def test_increment_model_calls(self):
        pm = PluginManager()
        assert pm.increment_model_calls() == 1
        assert pm.increment_model_calls() == 2
        assert pm.model_call_count == 2

    def test_increment_tool_calls(self):
        pm = PluginManager()
        assert pm.increment_tool_calls() == 1
        assert pm.increment_tool_calls() == 2

    def test_reset_counters(self):
        pm = PluginManager()
        pm.increment_model_calls()
        pm.increment_tool_calls()
        pm.reset_counters()
        assert pm.model_call_count == 0
        assert pm.tool_call_count == 0


class TestBeforeAgent:
    @pytest.mark.asyncio
    async def test_runs_all_plugins(self):
        p1 = TrackingPlugin("t1")
        p2 = TrackingPlugin("t2")
        pm = PluginManager([p1, p2])
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await pm.run_before_agent(ctx)
        assert "before_agent" in p1.calls
        assert "before_agent" in p2.calls
        assert result is ctx

    @pytest.mark.asyncio
    async def test_halt_stops_pipeline(self):
        halter = HaltPlugin()
        tracker = TrackingPlugin()
        pm = PluginManager([halter, tracker])
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        with pytest.raises(PluginHalt) as exc_info:
            await pm.run_before_agent(ctx)
        assert exc_info.value.result == "agent_halted"
        assert "before_agent" not in tracker.calls

    @pytest.mark.asyncio
    async def test_context_modification(self):
        modifier = ModifyingPlugin()
        pm = PluginManager([modifier])
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await pm.run_before_agent(ctx)
        assert result.metadata["modified_by"] == "modifier"

    @pytest.mark.asyncio
    async def test_none_return_keeps_context(self):
        class ObserverPlugin(BasePlugin):
            async def before_agent(self, ctx):
                return None  # observe only

        pm = PluginManager([ObserverPlugin()])
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await pm.run_before_agent(ctx)
        assert result is ctx  # unchanged


class TestAfterAgent:
    @pytest.mark.asyncio
    async def test_result_passthrough(self):
        pm = PluginManager([TrackingPlugin()])
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await pm.run_after_agent(ctx, "original")
        assert result == "original"

    @pytest.mark.asyncio
    async def test_result_modification(self):
        pm = PluginManager([ModifyingPlugin()])
        ctx = AgentContext(agent_name="a", task="t", state="s", config={})
        result = await pm.run_after_agent(ctx, "hello")
        assert result == "modified: hello"


class TestBeforeModel:
    @pytest.mark.asyncio
    async def test_runs_all_plugins(self):
        p1 = TrackingPlugin("t1")
        p2 = TrackingPlugin("t2")
        pm = PluginManager([p1, p2])
        req = ModelRequest(agent_name="a")
        await pm.run_before_model(req)
        assert "before_model" in p1.calls
        assert "before_model" in p2.calls

    @pytest.mark.asyncio
    async def test_halt(self):
        pm = PluginManager([HaltPlugin()])
        req = ModelRequest(agent_name="a")
        with pytest.raises(PluginHalt) as exc_info:
            await pm.run_before_model(req)
        assert exc_info.value.result == "model_halted"

    @pytest.mark.asyncio
    async def test_request_modification_via_with(self):
        modifier = ModifyingPlugin()
        pm = PluginManager([modifier])
        req = ModelRequest(agent_name="a")
        result = await pm.run_before_model(req)
        assert result.metadata["model_modified"] is True


class TestAfterModel:
    @pytest.mark.asyncio
    async def test_response_modification(self):
        pm = PluginManager([ModifyingPlugin()])
        req = ModelRequest(agent_name="a")
        result = await pm.run_after_model(req, "raw_response")
        assert result == {"wrapped": "raw_response"}


class TestExecuteModelCall:
    @pytest.mark.asyncio
    async def test_no_plugins(self):
        pm = PluginManager()
        req = ModelRequest(
            agent_name="a",
            model="test",
            messages=[],
        )
        mock_llm = AsyncMock(return_value="llm_response")
        result = await pm.execute_model_call(req, mock_llm)
        assert result == "llm_response"
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_wrapper(self):
        tracker = TrackingPlugin()
        pm = PluginManager([tracker])
        req = ModelRequest(agent_name="a", model="test")
        mock_llm = AsyncMock(return_value="llm_response")
        result = await pm.execute_model_call(req, mock_llm)
        assert result == "llm_response"
        assert "wrap_model_call_enter" in tracker.calls
        assert "wrap_model_call_exit" in tracker.calls

    @pytest.mark.asyncio
    async def test_chain_order(self):
        """First plugin is outermost wrapper, last is closest to the LLM."""
        order = []

        class P(BasePlugin):
            def __init__(self, n):
                self._n = n

            @property
            def name(self):
                return self._n

            async def wrap_model_call(self, request, handler):
                order.append(f"{self._n}_enter")
                result = await handler(request)
                order.append(f"{self._n}_exit")
                return result

        pm = PluginManager([P("outer"), P("inner")])
        req = ModelRequest(agent_name="a", model="t")
        await pm.execute_model_call(req, AsyncMock(return_value="ok"))
        assert order == ["outer_enter", "inner_enter", "inner_exit", "outer_exit"]

    @pytest.mark.asyncio
    async def test_short_circuit(self):
        """Plugin can skip the actual LLM call by not calling handler."""

        class ShortCircuit(BasePlugin):
            async def wrap_model_call(self, request, handler):
                return "cached_result"

        mock_llm = AsyncMock()
        pm = PluginManager([ShortCircuit()])
        req = ModelRequest(agent_name="a", model="t")
        result = await pm.execute_model_call(req, mock_llm)
        assert result == "cached_result"
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_receives_modified_request(self):
        """Wrap plugin can modify the request via .with_() before calling handler."""
        captured_models = []

        class DowngradePlugin(BasePlugin):
            async def wrap_model_call(self, request, handler):
                return await handler(request.with_(model="gpt-4o-mini"))

        pm = PluginManager([DowngradePlugin()])
        req = ModelRequest(agent_name="a", model="gpt-4")

        async def mock_llm(**kwargs):
            captured_models.append(kwargs.get("model"))
            return "ok"

        await pm.execute_model_call(req, mock_llm)
        assert captured_models == ["gpt-4o-mini"]


class TestExecuteToolCall:
    @pytest.mark.asyncio
    async def test_no_plugins(self):
        pm = PluginManager()
        req = ToolRequest(
            agent_name="a",
            tool_name="calc",
            tool_args={"x": 1},
            tool_call_id="tc_1",
        )
        mock_exec = AsyncMock(return_value=42)
        result = await pm.execute_tool_call(req, mock_exec)
        assert result == 42

    @pytest.mark.asyncio
    async def test_wrapper_chain(self):
        tracker = TrackingPlugin()
        pm = PluginManager([tracker])
        req = ToolRequest(agent_name="a", tool_name="calc")
        mock_exec = AsyncMock(return_value=42)
        result = await pm.execute_tool_call(req, mock_exec)
        assert result == 42
        assert "wrap_tool_call_enter" in tracker.calls

    @pytest.mark.asyncio
    async def test_short_circuit_tool(self):
        class BlockTool(BasePlugin):
            async def wrap_tool_call(self, request, handler):
                return "blocked"

        pm = PluginManager([BlockTool()])
        req = ToolRequest(agent_name="a", tool_name="dangerous")
        mock_exec = AsyncMock()
        result = await pm.execute_tool_call(req, mock_exec)
        assert result == "blocked"
        mock_exec.assert_not_called()
