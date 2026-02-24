"""Unit tests for built-in plugins."""

from unittest.mock import AsyncMock

import pytest
from nucleusiq.plugins.base import ModelRequest, ToolRequest
from nucleusiq.plugins.builtin.model_call_limit import ModelCallLimitPlugin
from nucleusiq.plugins.builtin.tool_call_limit import ToolCallLimitPlugin
from nucleusiq.plugins.builtin.tool_retry import ToolRetryPlugin
from nucleusiq.plugins.errors import PluginHalt

# ------------------------------------------------------------------ #
# ModelCallLimitPlugin                                                 #
# ------------------------------------------------------------------ #


class TestModelCallLimitPlugin:
    def test_name(self):
        p = ModelCallLimitPlugin()
        assert p.name == "model_call_limit"

    @pytest.mark.asyncio
    async def test_under_limit_returns_none(self):
        p = ModelCallLimitPlugin(max_calls=10)
        req = ModelRequest(agent_name="a", call_count=5)
        result = await p.before_model(req)
        assert result is None  # no change

    @pytest.mark.asyncio
    async def test_at_limit_returns_none(self):
        p = ModelCallLimitPlugin(max_calls=10)
        req = ModelRequest(agent_name="a", call_count=10)
        result = await p.before_model(req)
        assert result is None  # at limit is ok

    @pytest.mark.asyncio
    async def test_over_limit_halts(self):
        p = ModelCallLimitPlugin(max_calls=5)
        req = ModelRequest(agent_name="a", call_count=6)
        with pytest.raises(PluginHalt) as exc_info:
            await p.before_model(req)
        assert "limit exceeded" in str(exc_info.value.result).lower()

    @pytest.mark.asyncio
    async def test_custom_limit(self):
        p = ModelCallLimitPlugin(max_calls=1)
        req1 = ModelRequest(agent_name="a", call_count=1)
        result = await p.before_model(req1)
        assert result is None

        req2 = ModelRequest(agent_name="a", call_count=2)
        with pytest.raises(PluginHalt):
            await p.before_model(req2)


# ------------------------------------------------------------------ #
# ToolCallLimitPlugin                                                  #
# ------------------------------------------------------------------ #


class TestToolCallLimitPlugin:
    def test_name(self):
        p = ToolCallLimitPlugin()
        assert p.name == "tool_call_limit"

    @pytest.mark.asyncio
    async def test_under_limit(self):
        p = ToolCallLimitPlugin(max_calls=10)
        req = ToolRequest(agent_name="a", tool_name="t", call_count=5)
        handler = AsyncMock(return_value="ok")
        result = await p.wrap_tool_call(req, handler)
        assert result == "ok"
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_over_limit_halts(self):
        p = ToolCallLimitPlugin(max_calls=3)
        req = ToolRequest(agent_name="a", tool_name="t", call_count=4)
        handler = AsyncMock()
        with pytest.raises(PluginHalt) as exc_info:
            await p.wrap_tool_call(req, handler)
        assert "limit exceeded" in str(exc_info.value.result).lower()
        handler.assert_not_called()


# ------------------------------------------------------------------ #
# ToolRetryPlugin                                                      #
# ------------------------------------------------------------------ #


class TestToolRetryPlugin:
    def test_name(self):
        p = ToolRetryPlugin()
        assert p.name == "tool_retry"

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        p = ToolRetryPlugin(max_retries=3, base_delay=0.01)
        req = ToolRequest(agent_name="a", tool_name="t")
        handler = AsyncMock(return_value="ok")
        result = await p.wrap_tool_call(req, handler)
        assert result == "ok"
        assert handler.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        p = ToolRetryPlugin(max_retries=2, base_delay=0.01)
        req = ToolRequest(agent_name="a", tool_name="t")
        handler = AsyncMock(side_effect=[ValueError("fail"), ValueError("fail"), "ok"])
        result = await p.wrap_tool_call(req, handler)
        assert result == "ok"
        assert handler.call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        p = ToolRetryPlugin(max_retries=1, base_delay=0.01)
        req = ToolRequest(agent_name="a", tool_name="t")
        handler = AsyncMock(side_effect=RuntimeError("permanent"))
        with pytest.raises(RuntimeError, match="permanent"):
            await p.wrap_tool_call(req, handler)
        assert handler.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_retries_raises_immediately(self):
        p = ToolRetryPlugin(max_retries=0, base_delay=0.01)
        req = ToolRequest(agent_name="a", tool_name="t")
        handler = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError, match="fail"):
            await p.wrap_tool_call(req, handler)
        assert handler.call_count == 1
