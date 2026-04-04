"""Tests for _GeminiNativeTool proxy mode — enable/disable/execute lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from nucleusiq_gemini.tools.gemini_tool import GeminiTool

from tests.unit._mock_factories import (
    make_candidate,
    make_code_execution_part,
    make_response,
    make_text_part,
)

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def _make_mock_llm(response=None):
    """Create a mock BaseGemini with a mock _client."""
    llm = MagicMock()
    llm.model = "gemini-2.5-flash"
    llm._client = MagicMock()
    if response is None:
        response = make_response()
    llm._client.generate_content = AsyncMock(return_value=response)
    return llm


# ------------------------------------------------------------------ #
# Proxy mode lifecycle                                                 #
# ------------------------------------------------------------------ #


class TestProxyModeLifecycle:
    def test_initial_state_is_native(self):
        tool = GeminiTool.google_search()
        assert tool.is_native is True
        assert tool.is_proxy_mode is False
        assert tool._proxy_llm is None
        assert tool._proxy_spec is None

    def test_enable_proxy_flips_is_native(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        assert tool.is_native is False
        assert tool.is_proxy_mode is True

    def test_enable_proxy_sets_llm_reference(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        assert tool._proxy_llm is llm

    def test_enable_proxy_sets_model(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        assert tool._proxy_model == "gemini-2.5-flash"

    def test_enable_proxy_creates_proxy_spec(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        assert tool._proxy_spec is not None
        assert tool._proxy_spec["name"] == "google_search"
        assert "parameters" in tool._proxy_spec

    def test_disable_proxy_reverts_to_native(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        assert tool.is_native is False

        tool._disable_proxy_mode()
        assert tool.is_native is True
        assert tool.is_proxy_mode is False
        assert tool._proxy_llm is None
        assert tool._proxy_spec is None

    def test_enable_proxy_idempotent(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        tool._enable_proxy_mode(llm)
        assert tool.is_native is False
        assert tool.is_proxy_mode is True


# ------------------------------------------------------------------ #
# get_spec() behaviour                                                 #
# ------------------------------------------------------------------ #


class TestGetSpec:
    def test_native_mode_returns_native_spec(self):
        tool = GeminiTool.google_search()
        spec = tool.get_spec()
        assert "type" in spec
        assert spec["type"] == "google_search"

    def test_proxy_mode_returns_function_declaration(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        spec = tool.get_spec()
        assert "type" not in spec
        assert spec["name"] == "google_search"
        assert "parameters" in spec
        assert "description" in spec

    def test_reverted_returns_native_spec(self):
        tool = GeminiTool.google_search()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        tool._disable_proxy_mode()
        spec = tool.get_spec()
        assert spec["type"] == "google_search"

    def test_code_execution_proxy_spec(self):
        tool = GeminiTool.code_execution()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        spec = tool.get_spec()
        assert spec["name"] == "code_execution"
        assert "task" in spec["parameters"]["properties"]

    def test_url_context_proxy_spec(self):
        tool = GeminiTool.url_context()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        spec = tool.get_spec()
        assert spec["name"] == "url_context"
        assert "url" in spec["parameters"]["properties"]

    def test_google_maps_proxy_spec(self):
        tool = GeminiTool.google_maps()
        llm = _make_mock_llm()
        tool._enable_proxy_mode(llm)
        spec = tool.get_spec()
        assert spec["name"] == "google_maps"
        assert "query" in spec["parameters"]["properties"]


# ------------------------------------------------------------------ #
# execute() — native mode                                              #
# ------------------------------------------------------------------ #


class TestExecuteNativeMode:
    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        tool = GeminiTool.google_search()
        with pytest.raises(NotImplementedError, match="server-side"):
            await tool.execute(query="test")


# ------------------------------------------------------------------ #
# execute() — proxy mode                                               #
# ------------------------------------------------------------------ #


class TestExecuteProxyMode:
    @pytest.mark.asyncio
    async def test_calls_generate_content(self):
        response = make_response()
        llm = _make_mock_llm(response)
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        await tool.execute(query="AAPL stock price")

        llm._client.generate_content.assert_called_once()
        call_kwargs = llm._client.generate_content.call_args
        assert "AAPL stock price" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_passes_native_tool_in_config(self):
        llm = _make_mock_llm()
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        await tool.execute(query="test")

        call_kwargs = llm._client.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config", {})
        assert {"google_search": {}} in config["tools"]

    @pytest.mark.asyncio
    async def test_returns_text_content(self):
        response = make_response(
            candidates=[make_candidate([make_text_part("AAPL is at $189.50")])]
        )
        llm = _make_mock_llm(response)
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        result = await tool.execute(query="AAPL price")
        assert "189.50" in result

    @pytest.mark.asyncio
    async def test_handles_code_execution_result(self):
        response = make_response(
            candidates=[
                make_candidate(
                    [
                        make_text_part("Running code..."),
                        make_code_execution_part("print(42)", "42"),
                    ]
                )
            ]
        )
        llm = _make_mock_llm(response)
        tool = GeminiTool.code_execution()
        tool._enable_proxy_mode(llm)

        result = await tool.execute(task="compute 42")
        assert "42" in result
        assert "Code output" in result

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        response = make_response(candidates=[])
        llm = _make_mock_llm(response)
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        result = await tool.execute(query="test")
        assert "no results" in result.lower()

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        llm = _make_mock_llm()
        llm._client.generate_content = AsyncMock(
            side_effect=RuntimeError("API quota exceeded")
        )
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        result = await tool.execute(query="test")
        assert "Error" in result
        assert "quota" in result.lower()

    @pytest.mark.asyncio
    async def test_uses_correct_model(self):
        llm = _make_mock_llm()
        llm.model = "gemini-2.5-pro"
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        await tool.execute(query="test")

        call_args = llm._client.generate_content.call_args
        assert call_args.kwargs.get("model") == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_fallback_query_from_kwargs_values(self):
        """When none of query/task/url match, concatenate all values."""
        llm = _make_mock_llm()
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        await tool.execute(custom_param="hello world")

        call_args = llm._client.generate_content.call_args
        contents = call_args.kwargs.get("contents", [])
        assert "hello world" in str(contents)

    @pytest.mark.asyncio
    async def test_no_input_returns_error(self):
        llm = _make_mock_llm()
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(llm)

        result = await tool.execute()
        assert "No input" in result


# ------------------------------------------------------------------ #
# Executor integration (simulated)                                     #
# ------------------------------------------------------------------ #


class TestExecutorIntegration:
    """Verify that the proxy pattern works with how the core Executor operates."""

    def test_executor_would_reject_native_tool(self):
        """In native mode, is_native=True which makes Executor skip execute()."""
        tool = GeminiTool.google_search()
        assert getattr(tool, "is_native", False) is True

    def test_executor_would_accept_proxy_tool(self):
        """In proxy mode, is_native=False so Executor calls execute()."""
        tool = GeminiTool.google_search()
        tool._enable_proxy_mode(_make_mock_llm())
        assert getattr(tool, "is_native", False) is False

    def test_object_identity_preserved(self):
        """Executor holds a reference; mutations on the same object are visible."""
        tool = GeminiTool.google_search()
        registry = {tool.name: tool}

        assert registry["google_search"].is_native is True

        tool._enable_proxy_mode(_make_mock_llm())

        assert registry["google_search"].is_native is False
        assert registry["google_search"] is tool
