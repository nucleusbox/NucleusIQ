"""Tests for nucleusiq_gemini.tools.gemini_tool — native tool factory."""

import pytest
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq_gemini.tools.gemini_tool import (
    NATIVE_TOOL_TYPES,
    GeminiTool,
    _GeminiNativeTool,
)


class TestNativeToolTypes:
    def test_google_search_registered(self):
        assert "google_search" in NATIVE_TOOL_TYPES

    def test_code_execution_registered(self):
        assert "code_execution" in NATIVE_TOOL_TYPES

    def test_url_context_registered(self):
        assert "url_context" in NATIVE_TOOL_TYPES

    def test_google_maps_registered(self):
        assert "google_maps" in NATIVE_TOOL_TYPES

    def test_frozenset(self):
        assert isinstance(NATIVE_TOOL_TYPES, frozenset)

    def test_total_count(self):
        assert len(NATIVE_TOOL_TYPES) == 4


class TestGeminiToolFactory:
    def test_google_search_creates_tool(self):
        tool = GeminiTool.google_search()
        assert isinstance(tool, BaseTool)
        assert tool.name == "google_search"

    def test_google_search_spec(self):
        tool = GeminiTool.google_search()
        spec = tool.get_spec()
        assert spec["type"] == "google_search"
        assert "google_search" in spec

    def test_google_search_with_config(self):
        tool = GeminiTool.google_search(
            dynamic_retrieval_config={"mode": "MODE_DYNAMIC", "dynamic_threshold": 0.5}
        )
        spec = tool.get_spec()
        assert "dynamic_retrieval_config" in spec["google_search"]

    def test_code_execution_creates_tool(self):
        tool = GeminiTool.code_execution()
        assert isinstance(tool, BaseTool)
        assert tool.name == "code_execution"

    def test_code_execution_spec(self):
        tool = GeminiTool.code_execution()
        spec = tool.get_spec()
        assert spec["type"] == "code_execution"

    def test_url_context_creates_tool(self):
        tool = GeminiTool.url_context()
        assert isinstance(tool, BaseTool)
        assert tool.name == "url_context"

    def test_url_context_spec(self):
        tool = GeminiTool.url_context()
        spec = tool.get_spec()
        assert spec["type"] == "url_context"

    def test_google_maps_creates_tool(self):
        tool = GeminiTool.google_maps()
        assert isinstance(tool, BaseTool)
        assert tool.name == "google_maps"

    def test_google_maps_spec(self):
        tool = GeminiTool.google_maps()
        spec = tool.get_spec()
        assert spec["type"] == "google_maps"
        assert "google_maps" in spec

    def test_google_maps_is_native(self):
        tool = GeminiTool.google_maps()
        assert tool.is_native is True


class TestGeminiNativeTool:
    def test_is_native_flag(self):
        tool = GeminiTool.google_search()
        assert tool.is_native is True

    @pytest.mark.asyncio
    async def test_execute_raises(self):
        tool = GeminiTool.google_search()
        with pytest.raises(NotImplementedError):
            await tool.execute()

    @pytest.mark.asyncio
    async def test_initialize_noop(self):
        tool = GeminiTool.google_search()
        await tool.initialize()

    def test_tool_type_attribute(self):
        tool = GeminiTool.code_execution()
        assert isinstance(tool, _GeminiNativeTool)
        assert tool.tool_type == "code_execution"


class TestTypeConstants:
    def test_google_search_type(self):
        assert GeminiTool.GOOGLE_SEARCH_TYPE == "google_search"

    def test_code_execution_type(self):
        assert GeminiTool.CODE_EXECUTION_TYPE == "code_execution"

    def test_url_context_type(self):
        assert GeminiTool.URL_CONTEXT_TYPE == "url_context"

    def test_google_maps_type(self):
        assert GeminiTool.GOOGLE_MAPS_TYPE == "google_maps"
