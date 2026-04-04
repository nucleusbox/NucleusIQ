"""Tests for BaseGemini.convert_tool_specs() — mixed tool proxy integration.

Validates that the override in BaseGemini correctly:
1. Detects mixed tools and enables proxy mode
2. Keeps native-only and custom-only lists unchanged
3. Reverts proxy mode when mixing is removed
4. Produces specs that are compatible with Gemini's API format
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from nucleusiq.tools.base_tool import BaseTool
from nucleusiq_gemini.tools.gemini_tool import GeminiTool

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


class _FakeCustomTool(BaseTool):
    """Minimal custom tool for testing."""

    def __init__(self, name: str = "calculator"):
        super().__init__(name=name, description=f"A {name} tool", version=None)
        self.is_native = False

    async def initialize(self) -> None:
        pass

    async def execute(self, **kwargs) -> str:
        return "42"

    def get_spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression",
                    }
                },
                "required": ["expression"],
            },
        }


def _make_base_gemini():
    """Create a BaseGemini instance with mocked client for unit tests."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("nucleusiq_gemini.nb_gemini.base.GeminiClient") as MockClient:
            MockClient.return_value = MagicMock()
            MockClient.return_value.generate_content = AsyncMock()
            from nucleusiq_gemini.nb_gemini.base import BaseGemini

            llm = BaseGemini(model_name="gemini-2.5-flash")
            return llm


# ------------------------------------------------------------------ #
# convert_tool_specs with mixed tools                                  #
# ------------------------------------------------------------------ #


class TestConvertToolSpecsMixed:
    """When both native and custom tools are passed."""

    def test_native_tools_get_proxy_mode_enabled(self):
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calculator")

        llm.convert_tool_specs([search, calc])

        assert search.is_native is False
        assert search.is_proxy_mode is True

    def test_proxy_spec_is_function_declaration(self):
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calculator")

        specs = llm.convert_tool_specs([search, calc])

        search_spec = next(s for s in specs if s.get("name") == "google_search")
        assert "type" not in search_spec
        assert "parameters" in search_spec
        assert "description" in search_spec

    def test_custom_tools_unchanged(self):
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calculator")

        specs = llm.convert_tool_specs([search, calc])

        calc_spec = next(s for s in specs if s.get("name") == "calculator")
        assert calc_spec["name"] == "calculator"
        assert "expression" in calc_spec["parameters"]["properties"]

    def test_multiple_native_tools_all_proxied(self):
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        code = GeminiTool.code_execution()
        calc = _FakeCustomTool("calc")

        llm.convert_tool_specs([search, code, calc])

        assert search.is_proxy_mode is True
        assert code.is_proxy_mode is True
        assert not getattr(calc, "is_proxy_mode", False)

    def test_all_specs_are_function_declarations(self):
        """After proxy mode, every spec should be a function declaration (no native type key)."""
        llm = _make_base_gemini()
        tools = [
            GeminiTool.google_search(),
            GeminiTool.code_execution(),
            _FakeCustomTool("calc"),
            _FakeCustomTool("db_query"),
        ]

        specs = llm.convert_tool_specs(tools)

        for spec in specs:
            assert "name" in spec
            assert "type" not in spec, (
                f"Spec for '{spec.get('name')}' still has native 'type' key"
            )


# ------------------------------------------------------------------ #
# convert_tool_specs with only native or only custom                   #
# ------------------------------------------------------------------ #


class TestConvertToolSpecsHomogeneous:
    """When all tools are the same type, no proxy mode needed."""

    def test_only_native_stays_native(self):
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        code = GeminiTool.code_execution()

        specs = llm.convert_tool_specs([search, code])

        assert search.is_native is True
        assert code.is_native is True
        assert any("type" in s for s in specs)

    def test_only_custom_stays_custom(self):
        llm = _make_base_gemini()
        calc = _FakeCustomTool("calc")
        db = _FakeCustomTool("db")

        specs = llm.convert_tool_specs([calc, db])

        assert len(specs) == 2
        for spec in specs:
            assert "type" not in spec
            assert "name" in spec


# ------------------------------------------------------------------ #
# Proxy mode revert                                                    #
# ------------------------------------------------------------------ #


class TestProxyModeRevert:
    """When tools change from mixed to homogeneous, proxy should be reverted."""

    def test_revert_when_custom_removed(self):
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calc")

        llm.convert_tool_specs([search, calc])
        assert search.is_proxy_mode is True

        llm.convert_tool_specs([search])
        assert search.is_proxy_mode is False
        assert search.is_native is True


# ------------------------------------------------------------------ #
# Spec format compatibility                                            #
# ------------------------------------------------------------------ #


class TestSpecFormatCompatibility:
    """Converted specs should be valid for Gemini's tool_converter."""

    def test_proxy_specs_pass_through_tool_converter(self):
        from nucleusiq_gemini.tools.tool_converter import (
            build_tools_payload,
            convert_tool_spec,
        )

        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calc")

        specs = llm.convert_tool_specs([search, calc])
        declarations = [convert_tool_spec(s) for s in specs]
        payload = build_tools_payload(declarations)

        assert len(payload) == 1
        assert "function_declarations" in payload[0]
        fn_decls = payload[0]["function_declarations"]
        names = {d["name"] for d in fn_decls}
        assert "google_search" in names
        assert "calc" in names

    def test_native_only_preserves_native_format(self):
        from nucleusiq_gemini.tools.tool_converter import (
            build_tools_payload,
            convert_tool_spec,
        )

        llm = _make_base_gemini()
        search = GeminiTool.google_search()

        specs = llm.convert_tool_specs([search])
        declarations = [convert_tool_spec(s) for s in specs]
        payload = build_tools_payload(declarations)

        native_payloads = [p for p in payload if "function_declarations" not in p]
        assert len(native_payloads) == 1
        assert native_payloads[0]["type"] == "google_search"


# ------------------------------------------------------------------ #
# Regression: all execution modes use the same path                    #
# ------------------------------------------------------------------ #


class TestAllModesSharePath:
    """All execution modes call agent.llm.convert_tool_specs().

    These tests validate the invariant at the convert_tool_specs level
    (not full mode tests — those live in core).
    """

    def test_convert_tool_specs_is_deterministic(self):
        """Calling twice with the same input produces the same specs."""
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calc")

        specs1 = llm.convert_tool_specs([search, calc])
        specs2 = llm.convert_tool_specs([search, calc])

        assert specs1 == specs2
        assert search.is_proxy_mode is True

    def test_object_identity_preserved_across_calls(self):
        """The same tool objects are returned — Executor holds references."""
        llm = _make_base_gemini()
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calc")

        tool_list = [search, calc]
        llm.convert_tool_specs(tool_list)

        assert tool_list[0] is search
        assert search.is_proxy_mode is True
