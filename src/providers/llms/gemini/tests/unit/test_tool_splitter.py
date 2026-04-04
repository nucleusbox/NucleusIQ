"""Tests for nucleusiq_gemini.tools.tool_splitter — mixed tool detection and proxy specs."""

from nucleusiq_gemini.tools.gemini_tool import GeminiTool
from nucleusiq_gemini.tools.tool_splitter import (
    PROXY_DESCRIPTIONS,
    build_proxy_spec,
    classify_tools,
    has_mixed_tools,
)

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


class _FakeCustomTool:
    """Minimal stand-in for a custom @tool function."""

    def __init__(self, name: str = "calculator"):
        self.name = name
        self.is_native = False

    def get_spec(self):
        return {
            "name": self.name,
            "description": f"A {self.name} tool",
            "parameters": {"type": "object", "properties": {}},
        }


# ------------------------------------------------------------------ #
# has_mixed_tools                                                      #
# ------------------------------------------------------------------ #


class TestHasMixedTools:
    def test_native_and_custom_returns_true(self):
        tools = [GeminiTool.google_search(), _FakeCustomTool()]
        assert has_mixed_tools(tools) is True

    def test_multiple_native_and_custom(self):
        tools = [
            GeminiTool.google_search(),
            GeminiTool.code_execution(),
            _FakeCustomTool("calc"),
            _FakeCustomTool("db"),
        ]
        assert has_mixed_tools(tools) is True

    def test_only_native_returns_false(self):
        tools = [GeminiTool.google_search(), GeminiTool.code_execution()]
        assert has_mixed_tools(tools) is False

    def test_only_custom_returns_false(self):
        tools = [_FakeCustomTool("a"), _FakeCustomTool("b")]
        assert has_mixed_tools(tools) is False

    def test_empty_list_returns_false(self):
        assert has_mixed_tools([]) is False

    def test_single_native_returns_false(self):
        assert has_mixed_tools([GeminiTool.google_search()]) is False

    def test_single_custom_returns_false(self):
        assert has_mixed_tools([_FakeCustomTool()]) is False


# ------------------------------------------------------------------ #
# classify_tools                                                       #
# ------------------------------------------------------------------ #


class TestClassifyTools:
    def test_split_mixed(self):
        search = GeminiTool.google_search()
        calc = _FakeCustomTool("calc")
        native, custom = classify_tools([search, calc])
        assert native == [search]
        assert custom == [calc]

    def test_split_multiple(self):
        s = GeminiTool.google_search()
        c = GeminiTool.code_execution()
        t1 = _FakeCustomTool("a")
        t2 = _FakeCustomTool("b")
        native, custom = classify_tools([s, t1, c, t2])
        assert native == [s, c]
        assert custom == [t1, t2]

    def test_all_native(self):
        tools = [GeminiTool.google_search(), GeminiTool.google_maps()]
        native, custom = classify_tools(tools)
        assert len(native) == 2
        assert len(custom) == 0

    def test_all_custom(self):
        tools = [_FakeCustomTool("x"), _FakeCustomTool("y")]
        native, custom = classify_tools(tools)
        assert len(native) == 0
        assert len(custom) == 2

    def test_empty(self):
        native, custom = classify_tools([])
        assert native == []
        assert custom == []


# ------------------------------------------------------------------ #
# build_proxy_spec                                                     #
# ------------------------------------------------------------------ #


class TestBuildProxySpec:
    def test_google_search_spec(self):
        spec = build_proxy_spec("google_search", "google_search")
        assert spec["name"] == "google_search"
        assert "search" in spec["description"].lower()
        assert spec["parameters"]["type"] == "object"
        assert "query" in spec["parameters"]["properties"]

    def test_code_execution_spec(self):
        spec = build_proxy_spec("code_execution", "code_execution")
        assert spec["name"] == "code_execution"
        assert "task" in spec["parameters"]["properties"]

    def test_url_context_spec(self):
        spec = build_proxy_spec("url_context", "url_context")
        assert spec["name"] == "url_context"
        assert "url" in spec["parameters"]["properties"]

    def test_google_maps_spec(self):
        spec = build_proxy_spec("google_maps", "google_maps")
        assert spec["name"] == "google_maps"
        assert "query" in spec["parameters"]["properties"]

    def test_unknown_type_fallback(self):
        spec = build_proxy_spec("future_tool", "future_tool")
        assert spec["name"] == "future_tool"
        assert "future_tool" in spec["description"]
        assert "query" in spec["parameters"]["properties"]

    def test_custom_name(self):
        spec = build_proxy_spec("google_search", "my_search")
        assert spec["name"] == "my_search"

    def test_all_known_types_have_descriptions(self):
        for tool_type in PROXY_DESCRIPTIONS:
            spec = build_proxy_spec(tool_type, tool_type)
            assert spec["description"]
            assert spec["parameters"]["required"]


# ------------------------------------------------------------------ #
# PROXY_DESCRIPTIONS coverage                                          #
# ------------------------------------------------------------------ #


class TestProxyDescriptions:
    def test_all_four_types_present(self):
        expected = {"google_search", "code_execution", "url_context", "google_maps"}
        assert set(PROXY_DESCRIPTIONS.keys()) == expected

    def test_each_has_description_and_parameters(self):
        for tool_type, meta in PROXY_DESCRIPTIONS.items():
            assert "description" in meta, f"{tool_type} missing description"
            assert "parameters" in meta, f"{tool_type} missing parameters"
            assert meta["parameters"]["type"] == "object"
