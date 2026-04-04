"""Integration tests for mixed native + custom tool scenarios.

These tests require ``GEMINI_API_KEY`` to be set in the environment.
They validate that the proxy pattern works end-to-end with real API calls,
from low-level proxy mode through full Agent.execute() with mixed tools
across Standard and Direct execution modes.

Evidence tests (``TestMultiToolEvidence*``) use 2 native + 2 custom tools
with print-statement instrumentation to prove each tool is actually invoked.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.agents.task import Task
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq_gemini import BaseGemini
from nucleusiq_gemini.tools.gemini_tool import GeminiTool

_HAS_GEMINI_KEY = bool(os.getenv("GEMINI_API_KEY"))
pytestmark = pytest.mark.skipif(not _HAS_GEMINI_KEY, reason="GEMINI_API_KEY not set")


# ------------------------------------------------------------------ #
# Custom tool fixtures                                                 #
# ------------------------------------------------------------------ #


class CalculatorTool(BaseTool):
    """Simple calculator for integration tests."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform arithmetic calculations. Returns the numeric result.",
            version=None,
        )
        self.is_native = False

    async def initialize(self) -> None:
        pass

    async def execute(self, expression: str = "", **kwargs: Any) -> str:
        try:
            result = eval(expression)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python arithmetic expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
        }


class UnitConverterTool(BaseTool):
    """Unit converter with call instrumentation for evidence tests."""

    def __init__(self):
        super().__init__(
            name="unit_converter",
            description=(
                "Convert between units. Supports: km<->miles, kg<->pounds, "
                "celsius<->fahrenheit, liters<->gallons. "
                "Provide value, from_unit, to_unit."
            ),
            version=None,
        )
        self.call_count = 0
        self.call_log: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        pass

    async def execute(
        self, value: float = 0, from_unit: str = "", to_unit: str = "", **kwargs: Any
    ) -> str:
        self.call_count += 1
        t0 = time.perf_counter()
        print(f"\n>>> [UnitConverterTool] CALLED #{self.call_count}")
        print(f"    value={value}, from_unit='{from_unit}', to_unit='{to_unit}'")

        value = float(value)
        conversions = {
            ("km", "miles"): lambda v: v * 0.621371,
            ("miles", "km"): lambda v: v * 1.60934,
            ("kg", "pounds"): lambda v: v * 2.20462,
            ("pounds", "kg"): lambda v: v * 0.453592,
            ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
            ("liters", "gallons"): lambda v: v * 0.264172,
            ("gallons", "liters"): lambda v: v * 3.78541,
        }

        key = (from_unit.lower().strip(), to_unit.lower().strip())
        if key in conversions:
            result = conversions[key](value)
            output = f"{value} {from_unit} = {result:.4f} {to_unit}"
        else:
            output = f"Unsupported conversion: {from_unit} -> {to_unit}"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.call_log.append(
            {"value": value, "from": from_unit, "to": to_unit, "ms": elapsed_ms}
        )
        print(f"    Result: {output}")
        print(f"    Execution time: {elapsed_ms:.3f} ms")
        print("<<< [UnitConverterTool] DONE\n")
        return output

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        }


class NoteTakerTool(BaseTool):
    """Note-taking tool with call instrumentation for evidence tests."""

    def __init__(self):
        super().__init__(
            name="note_taker",
            description=(
                "Save a note with title and content. "
                "action='save' with title+content to store. "
                "action='list' to retrieve all notes."
            ),
            version=None,
        )
        self.notes: list[dict[str, str]] = []
        self.call_count = 0
        self.call_log: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        pass

    async def execute(
        self, action: str = "save", title: str = "", content: str = "", **kwargs: Any
    ) -> str:
        self.call_count += 1
        t0 = time.perf_counter()
        print(f"\n>>> [NoteTakerTool] CALLED #{self.call_count}")
        print(f"    action='{action}', title='{title}'")

        if action == "save" and title:
            self.notes.append({"title": title, "content": content})
            output = (
                f"Note saved: '{title}' ({len(content)} chars). "
                f"Total: {len(self.notes)}"
            )
        elif action == "list":
            if not self.notes:
                output = "No notes saved."
            else:
                lines = [f"  {i + 1}. {n['title']}" for i, n in enumerate(self.notes)]
                output = f"Notes ({len(self.notes)}):\n" + "\n".join(lines)
        else:
            output = f"Unknown action '{action}' or missing title."

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.call_log.append({"action": action, "title": title, "ms": elapsed_ms})
        print(f"    Result: {output}")
        print(f"    Execution time: {elapsed_ms:.3f} ms")
        print("<<< [NoteTakerTool] DONE\n")
        return output

    def get_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["save", "list"]},
                    "title": {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Note content"},
                },
                "required": ["action"],
            },
        }


# ------------------------------------------------------------------ #
# Proxy mode lifecycle with real LLM                                   #
# ------------------------------------------------------------------ #


class TestMixedToolProxySetup:
    """Verify proxy mode setup with a real BaseGemini instance."""

    def test_convert_tool_specs_enables_proxy(self, gemini_llm):
        search = GeminiTool.google_search()
        calc = CalculatorTool()

        specs = gemini_llm.convert_tool_specs([search, calc])

        assert search.is_proxy_mode is True
        assert search.is_native is False
        assert len(specs) == 2
        for spec in specs:
            assert "name" in spec
            assert "type" not in spec

    def test_three_native_one_custom(self, gemini_llm):
        search = GeminiTool.google_search()
        code = GeminiTool.code_execution()
        maps = GeminiTool.google_maps()
        calc = CalculatorTool()

        specs = gemini_llm.convert_tool_specs([search, code, maps, calc])

        assert all(t.is_proxy_mode for t in [search, code, maps])
        assert len(specs) == 4

    def test_native_only_no_proxy(self, gemini_llm):
        search = GeminiTool.google_search()
        code = GeminiTool.code_execution()

        gemini_llm.convert_tool_specs([search, code])

        assert search.is_native is True
        assert code.is_native is True


# ------------------------------------------------------------------ #
# Proxy execution with real API (google_search)                        #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
class TestMixedToolProxyExecution:
    """Execute native tools through proxy mode with real API calls."""

    async def test_google_search_proxy(self, gemini_llm):
        """google_search should return real web results via proxy."""
        search = GeminiTool.google_search()
        calc = CalculatorTool()

        gemini_llm.convert_tool_specs([search, calc])
        assert search.is_proxy_mode is True

        result = await search.execute(query="Python 3.13 release date")
        assert isinstance(result, str)
        assert len(result) > 10
        assert result != "(no results)"


# ------------------------------------------------------------------ #
# Spec compatibility regression tests                                  #
# ------------------------------------------------------------------ #


class TestSpecCompatibilityRegression:
    """The converted specs must flow through the existing tool pipeline."""

    def test_proxy_specs_work_with_build_tools_payload(self, gemini_llm):
        from nucleusiq_gemini.tools.tool_converter import (
            build_tools_payload,
            convert_tool_spec,
        )

        search = GeminiTool.google_search()
        code = GeminiTool.code_execution()
        calc = CalculatorTool()

        specs = gemini_llm.convert_tool_specs([search, code, calc])
        declarations = [convert_tool_spec(s) for s in specs]
        payload = build_tools_payload(declarations)

        assert len(payload) == 1
        assert "function_declarations" in payload[0]
        fn_names = {d["name"] for d in payload[0]["function_declarations"]}
        assert "google_search" in fn_names
        assert "code_execution" in fn_names
        assert "calculator" in fn_names


# ------------------------------------------------------------------ #
# Full Agent.execute() — mixed tools, Standard mode                    #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
class TestMixedToolsAgentStandardMode:
    """End-to-end agent execution with google_search + custom tool
    in Standard mode.  Proves the proxy pattern works through the
    full Agent -> StandardMode -> Executor -> _proxy_execute pipeline."""

    async def test_search_and_calculate(self):
        """Agent uses google_search (proxy) then calculator (custom)."""
        search = GeminiTool.google_search()
        calc = CalculatorTool()

        agent = Agent(
            name="IntegrationStandard",
            role="Research assistant with calculation abilities",
            objective="Answer questions using web search and calculations",
            narrative="Use google_search for live data, calculator for math.",
            llm=BaseGemini(model_name="gemini-2.5-flash", temperature=0.0),
            tools=[search, calc],
            config=AgentConfig(
                execution_mode="standard",
                verbose=False,
                enable_tracing=True,
                max_tool_calls=10,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            Task(
                id="int-std-1",
                objective=(
                    "What is the population of India according to the latest data? "
                    "Then calculate what 15% of that population would be."
                ),
            )
        )

        assert result.status.value == "success"
        assert len(result.tool_calls) >= 1
        tool_names = [tc.tool_name for tc in result.tool_calls]
        assert "google_search" in tool_names, (
            f"Expected google_search in tool calls, got: {tool_names}"
        )
        assert all(tc.success for tc in result.tool_calls)
        assert len(result.llm_calls) >= 2
        assert result.output is not None
        assert len(str(result.output)) > 20

    async def test_search_only_task(self):
        """Agent should use google_search proxy for a search-only task."""
        search = GeminiTool.google_search()
        calc = CalculatorTool()

        agent = Agent(
            name="IntegrationSearchOnly",
            role="Web researcher",
            objective="Find information on the web",
            narrative="Use google_search to find current information.",
            llm=BaseGemini(model_name="gemini-2.5-flash", temperature=0.0),
            tools=[search, calc],
            config=AgentConfig(
                execution_mode="standard",
                verbose=False,
                enable_tracing=True,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            Task(
                id="int-std-2",
                objective="What is the latest version of Python released in 2025?",
            )
        )

        assert result.status.value == "success"
        assert len(result.tool_calls) >= 1
        assert any(tc.tool_name == "google_search" for tc in result.tool_calls)
        assert result.output is not None

    async def test_tracing_captures_proxy_tool_calls(self):
        """Tracing should record proxy tool calls with correct metadata."""
        search = GeminiTool.google_search()
        calc = CalculatorTool()

        agent = Agent(
            name="IntegrationTracing",
            role="Assistant",
            objective="Answer questions",
            narrative="Use tools as needed.",
            llm=BaseGemini(model_name="gemini-2.5-flash", temperature=0.0),
            tools=[search, calc],
            config=AgentConfig(
                execution_mode="standard",
                verbose=False,
                enable_tracing=True,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            Task(id="int-std-3", objective="Search the web for NucleusIQ framework")
        )

        assert result.status.value == "success"
        for tc in result.tool_calls:
            assert tc.duration_ms >= 0
            assert tc.tool_name in ("google_search", "calculator")
        for lc in result.llm_calls:
            assert lc.duration_ms >= 0
            assert lc.model is not None


# ------------------------------------------------------------------ #
# Full Agent.execute() — mixed tools, Direct mode                      #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
class TestMixedToolsAgentDirectMode:
    """End-to-end agent execution with mixed tools in Direct mode."""

    async def test_direct_mode_uses_proxy_search(self):
        """Direct mode with mixed tools should activate proxy and call tools."""
        search = GeminiTool.google_search()
        calc = CalculatorTool()

        agent = Agent(
            name="IntegrationDirect",
            role="Research assistant",
            objective="Answer questions using search and calculator",
            narrative="Use google_search for live data.",
            llm=BaseGemini(model_name="gemini-2.5-flash", temperature=0.0),
            tools=[search, calc],
            config=AgentConfig(
                execution_mode="direct",
                verbose=False,
                enable_tracing=True,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            Task(
                id="int-dir-1",
                objective="Search the web for the current price of gold per ounce.",
            )
        )

        assert result.status.value == "success"
        assert result.output is not None
        if result.tool_calls:
            assert any(tc.tool_name == "google_search" for tc in result.tool_calls)
            assert all(tc.success for tc in result.tool_calls)


# ------------------------------------------------------------------ #
# Proxy mode lifecycle — verbose proof                                  #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
class TestProxyModeLifecycleProof:
    """Detailed step-by-step proof that proxy mode activates, executes,
    and produces real results — designed to be run with ``-s`` for
    human-readable output."""

    async def test_proxy_activation_and_execution(self, gemini_llm):
        """Step-by-step proof: activation -> spec conversion -> execution."""
        search = GeminiTool.google_search()
        calc = CalculatorTool()

        # Before: native tool
        assert search.is_native is True
        assert search.is_proxy_mode is False

        # Activate proxy mode via convert_tool_specs
        specs = gemini_llm.convert_tool_specs([search, calc])

        # After: proxy mode active
        assert search.is_native is False
        assert search.is_proxy_mode is True

        # Specs are all function declarations (no native tool specs)
        assert len(specs) == 2
        spec_names = {s["name"] for s in specs}
        assert spec_names == {"google_search", "calculator"}
        for spec in specs:
            assert "description" in spec
            assert "parameters" in spec

        # Execute google_search through proxy — real API call
        result = await search.execute(query="NucleusIQ AI framework")
        assert isinstance(result, str)
        assert len(result) > 50
        assert result != "(no results)"

        # Disable proxy mode
        search._disable_proxy_mode()
        assert search.is_native is True
        assert search.is_proxy_mode is False


# ------------------------------------------------------------------ #
# Evidence tests: 2 native + 2 custom tools                           #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
class TestMultiToolEvidenceFlash:
    """Comprehensive evidence: 2 native + 2 custom tools on gemini-2.5-flash.

    Run with ``pytest -s`` to see custom-tool print statements proving
    each tool is actually invoked.
    """

    async def test_search_convert_note(self):
        """google_search + unit_converter + note_taker in one session."""
        agent = Agent(
            name="EvidenceFlash",
            role="Research assistant with conversion and notes",
            objective="Search, convert units, and save notes",
            narrative=(
                "Tools:\n"
                "1. google_search - web search\n"
                "2. code_execution - run Python code\n"
                "3. unit_converter - convert units "
                "(km/miles, celsius/fahrenheit, kg/pounds, liters/gallons)\n"
                "4. note_taker - save/list notes"
            ),
            llm=BaseGemini(model_name="gemini-2.5-flash", temperature=0.0),
            tools=[
                GeminiTool.google_search(),
                GeminiTool.code_execution(),
                UnitConverterTool(),
                NoteTakerTool(),
            ],
            config=AgentConfig(
                execution_mode="standard",
                verbose=False,
                enable_tracing=True,
                max_tool_calls=15,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            Task(
                id="evidence-flash-1",
                objective=(
                    "1. Search the web for the distance from New York to London in km.\n"
                    "2. Convert that distance from km to miles using unit_converter.\n"
                    "3. Save a note titled 'NY-London' with both distances."
                ),
            )
        )

        assert result.status.value == "success"
        tool_names = [tc.tool_name for tc in result.tool_calls]
        assert "google_search" in tool_names
        assert "unit_converter" in tool_names
        assert "note_taker" in tool_names
        assert all(tc.success for tc in result.tool_calls)

        for tc in result.tool_calls:
            if tc.tool_name == "google_search":
                assert tc.duration_ms > 100, (
                    f"Proxy google_search should take >100ms (API call), "
                    f"got {tc.duration_ms:.1f}ms"
                )

    async def test_code_execution_and_converter(self):
        """code_execution (proxy) + unit_converter (custom) together."""
        agent = Agent(
            name="EvidenceCodeCalc",
            role="Calculation assistant",
            objective="Calculate and convert units",
            narrative=(
                "Tools:\n"
                "1. code_execution - run Python code\n"
                "2. unit_converter - convert units"
            ),
            llm=BaseGemini(model_name="gemini-2.5-flash", temperature=0.0),
            tools=[GeminiTool.code_execution(), UnitConverterTool()],
            config=AgentConfig(
                execution_mode="standard",
                verbose=False,
                enable_tracing=True,
                max_tool_calls=10,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            Task(
                id="evidence-flash-2",
                objective=(
                    "Use code_execution to calculate factorial of 12. "
                    "Then convert 100 celsius to fahrenheit using unit_converter."
                ),
            )
        )

        assert result.status.value == "success"
        tool_names = [tc.tool_name for tc in result.tool_calls]
        assert "code_execution" in tool_names
        assert "unit_converter" in tool_names
        assert all(tc.success for tc in result.tool_calls)

        for tc in result.tool_calls:
            if tc.tool_name == "code_execution":
                assert tc.duration_ms > 100, (
                    f"Proxy code_execution should take >100ms (API call), "
                    f"got {tc.duration_ms:.1f}ms"
                )


@pytest.mark.asyncio
class TestMultiToolEvidencePro:
    """Comprehensive evidence: ALL 4 tools in one session on gemini-2.5-pro.

    This test forces the LLM to use google_search, code_execution,
    unit_converter, and note_taker in a single agent execution.
    """

    async def test_all_four_tools_single_session(self):
        """All 4 tools exercised in one session with timing assertions."""
        agent = Agent(
            name="EvidencePro",
            role="Full-suite research assistant",
            objective="Use all tools to complete research",
            narrative=(
                "You have exactly 4 tools. Use ALL of them:\n"
                "1. google_search - Search the web\n"
                "2. code_execution - Execute Python code\n"
                "3. unit_converter - Convert units "
                "(km/miles, celsius/fahrenheit, kg/pounds, liters/gallons)\n"
                "4. note_taker - Save/list notes (action='save'/'list')"
            ),
            llm=BaseGemini(model_name="gemini-2.5-pro", temperature=0.0),
            tools=[
                GeminiTool.google_search(),
                GeminiTool.code_execution(),
                UnitConverterTool(),
                NoteTakerTool(),
            ],
            config=AgentConfig(
                execution_mode="standard",
                verbose=False,
                enable_tracing=True,
                max_tool_calls=15,
            ),
        )
        await agent.initialize()

        result = await agent.execute(
            Task(
                id="evidence-pro-1",
                objective=(
                    "Step 1: google_search for the temperature in Tokyo today.\n"
                    "Step 2: unit_converter to convert that from celsius to fahrenheit.\n"
                    "Step 3: code_execution to calculate 25 ** 5.\n"
                    "Step 4: note_taker action='save' title='Tokyo Research' with findings.\n"
                    "Step 5: note_taker action='list' to show saved notes."
                ),
            )
        )

        assert result.status.value == "success"
        tool_names = [tc.tool_name for tc in result.tool_calls]

        assert "google_search" in tool_names, f"Missing google_search: {tool_names}"
        assert "unit_converter" in tool_names, f"Missing unit_converter: {tool_names}"
        assert "code_execution" in tool_names, f"Missing code_execution: {tool_names}"
        assert "note_taker" in tool_names, f"Missing note_taker: {tool_names}"
        assert all(tc.success for tc in result.tool_calls)

        proxy_calls = [
            tc
            for tc in result.tool_calls
            if tc.tool_name in ("google_search", "code_execution")
        ]
        custom_calls = [
            tc
            for tc in result.tool_calls
            if tc.tool_name in ("unit_converter", "note_taker")
        ]

        for tc in proxy_calls:
            assert tc.duration_ms > 100, (
                f"Proxy {tc.tool_name} should take >100ms (real API sub-call), "
                f"got {tc.duration_ms:.1f}ms"
            )
        for tc in custom_calls:
            assert tc.duration_ms < 50, (
                f"Custom {tc.tool_name} should be fast (<50ms), "
                f"got {tc.duration_ms:.1f}ms"
            )
