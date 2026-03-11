"""
Example: v0.5.0 Features — Token Origin Split, FileWriteTool, and Tool UX

Demonstrates every new feature in the v0.5.0 release:

  1. Token Origin Split — agent.last_usage now has by_origin breakdown
  2. FileWriteTool — write / append / backup
  3. FileExtractTool query filtering — columns for CSV, key_path for JSON
  4. FileSearchTool configurable extensions — include / exclude
  5. DirectoryListTool max_entries — truncation on large dirs
  6. FileReadTool encoding auto-detection — "auto" default

Run with:
    python examples/agents/v050_features_example.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.builtin import (
    DirectoryListTool,
    FileExtractTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
)


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _create_workspace() -> str:
    ws = tempfile.mkdtemp(prefix="nucleusiq_v050_")
    p = Path(ws)
    (p / "report.txt").write_text(
        "Q1 Revenue: $1.2M\nQ2 Revenue: $1.5M\nQ3 Revenue: $1.8M\n"
    )
    (p / "sales.csv").write_text(
        "product,region,amount,quarter\n"
        "Widget,US,500000,Q1\nWidget,EU,300000,Q2\n"
        "Gadget,US,400000,Q1\nGadget,APAC,200000,Q3\n"
    )
    (p / "config.json").write_text(
        json.dumps(
            {
                "app": {"name": "MyApp", "version": "2.1"},
                "database": {"host": "db.prod.internal", "port": 5432},
                "features": ["auth", "billing", "reports"],
            }
        )
    )
    (p / "readme.md").write_text("# Project\nhello world\n")
    sub = p / "src"
    sub.mkdir()
    (sub / "app.py").write_text("def main():\n    print('hello')\n")
    (sub / "utils.py").write_text("def helper():\n    return 42\n")
    return ws


# ================================================================== #
# 1. Token Origin Split — the real user workflow                       #
# ================================================================== #


async def example_token_origin_split() -> None:
    """Show how agent.last_usage now separates user vs framework tokens.

    After agent.execute(), call agent.last_usage to get the full
    breakdown. The new 'by_origin' key tells you how much of your
    token spend was your actual user content vs framework overhead.
    """
    _banner("1. Token Origin Split (agent.last_usage)")

    ws = _create_workspace()

    agent = Agent(
        name="Analyst",
        role="Data Analyst",
        objective="Analyze data",
        llm=MockLLM(),
        tools=[
            FileReadTool(ws),
            FileExtractTool(ws),
        ],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )

    # Run a task -- the agent will make multiple LLM calls:
    #   1. MAIN call (user's task) -- tagged as USER origin
    #   2. TOOL_LOOP calls (agent decides to use tools) -- tagged as FRAMEWORK
    result = await agent.execute(
        Task(id="t1", objective="Read report.txt and summarize the revenue figures")
    )
    print(f"  Result: {str(result)[:80]}...")

    # ---- This is what the user actually calls ----
    # agent.last_usage returns a UsageSummary Pydantic model.
    # Access fields via attributes, or call .model_dump() for a plain dict.
    usage = agent.last_usage

    print("\n  agent.last_usage -> UsageSummary (Pydantic model)")
    print("\n  Attribute access (typed, IDE-autocomplete friendly):")
    print(f"    usage.total.prompt_tokens     = {usage.total.prompt_tokens}")
    print(f"    usage.total.completion_tokens = {usage.total.completion_tokens}")
    print(f"    usage.total.total_tokens      = {usage.total.total_tokens}")
    print(f"    usage.total.reasoning_tokens  = {usage.total.reasoning_tokens}")
    print(f"    usage.call_count              = {usage.call_count}")

    print("\n  usage.by_purpose:")
    for purpose, stats in usage.by_purpose.items():
        print(f"    '{purpose}': tokens={stats.total_tokens}, calls={stats.calls}")

    print("\n  usage.by_origin:  (NEW in v0.5.0)")
    for origin, stats in usage.by_origin.items():
        print(f"    '{origin}': tokens={stats.total_tokens}, calls={stats.calls}")

    # Compute the split percentage
    total = usage.total.total_tokens
    if total > 0:
        user_bucket = usage.by_origin.get("user")
        fw_bucket = usage.by_origin.get("framework")
        user_tokens = user_bucket.total_tokens if user_bucket else 0
        framework_tokens = fw_bucket.total_tokens if fw_bucket else 0
        print("\n  Token split:")
        print(
            f"    User content:      {user_tokens:>5} tokens"
            f" ({user_tokens / total * 100:.0f}%)"
        )
        print(
            f"    Framework overhead: {framework_tokens:>5} tokens"
            f" ({framework_tokens / total * 100:.0f}%)"
        )
    else:
        print("\n  (No token usage reported -- connect a real LLM provider)")

    # .model_dump() gives a plain dict for JSON serialization / logging
    print("\n  usage.model_dump() -> plain dict for JSON/logging:")
    d = usage.model_dump()
    print(f"    type={type(d).__name__}, keys={list(d.keys())}")

    print("\n  How the tagging works:")
    print("    MAIN call      -> origin=USER      (your task objective)")
    print("    TOOL_LOOP call -> origin=FRAMEWORK  (agent orchestration)")
    print("    PLANNING call  -> origin=FRAMEWORK  (autonomous planning)")
    print("    CRITIC call    -> origin=FRAMEWORK  (quality verification)")
    print("    REFINER call   -> origin=FRAMEWORK  (answer refinement)")


# ================================================================== #
# 2. FileWriteTool                                                     #
# ================================================================== #


async def example_file_write_tool() -> None:
    _banner("2. FileWriteTool (write / append / backup)")

    ws = _create_workspace()
    tool = FileWriteTool(ws)

    result = await tool.execute(
        path="output/report.txt", content="# Q1 Report\nRevenue: $1.2M\n"
    )
    print(f"  Create:    {result}")

    result = await tool.execute(
        path="output/report.txt", content="\n## Notes\nGrowth: 15%\n", mode="append"
    )
    print(f"  Append:    {result}")

    result = await tool.execute(
        path="output/report.txt", content="# Q1 Report v2\nRevised.\n"
    )
    print(f"  Overwrite: {result}")
    print(f"  Backup:    {(Path(ws) / 'output' / 'report.txt.bak').exists()}")

    result = await tool.execute(path="../escape.txt", content="bad")
    print(f"  Traversal: {result}")


# ================================================================== #
# 3. FileExtractTool — query filtering                                 #
# ================================================================== #


async def example_extract_filtering() -> None:
    _banner("3. FileExtractTool Query Filtering (columns, key_path)")

    ws = _create_workspace()
    tool = FileExtractTool(ws)

    print("\n  CSV -- all columns:")
    result = await tool.execute(path="sales.csv")
    for line in result.splitlines()[:6]:
        print(f"    {line}")

    print("\n  CSV -- columns='product,amount' (select only these):")
    result = await tool.execute(path="sales.csv", columns="product,amount")
    for line in result.splitlines()[:6]:
        print(f"    {line}")

    print("\n  JSON -- key_path='database.host' (navigate nested keys):")
    result = await tool.execute(path="config.json", key_path="database.host")
    for line in result.splitlines():
        print(f"    {line}")

    print("\n  JSON -- key_path='features.0' (array index):")
    result = await tool.execute(path="config.json", key_path="features.0")
    for line in result.splitlines():
        print(f"    {line}")

    print("\n  JSON -- key_path='missing.path' (not found):")
    result = await tool.execute(path="config.json", key_path="missing.path")
    print(f"    {result}")


# ================================================================== #
# 4. FileSearchTool — configurable binary extensions                   #
# ================================================================== #


async def example_search_config() -> None:
    _banner("4. FileSearchTool Configurable Extensions")

    ws = _create_workspace()

    print("\n  Default (all text files):")
    tool = FileSearchTool(ws)
    result = await tool.execute(pattern="hello")
    for line in result.strip().splitlines():
        print(f"    {line}")

    print("\n  include_extensions=['.py'] (only Python):")
    tool_py = FileSearchTool(ws, include_extensions=[".py"])
    result = await tool_py.execute(pattern="hello")
    for line in result.strip().splitlines():
        print(f"    {line}")

    print("\n  exclude_extensions=['.md'] (skip Markdown):")
    tool_no_md = FileSearchTool(ws, exclude_extensions=[".md"])
    result = await tool_no_md.execute(pattern="hello")
    for line in result.strip().splitlines():
        print(f"    {line}")


# ================================================================== #
# 5. DirectoryListTool — max_entries                                   #
# ================================================================== #


async def example_dir_max_entries() -> None:
    _banner("5. DirectoryListTool max_entries Limit")

    ws = tempfile.mkdtemp(prefix="nucleusiq_v050_")
    for i in range(15):
        (Path(ws) / f"file_{i:03d}.txt").write_text(f"Content {i}\n")

    print("\n  max_entries=200 (default, no truncation):")
    tool = DirectoryListTool(ws)
    result = await tool.execute()
    lines = result.strip().splitlines()
    print(f"    {lines[0]}  ({len(lines) - 1} entries shown)")

    print("\n  max_entries=5 (truncated):")
    tool_small = DirectoryListTool(ws, max_entries=5)
    result = await tool_small.execute()
    for line in result.strip().splitlines():
        print(f"    {line}")


# ================================================================== #
# 6. FileReadTool — encoding auto-detection                            #
# ================================================================== #


async def example_encoding_auto() -> None:
    _banner("6. FileReadTool Encoding Auto-Detection")

    ws = _create_workspace()

    tool = FileReadTool(ws)

    print("\n  UTF-8 file (encoding='auto' -- the new default):")
    result = await tool.execute(path="report.txt")
    print(f"    {result.splitlines()[1]}")

    print("\n  Explicit encoding='utf-8' (still works):")
    result = await tool.execute(path="report.txt", encoding="utf-8")
    print(f"    {result.splitlines()[1]}")

    spec = tool.get_spec()
    print(
        f"\n  Spec encoding default: '{spec['parameters']['properties']['encoding']['default']}'"
    )
    print("  If chardet is installed, auto-detects from first 4KB.")
    print("  If not installed, falls back to utf-8.")


# ================================================================== #
# Main                                                                 #
# ================================================================== #


async def main() -> None:
    print("NucleusIQ v0.5.0 Features Example")
    print("=" * 60)

    await example_token_origin_split()
    await example_file_write_tool()
    await example_extract_filtering()
    await example_search_config()
    await example_dir_max_entries()
    await example_encoding_auto()

    print(f"\n{'=' * 60}")
    print("All v0.5.0 examples completed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
