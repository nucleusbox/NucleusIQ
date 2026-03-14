"""
Example: Built-in File Tools + Attachments (Combined Pattern)

Demonstrates both file-handling approaches in NucleusIQ:

  1. Attachments (file-as-context) -- embed file content in the prompt
  2. File Tools (file-as-workspace) -- agent explores files iteratively
  3. Combined -- attach a known file AND let the agent explore others

Run with:
    python examples/agents/file_tools_example.py
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.builtin import (
    DirectoryListTool,
    FileExtractTool,
    FileReadTool,
    FileSearchTool,
)

logging.basicConfig(level=logging.WARNING)


def create_sample_workspace() -> str:
    """Create a temp workspace with sample data files."""
    ws = tempfile.mkdtemp(prefix="nucleusiq_example_")
    p = Path(ws)

    (p / "report.txt").write_text(
        "Q1 Revenue: $1.2M\nQ2 Revenue: $1.5M\n"
        "Q3 Revenue: $1.8M\nQ4 Revenue: $2.5M\n"
        "Total: $7.0M (up 30% YoY)\n"
    )
    (p / "sales.csv").write_text(
        "product,region,amount\n"
        "Widget,US,500000\nWidget,EU,300000\n"
        "Gadget,US,400000\nGadget,APAC,200000\n"
    )
    (p / "config.json").write_text(
        json.dumps({"version": "2.0", "env": "production", "replicas": 3})
    )
    sub = p / "notes"
    sub.mkdir()
    (sub / "meeting.txt").write_text(
        "Action items:\n- Finalize Q4 report\n- Plan Q1 strategy\n"
    )
    return ws


async def example_tools_only():
    """Agent uses file tools to explore a workspace."""
    print("\n" + "=" * 60)
    print("1. File Tools Only (agent explores iteratively)")
    print("=" * 60)

    ws = create_sample_workspace()
    agent = Agent(
        name="Explorer",
        role="Data Analyst",
        objective="Analyze data files",
        llm=MockLLM(),
        tools=[
            DirectoryListTool(workspace_root=ws),
            FileReadTool(workspace_root=ws),
            FileSearchTool(workspace_root=ws),
            FileExtractTool(workspace_root=ws),
        ],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )

    result = await agent.execute(
        Task(id="explore", objective="Find all revenue figures in the workspace")
    )
    print(f"  Result: {str(result)[:120]}...")
    print(f"  Tools available: {[t.name for t in agent.tools]}")


async def example_attachment_plus_tools():
    """Combine an attachment (known context) with tools (exploration)."""
    print("\n" + "=" * 60)
    print("2. Combined: Attachment + File Tools")
    print("=" * 60)

    ws = create_sample_workspace()
    agent = Agent(
        name="Researcher",
        role="Financial Analyst",
        objective="Cross-reference financial data",
        llm=MockLLM(),
        tools=[
            FileReadTool(workspace_root=ws),
            FileSearchTool(workspace_root=ws),
        ],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )

    task = Task(
        id="cross-ref",
        objective=(
            "Compare the attached summary with the raw data files. "
            "Use the file tools to look up detailed figures."
        ),
        attachments=[
            Attachment(
                type=AttachmentType.TEXT,
                data="Executive Summary: Revenue grew 30% to $7M.",
                name="exec_summary.txt",
            ),
        ],
    )

    result = await agent.execute(task)
    print("  Attachment: exec_summary.txt (in prompt)")
    print(f"  Tools: {[t.name for t in agent.tools]} (for exploration)")
    print(f"  Result: {str(result)[:120]}...")


async def example_extract_csv():
    """Use FileExtractTool for structured data extraction."""
    print("\n" + "=" * 60)
    print("3. FileExtractTool -- structured CSV extraction")
    print("=" * 60)

    ws = create_sample_workspace()
    tool = FileExtractTool(workspace_root=ws)
    await tool.initialize()

    result = await tool.execute(path="sales.csv")
    print(f"  {result}")


async def main():
    print("NucleusIQ File Tools + Attachments Example")
    print("=" * 60)
    print()
    print("Two approaches:")
    print("  Attachment  = file-as-context (one-shot, in the prompt)")
    print("  File Tool   = file-as-workspace (iterative, agent decides)")
    print()
    print("See https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/file-handling/ for the full decision guide.")

    await example_tools_only()
    await example_attachment_plus_tools()
    await example_extract_csv()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
