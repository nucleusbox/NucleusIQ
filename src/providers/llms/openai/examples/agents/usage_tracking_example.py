"""
Example: Usage Tracking with OpenAI

Shows the ONE method you need: ``agent.last_usage``

After every ``agent.execute()``, call ``agent.last_usage`` to get a
``UsageSummary`` Pydantic model with the full token breakdown:
  - total tokens (prompt + completion + reasoning)
  - by_purpose  (main, tool_loop, planning, critic, refiner)
  - by_origin   (user vs framework overhead)

Run with:
    # With a real OpenAI key:
    export OPENAI_API_KEY=sk-...
    python examples/agents/usage_tracking_example.py

    # Without a key (uses MockLLM):
    python examples/agents/usage_tracking_example.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.tools.builtin import FileReadTool

try:
    from nucleusiq_openai import BaseOpenAI

    HAS_OPENAI = os.getenv("OPENAI_API_KEY") is not None
except ImportError:
    HAS_OPENAI = False


def _separator(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print("=" * 64)


async def main() -> None:
    _separator("Usage Tracking Example")

    if HAS_OPENAI:
        llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0)
        print("  Using: OpenAI gpt-4o-mini")
    else:
        from nucleusiq.llms.mock_llm import MockLLM

        llm = MockLLM()
        print("  Using: MockLLM (set OPENAI_API_KEY for real results)")

    import tempfile
    from pathlib import Path

    ws = tempfile.mkdtemp(prefix="usage_demo_")
    (Path(ws) / "data.json").write_text(
        json.dumps({"revenue": {"q1": 1200000, "q2": 1500000, "q3": 1800000}})
    )

    agent = Agent(
        name="Analyst",
        role="Data Analyst",
        objective="Analyze business data",
        llm=llm,
        tools=[FileReadTool(ws)],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )

    # ------------------------------------------------------------------ #
    # Step 1: Execute a task                                               #
    # ------------------------------------------------------------------ #
    _separator("1. Execute a task")

    result = await agent.execute(
        Task(id="t1", objective="Read data.json and summarize the revenue trend")
    )
    print(f"  Result: {str(result)[:100]}...")

    # ------------------------------------------------------------------ #
    # Step 2: .display() -- consolidated view in one call                  #
    # ------------------------------------------------------------------ #
    _separator("2. usage.display() -- everything in one call")

    usage = agent.last_usage
    print()
    print(usage.display())

    # ------------------------------------------------------------------ #
    # Step 3: Individual fields (when you need just one value)             #
    # ------------------------------------------------------------------ #
    _separator("3. Individual attribute access")

    print(f"\n  usage.total.prompt_tokens       = {usage.total.prompt_tokens}")
    print(f"  usage.total.total_tokens        = {usage.total.total_tokens}")
    print(f"  usage.call_count                = {usage.call_count}")
    if "user" in usage.by_origin:
        print(
            f"  usage.by_origin['user'].tokens  = {usage.by_origin['user'].total_tokens}"
        )
    if "framework" in usage.by_origin:
        print(
            f"  usage.by_origin['framework']    = {usage.by_origin['framework'].total_tokens}"
        )

    # ------------------------------------------------------------------ #
    # Step 4: .summary() -- plain dict for logging / dashboards            #
    # ------------------------------------------------------------------ #
    _separator("4. usage.summary() -- plain dict")

    d = usage.summary()
    print(f"\n  type: {type(d).__name__}")
    print(f"  json:\n{json.dumps(d, indent=2)}")

    # ------------------------------------------------------------------ #
    # Step 5: Usage resets each execute()                                  #
    # ------------------------------------------------------------------ #
    _separator("5. Usage resets on each execute()")

    await agent.execute(Task(id="t2", objective="What is 2 + 2?"))
    usage2 = agent.last_usage
    print("\n  After second task:")
    print(usage2.display())
    print("\n  (Previous task's usage is gone -- each execute() starts fresh)")

    _separator("API Summary")
    print(
        "\n"
        "  usage = agent.last_usage             # UsageSummary model\n"
        "  usage.summary()                      # plain dict\n"
        "  usage.display()                      # formatted string\n"
        "  usage.total.prompt_tokens            # single int\n"
        "  usage.by_origin['user'].total_tokens # single int\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
