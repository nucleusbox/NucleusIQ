"""Gear 1 DIRECT with one local tool (Agent + Anthropic).

One short tool round-trip; compare with **STANDARD** for multi-step tool loops.

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/02_anthropic_direct_with_tool.py

Requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from util_env import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from nucleusiq.agents import Agent  # noqa: E402
from nucleusiq.agents.config import AgentConfig, ExecutionMode  # noqa: E402
from nucleusiq.agents.task import Task  # noqa: E402
from nucleusiq.llms.llm_params import LLMParams  # noqa: E402
from nucleusiq.prompts.zero_shot import ZeroShotPrompt  # noqa: E402
from nucleusiq.tools.decorators import tool  # noqa: E402
from nucleusiq_anthropic import BaseAnthropic  # noqa: E402


def _model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")


@tool
def multiply_numbers(a: int, b: int) -> str:
    """Multiply two integers and return the product as text.

    Args:
        a: First factor.
        b: Second factor.
    """
    return str(a * b)


async def main() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)

    agent = Agent(
        name="anthropic-direct-tools",
        prompt=ZeroShotPrompt().configure(
            system="You are a math helper. When asked for a product, call multiply_numbers.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=LLMParams(temperature=0.2, max_output_tokens=256),
        ),
        tools=[multiply_numbers],
    )

    await agent.initialize()

    result = await agent.execute(
        Task(id="anthropic-direct-tool-1", objective="What is 7 times 8? Use the tool.")
    )
    print("=== DIRECT + local function tool ===")
    print(result.output)
    print(f"\nTool calls recorded: {result.tool_call_count}")
    if result.tool_calls:
        for tc in result.tool_calls:
            print(f"  - {tc.name} ok={tc.success}")


if __name__ == "__main__":
    asyncio.run(main())
