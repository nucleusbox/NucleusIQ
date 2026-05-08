"""Gear 1 DIRECT with local function tool (real Agent + Groq).

One LLM call may request a tool; Direct mode handles a single tool round-trip.

Run::

    uv run python examples/agents/02_groq_direct_with_tool.py
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
from nucleusiq.prompts.zero_shot import ZeroShotPrompt  # noqa: E402
from nucleusiq.tools.decorators import tool  # noqa: E402
from nucleusiq_groq import BaseGroq, GroqLLMParams  # noqa: E402


def _model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@tool
def multiply_numbers(a: int, b: int) -> str:
    """Multiply two integers and return the product as text.

    Args:
        a: First factor.
        b: Second factor.
    """
    return str(a * b)


async def main() -> None:
    llm = BaseGroq(model_name=_model(), async_mode=True)

    agent = Agent(
        name="groq-direct-tools",
        prompt=ZeroShotPrompt().configure(
            system="You are a math helper. When asked for a product, call multiply_numbers.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=GroqLLMParams(temperature=0.2, max_output_tokens=256),
        ),
        tools=[multiply_numbers],
    )

    await agent.initialize()

    result = await agent.execute(
        Task(id="groq-direct-tool-1", objective="What is 7 times 8? Use the tool.")
    )
    print("=== DIRECT + local function tool ===")
    print(result.output)
    print(f"\nTool calls recorded: {result.tool_call_count}")
    if result.tool_calls:
        for tc in result.tool_calls:
            print(f"  - {tc.name} ok={tc.success}")


if __name__ == "__main__":
    asyncio.run(main())
