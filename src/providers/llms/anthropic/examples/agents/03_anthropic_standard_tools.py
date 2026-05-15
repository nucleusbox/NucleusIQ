"""Gear 2 STANDARD: tool loop with local tools (Agent + Anthropic Claude).

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/03_anthropic_standard_tools.py

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
def get_weather(city: str) -> str:
    """Return fake weather for a city (demo data).

    Args:
        city: City name.
    """
    data = {
        "paris": "12°C, light rain",
        "tokyo": "22°C, sunny",
        "london": "15°C, cloudy",
    }
    return data.get(city.lower(), f"No demo weather for '{city}'")


@tool
def get_population(city: str) -> str:
    """Return fake population stats (demo data).

    Args:
        city: City name.
    """
    data = {
        "paris": "~2.1M (city)",
        "tokyo": "~14M (metro)",
        "london": "~9M (city)",
    }
    return data.get(city.lower(), f"No demo population for '{city}'")


async def main() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)

    agent = Agent(
        name="anthropic-standard",
        prompt=ZeroShotPrompt().configure(
            system=(
                "You are a travel assistant. Use tools for facts; then answer in 2-4 sentences."
            ),
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            llm_params=LLMParams(temperature=0.4, max_output_tokens=512),
        ),
        tools=[get_weather, get_population],
    )

    await agent.initialize()

    result = await agent.execute(
        Task(
            id="anthropic-standard-1",
            objective=(
                "For Tokyo: what's the weather and roughly how many people live there?"
            ),
        )
    )
    print("=== STANDARD + local tools (multi-step loop) ===")
    print(result.output)
    print(f"\nTool calls: {result.tool_call_count}")
    print(f"Tokens: {agent.last_usage.total.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
