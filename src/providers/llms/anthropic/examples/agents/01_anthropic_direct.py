"""Gear 1 DIRECT: fast chat, no tools (Agent + Anthropic Claude).

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/01_anthropic_direct.py

Requires ``ANTHROPIC_API_KEY`` (e.g. repo-root ``.env``).
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
from nucleusiq_anthropic import BaseAnthropic  # noqa: E402


def _model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")


async def main() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)

    agent = Agent(
        name="anthropic-direct",
        prompt=ZeroShotPrompt().configure(
            system="You are a concise assistant. Reply in one or two short sentences.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=LLMParams(temperature=0.3, max_output_tokens=256),
        ),
    )

    await agent.initialize()

    result = await agent.execute(
        Task(id="anthropic-direct-1", objective="What is the capital of France?")
    )
    print("=== DIRECT (no tools) ===")
    print(result.output)
    print(f"\nTokens (approx): {agent.last_usage.total.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
