"""All three execution modes in one driver (Anthropic Claude).

Similar intent to ``openai/examples/agents/execution_modes_example.py`` —
compact smoke of **DIRECT → STANDARD → AUTONOMOUS**.

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/06_anthropic_execution_modes.py

Requires ``ANTHROPIC_API_KEY``. Uses several API calls in one process.
"""

from __future__ import annotations

import asyncio
import logging
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


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@tool
def add_integers(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First addend.
        b: Second addend.
    """
    return str(a + b)


async def run_direct() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)
    agent = Agent(
        name="mode-direct",
        prompt=ZeroShotPrompt().configure(system="Answer in one sentence."),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=LLMParams(temperature=0.3, max_output_tokens=180),
        ),
    )
    await agent.initialize()
    r = await agent.execute(Task(id="m1", objective="Name two colors."))
    log.info("\n[DIRECT]\n%s", r.output)


async def run_standard() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)
    agent = Agent(
        name="mode-standard",
        prompt=ZeroShotPrompt().configure(
            system="Use add_integers for arithmetic when asked.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.STANDARD,
            llm_params=LLMParams(temperature=0.2, max_output_tokens=320),
        ),
        tools=[add_integers],
    )
    await agent.initialize()
    r = await agent.execute(
        Task(id="m2", objective="What is 41 + 1? Call the tool and state the answer."),
    )
    log.info("\n[STANDARD] tools=%s\n%s", r.tool_call_count, r.output)


async def run_autonomous() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)
    agent = Agent(
        name="mode-autonomous",
        prompt=ZeroShotPrompt().configure(
            system="Use tools briefly; summarize in two sentences.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            require_quality_check=True,
            max_iterations=3,
            llm_params=LLMParams(temperature=0.35, max_output_tokens=384),
        ),
        tools=[add_integers],
    )
    await agent.initialize()
    r = await agent.execute(
        Task(
            id="m3",
            objective="Compute 100 + 5 with the tool, then conclude with what you computed.",
        ),
    )
    log.info("\n[AUTONOMOUS] tools=%s\n%s", r.tool_call_count, r.output)


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        log.error("Missing ANTHROPIC_API_KEY (set in repo-root .env or environment).")
        raise SystemExit(1)
    await run_direct()
    await run_standard()
    await run_autonomous()


if __name__ == "__main__":
    asyncio.run(main())
