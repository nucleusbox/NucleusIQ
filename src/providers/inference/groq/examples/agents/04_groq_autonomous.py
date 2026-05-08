"""Gear 3 AUTONOMOUS: orchestration + quality checks (real Agent + Groq).

Uses local tools; Critic/Refiner add extra LLM traffic — keep the task small.

Run::

    uv run python examples/agents/04_groq_autonomous.py
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
def add_integers(x: int, y: int) -> str:
    """Add two integers.

    Args:
        x: First addend.
        y: Second addend.
    """
    return str(x + y)


@tool
def fact_lookup(topic: str) -> str:
    """Return a one-line fact for known topics (demo).

    Args:
        topic: Short topic keyword.
    """
    facts = {
        "rust": "Rust emphasizes memory safety without a garbage collector.",
        "python": "Python is widely used for automation and ML.",
        "groq": "Groq offers fast LLM inference on their LPU stack.",
    }
    key = topic.strip().lower()
    for k, v in facts.items():
        if k in key:
            return v
    return f"No demo fact for '{topic}'."


async def main() -> None:
    llm = BaseGroq(model_name=_model(), async_mode=True)

    agent = Agent(
        name="groq-autonomous",
        prompt=ZeroShotPrompt().configure(
            system=(
                "You are a careful analyst. Use tools when you need facts or arithmetic, "
                "then synthesize a short answer."
            ),
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            require_quality_check=True,
            max_iterations=4,
            llm_max_output_tokens=1024,
            llm_params=GroqLLMParams(temperature=0.5, max_output_tokens=1024),
        ),
        tools=[add_integers, fact_lookup],
    )

    await agent.initialize()

    result = await agent.execute(
        Task(
            id="groq-auto-1",
            objective=(
                "Look up one fact about Python and compute 19 + 23 using the tools; "
                "then give a two-sentence summary."
            ),
        )
    )
    print("=== AUTONOMOUS + local tools ===")
    print(result.output)
    print(f"\nTool calls: {result.tool_call_count}")
    print(f"Tokens: {agent.last_usage.total.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
