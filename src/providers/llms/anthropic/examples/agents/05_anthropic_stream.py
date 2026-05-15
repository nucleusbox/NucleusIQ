"""Stream tokens from an Agent using **DIRECT** mode + Anthropic (``execute_stream``).

Mirrors OpenAI/Groq “live stream” ergonomics — prints TOKEN chunks until COMPLETE.

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/05_anthropic_stream.py

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
from nucleusiq.streaming.events import StreamEventType  # noqa: E402
from nucleusiq_anthropic import BaseAnthropic  # noqa: E402


def _model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")


async def main() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)

    agent = Agent(
        name="anthropic-stream-demo",
        prompt=ZeroShotPrompt().configure(
            system="Reply in fluent prose. Give exactly three short sentences.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=LLMParams(temperature=0.5, max_output_tokens=384),
        ),
    )

    await agent.initialize()

    task = Task(
        id="anthropic-stream-1",
        objective="Briefly explain what an agent framework does for LLM apps.",
    )

    print("=== STREAM (tokens as they arrive) ===\n", flush=True)
    async for event in agent.execute_stream(task):
        if event.type == StreamEventType.TOKEN and event.token:
            print(event.token, end="", flush=True)
        elif event.type == StreamEventType.THINKING and event.message:
            print(f"\n[thinking] {event.message}", flush=True)
        elif event.type == StreamEventType.ERROR and event.message:
            print(f"\n[error] {event.message}", flush=True)
            break
        elif event.type == StreamEventType.COMPLETE:
            print("\n\n--- COMPLETE ---", flush=True)
            if event.metadata:
                usage = event.metadata.get("usage") or {}
                if usage:
                    print(f"usage: {usage}")


if __name__ == "__main__":
    asyncio.run(main())
