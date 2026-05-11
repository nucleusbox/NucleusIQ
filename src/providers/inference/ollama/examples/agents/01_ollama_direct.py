"""DIRECT mode Agent + Ollama (real LLM call).

Run from ``src/providers/inference/ollama``::

    uv run python examples/agents/01_ollama_direct.py

Set ``OLLAMA_API_KEY`` in ``.env`` if your endpoint requires a Bearer token.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Load environment variables (optional) — same pattern as OpenAI examples
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nucleusiq.agents import Agent  # noqa: E402
from nucleusiq.agents.config import AgentConfig, ExecutionMode  # noqa: E402
from nucleusiq.agents.task import Task  # noqa: E402
from nucleusiq.prompts.zero_shot import ZeroShotPrompt  # noqa: E402
from nucleusiq_ollama import BaseOllama, OllamaLLMParams  # noqa: E402


def _model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2")


async def main() -> None:
    host = os.getenv("OLLAMA_HOST")
    print(f"model={_model()!r} host={host or '(default)'}", flush=True)

    llm = BaseOllama(model_name=_model(), async_mode=True)

    agent = Agent(
        name="ollama-direct",
        prompt=ZeroShotPrompt().configure(
            system="You are a concise assistant. Reply in one or two short sentences.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=OllamaLLMParams(temperature=0.3, max_output_tokens=256),
        ),
    )

    await agent.initialize()

    result = await agent.execute(
        Task(id="ollama-direct-1", objective="What is the capital of France?")
    )
    print("=== DIRECT (Ollama) ===")
    print(result.output)
    usage = getattr(agent, "last_usage", None)
    if usage is not None:
        print(f"\nTokens (approx): {usage.total.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
