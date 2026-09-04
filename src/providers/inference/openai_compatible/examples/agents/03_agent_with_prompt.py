"""A NucleusIQ agent backed by a self-hosted model.

Shows how the pieces connect:

    PromptFactory  -> Agent.prompt   -> system / user messages
    Task.objective -> a user message
    Agent.llm      -> OpenAICompatibleLLM -> POST /v1/chat/completions

The provider is injected, not discovered: there is no registry mapping model
names to providers, so a self-hosted model needs no naming convention.

Run:
    python examples/agents/03_agent_with_prompt.py
"""

from __future__ import annotations

import asyncio
import os

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq_openai_compatible import OpenAICompatibleLLM

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gemma-4-27b-it")


async def main() -> None:
    llm = OpenAICompatibleLLM(
        base_url=BASE_URL,
        model=MODEL,
        engine="vllm",
        context_window=32_768,
    )

    # The agent reads prompt.system and prompt.user directly; the task
    # objective arrives as a separate user message after them.
    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
        system="You are a senior Python engineer. Answer in at most three sentences.",
        user="Prefer concrete, runnable advice over general principles.",
    )

    agent = Agent(
        name="CodeAdvisor",
        role="Engineer",
        objective="Give focused Python advice",
        prompt=prompt,
        llm=llm,
        config=AgentConfig(
            # DIRECT is the lightest path: one LLM call, no tool loop.
            execution_mode=ExecutionMode.DIRECT,
            verbose=False,
        ),
    )
    await agent.initialize()

    # The context engine sizes its budget from the provider's window, so a
    # wrong context_window here means silent truncation or a server 400.
    print(f"Context window: {agent.llm.get_context_window():,} tokens\n")

    result = await agent.execute(
        Task(
            id="advice-1",
            objective="How should I structure a Python monorepo with several "
            "independently publishable packages?",
        )
    )

    print(result.output)
    print(f"\nstatus: {result.status}")


if __name__ == "__main__":
    asyncio.run(main())
