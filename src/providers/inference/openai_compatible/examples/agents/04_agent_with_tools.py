"""The tool loop against a self-hosted model.

Every tool is executed locally: a self-hosted inference server has no
server-side tool runtime, so ``NATIVE_TOOL_TYPES`` is empty and the round
trip is always

    model requests a call -> NucleusIQ runs the Python function
                          -> result is appended as a "tool" message
                          -> model is called again

**Your server must be started for tool calling**, or the model will describe
the tool in prose instead of calling it:

    vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes

The parser must match the model family (``hermes``, ``mistral``,
``llama3_json``, ``granite``, …).

Run:
    python examples/agents/04_agent_with_tools.py
"""

from __future__ import annotations

import asyncio
import os

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import tool
from nucleusiq_openai_compatible import OpenAICompatibleLLM

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gemma-4-27b-it")


@tool("get_weather")
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    fake = {"Paris": "17C, light rain", "Tokyo": "24C, clear"}
    return fake.get(city, f"No data for {city}")


@tool("convert_currency")
async def convert_currency(amount: float, source: str, target: str) -> str:
    """Convert an amount from one currency to another."""
    rates = {("EUR", "JPY"): 165.0, ("USD", "EUR"): 0.92}
    rate = rates.get((source, target))
    if rate is None:
        return f"No rate for {source}->{target}"
    return f"{amount} {source} = {amount * rate:.2f} {target}"


async def main() -> None:
    llm = OpenAICompatibleLLM(
        base_url=BASE_URL,
        model=MODEL,
        engine="vllm",
        context_window=32_768,
        # The vllm preset already declares tool support; set this explicitly
        # only when using engine="generic" against a server you know can call
        # tools.
        supports_tools=True,
    )

    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
        system="You are a travel assistant. Use the tools before answering.",
    )

    agent = Agent(
        name="TravelAgent",
        role="Assistant",
        objective="Answer travel questions using tools",
        prompt=prompt,
        llm=llm,
        tools=[get_weather, convert_currency],
        config=AgentConfig(
            # STANDARD runs the full tool loop; DIRECT allows tools but is
            # tuned for one or two calls.
            execution_mode=ExecutionMode.STANDARD,
            verbose=False,
        ),
    )
    await agent.initialize()

    result = await agent.execute(
        Task(
            id="trip-1",
            objective=("What is the weather in Tokyo, and what is 250 EUR in JPY?"),
        )
    )

    print(result.output)
    print(f"\nstatus     : {result.status}")
    if result.tool_calls:
        print(f"tool calls : {[c.tool_name for c in result.tool_calls]}")


if __name__ == "__main__":
    asyncio.run(main())
