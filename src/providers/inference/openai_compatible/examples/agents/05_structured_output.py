"""Structured output, including the case that breaks on vLLM.

Without tools, the schema is sent natively as
``response_format={"type": "json_schema", ...}``.

**With tools it cannot be.**  vLLM's guided decoding forces the model to emit
text matching the schema, which makes emitting a tool call impossible — so
the model answers with JSON and never calls the tool, silently.  The default
``structured_output_with_tools="prompt"`` policy handles this by omitting
``response_format`` and injecting the schema into the system prompt, which
keeps both the tool loop and the schema.

The other two policies:

* ``"drop"``  — ignore the schema when tools are present (warns).
* ``"error"`` — refuse the combination before any HTTP call.

Run:
    python examples/agents/05_structured_output.py
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
from pydantic import BaseModel, Field

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gemma-4-27b-it")


class Assessment(BaseModel):
    """The shape we want back."""

    city: str = Field(description="The city assessed")
    verdict: str = Field(description="A one-sentence recommendation")
    confidence: float = Field(ge=0.0, le=1.0)


@tool("get_weather")
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return {"Paris": "17C, light rain", "Tokyo": "24C, clear"}.get(
        city, f"No data for {city}"
    )


def build_llm(**overrides) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        base_url=BASE_URL,
        model=MODEL,
        engine="vllm",
        context_window=32_768,
        **overrides,
    )


def build_agent(llm, *, tools=None) -> Agent:
    return Agent(
        name="Assessor",
        role="Analyst",
        objective="Assess travel conditions",
        prompt=PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
            system="You are a precise travel analyst.",
        ),
        llm=llm,
        tools=tools or [],
        response_format=Assessment,
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False),
    )


async def without_tools() -> None:
    print("=== Schema only (native json_schema) ===")
    agent = build_agent(build_llm())
    await agent.initialize()
    result = await agent.execute(
        Task(id="a1", objective="Assess visiting Tokyo in April. Be brief.")
    )
    print(result.output)


async def with_tools() -> None:
    print("\n=== Schema + tools (prompt policy) ===")
    # The schema moves into the system prompt so the tool call survives.
    agent = build_agent(build_llm(), tools=[get_weather])
    await agent.initialize()
    result = await agent.execute(
        Task(id="a2", objective="Check Tokyo's weather, then assess visiting.")
    )
    print(result.output)


async def strict_refusal() -> None:
    print("\n=== Schema + tools (error policy) ===")
    agent = build_agent(
        build_llm(structured_output_with_tools="error"), tools=[get_weather]
    )
    await agent.initialize()
    try:
        await agent.execute(Task(id="a3", objective="Check Tokyo, then assess."))
    except Exception as exc:  # noqa: BLE001 - demonstrating the failure mode
        print(f"Refused as configured: {type(exc).__name__}: {exc}")


async def main() -> None:
    await without_tools()
    await with_tools()
    await strict_refusal()


if __name__ == "__main__":
    asyncio.run(main())
