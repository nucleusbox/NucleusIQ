"""Structured output via Pydantic schema (real Agent + Groq ``response_format``).

Uses ``openai/gpt-oss-20b`` by default (Groq ``json_schema`` support is model-specific).
Override with env ``GROQ_MODEL_STRUCTURED`` if needed.

Run::

    uv run python examples/agents/05_groq_structured_output.py
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
from nucleusiq_groq import BaseGroq, GroqLLMParams  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402


def _model() -> str:
    # ``json_schema`` is not supported on all Groq models; use a listed model or set
    # ``GROQ_MODEL_STRUCTURED``. See https://console.groq.com/docs/structured-outputs
    return os.getenv("GROQ_MODEL_STRUCTURED", "openai/gpt-oss-20b")


class CountryCapitals(BaseModel):
    """Two country/capital pairs."""

    first_country: str = Field(description="First country name")
    first_capital: str = Field(description="Capital of first country")
    second_country: str = Field(description="Second country name")
    second_capital: str = Field(description="Capital of second country")


async def main() -> None:
    llm = BaseGroq(model_name=_model(), async_mode=True)

    agent = Agent(
        name="groq-structured",
        prompt=ZeroShotPrompt().configure(
            system="Answer exactly in the required JSON shape. No extra keys.",
        ),
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            llm_params=GroqLLMParams(temperature=0.1, max_output_tokens=512),
        ),
        response_format=CountryCapitals,
    )

    await agent.initialize()

    result = await agent.execute(
        Task(
            id="groq-structured-1",
            objective=(
                "Fill the schema for France/Paris and Japan/Tokyo (use those exact pairs)."
            ),
        )
    )
    print("=== DIRECT + structured output ===")
    print(result.output)
    out = result.output
    if isinstance(out, CountryCapitals):
        print("\nValidated model:")
        print(out.model_dump_json(indent=2))
    elif isinstance(out, dict) and "output" in out:
        print("\nWrapped output:", out["output"])


if __name__ == "__main__":
    asyncio.run(main())
