"""Reasoning ("thinking") models on a self-hosted server.

Three things have to line up, and getting any one wrong fails quietly:

1. **The server must separate thinking from the answer.**  Start it with
   ``--reasoning-parser <parser>``; otherwise the thinking text arrives
   inline in ``content`` and the model looks like it is rambling.

2. **The chat template's thinking switch must be set.**  Defaults differ by
   family — off for Gemma, Granite and DeepSeek-V3.1, on for Qwen3 — so pass
   it explicitly via ``chat_template_kwargs``.

3. **``is_reasoning_model=True`` must be declared.**  It is never guessed
   from the model name, because a served name is arbitrary on a self-hosted
   deployment.  The framework widens the token budget for internal calls
   (Critic, Refiner, Decomposer) when it is set; left off, those calls get
   truncated mid-thought and return nothing usable.

Server:
    vllm serve <model> --reasoning-parser deepseek_r1 \
      --enable-auto-tool-choice --tool-call-parser hermes

Run:
    python examples/agents/06_thinking_model.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq_openai_compatible import OpenAICompatibleLLM

# The provider logs actionable warnings when thinking is misconfigured.
logging.basicConfig(level=logging.WARNING)

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gemma-4-27b-it")


def build_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        base_url=BASE_URL,
        model=MODEL,
        engine="vllm",
        context_window=32_768,
        is_reasoning_model=True,
        # Gemma and Qwen3 read enable_thinking; Granite and DeepSeek read
        # thinking. Sent on every request because the template is applied
        # server-side at request time.
        chat_template_kwargs={"enable_thinking": True},
        # Thinking and the answer share one completion budget, so give it
        # room — a thinking model capped at 150 tokens returns nothing.
        max_output_tokens=2_048,
    )


async def direct_call() -> None:
    print("=== Direct call: thinking kept separate ===")
    llm = build_llm()
    response = await llm.call(
        messages=[
            {
                "role": "user",
                "content": "A farmer has 17 sheep. All but 9 run away. "
                "How many are left? Think it through.",
            }
        ],
        max_output_tokens=1_024,
        # vLLM maps reasoning_effort onto the template's thinking switch.
        reasoning_effort="high",
    )

    if response.has_reasoning:
        print(f"--- thinking ({len(response.reasoning)} chars) ---")
        print(response.reasoning[:400])
    else:
        print("(no separated thinking — is --reasoning-parser set?)")

    print("\n--- answer ---")
    print(response.content)

    if response.reasoning_tokens:
        print(f"\nreasoning tokens: {response.reasoning_tokens}")

    if response.reasoning_only:
        # vLLM #53284: template and parser disagree, so the whole answer
        # lands in the reasoning field and content comes back null.
        print("\nWARNING: all output arrived as thinking. Toggle mismatch.")


async def streamed_call() -> None:
    print("\n=== Streamed: thinking tagged, not merged ===")
    llm = build_llm()
    async for event in llm.call_stream(
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        max_output_tokens=800,
    ):
        if event.type != "token":
            continue
        # Thinking deltas are tagged so a UI can route them to a separate
        # pane, and a plain consumer can filter them with one check.
        if (event.metadata or {}).get("reasoning"):
            print(f"\033[90m{event.token}\033[0m", end="", flush=True)
        else:
            print(event.token, end="", flush=True)
    print()


async def through_an_agent() -> None:
    print("\n=== Through an agent ===")
    agent = Agent(
        name="Reasoner",
        role="Analyst",
        objective="Solve problems carefully",
        prompt=PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
            system="You are a careful reasoner.",
        ),
        llm=build_llm(),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
    )
    await agent.initialize()

    print(f"is_reasoning_model: {agent.llm.is_reasoning_model}")
    result = await agent.execute(
        Task(
            id="r1",
            objective="If a bat and ball cost $1.10 and the bat "
            "costs $1 more than the ball, what does the ball cost?",
        )
    )
    # The agent's output is the answer only; thinking never leaks into it or
    # into the next turn's history.
    print(result.output)


async def main() -> None:
    await direct_call()
    await streamed_call()
    await through_an_agent()


if __name__ == "__main__":
    asyncio.run(main())
