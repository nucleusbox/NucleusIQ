"""Gearbox Gear 1: DIRECT mode with Gemini.

Direct mode is the fastest execution path — a single LLM call with no
tool loop. Ideal for quick Q&A, classification, and summarization tasks
where tool usage is unnecessary.

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_gemini import BaseGemini, GeminiLLMParams


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")

    config = AgentConfig(
        execution_mode=ExecutionMode.DIRECT,
        llm_params=GeminiLLMParams(
            temperature=0.3,
            max_output_tokens=512,
        ),
    )

    agent = Agent(
        name="gemini-direct",
        prompt=ZeroShotPrompt().configure(
            system="You are a concise assistant. Answer in 2-3 sentences.",
        ),
        llm=llm,
        config=config,
    )

    result = await agent.execute("What is the capital of France?")
    print("=== DIRECT MODE ===")
    print(result.content)
    print(f"\nTokens used: {agent.last_usage.total.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
