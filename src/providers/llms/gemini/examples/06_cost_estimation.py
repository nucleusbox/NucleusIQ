"""Cost estimation with Gemini provider.

Demonstrates how to use the CostTracker alongside the Agent's UsageTracker
to estimate API costs after execution.

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.usage.pricing import CostTracker
from nucleusiq_gemini import BaseGemini, GeminiLLMParams


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")
    model = "gemini-2.5-flash"

    config = AgentConfig(
        execution_mode=ExecutionMode.STANDARD,
        llm_params=GeminiLLMParams(
            temperature=0.5,
            max_output_tokens=512,
        ),
    )

    agent = Agent(
        llm=llm,
        config=config,
        name="gemini-cost-demo",
        instructions="You are a helpful assistant.",
        model=model,
    )

    result = await agent.execute("Explain the theory of relativity in 3 sentences.")
    print("=== Response ===")
    print(result.content)

    usage = agent.last_usage
    print("\n=== Usage ===")
    print(f"Prompt tokens:     {usage.total.prompt_tokens}")
    print(f"Completion tokens: {usage.total.completion_tokens}")
    print(f"Total tokens:      {usage.total.total_tokens}")
    print(f"LLM calls:         {usage.total.calls}")

    tracker = CostTracker()
    cost = tracker.estimate(usage, model=model)
    print("\n=== Cost Estimate ===")
    print(f"Prompt cost:     ${cost.prompt_cost:.6f}")
    print(f"Completion cost: ${cost.completion_cost:.6f}")
    print(f"Total cost:      ${cost.total_cost:.6f}")

    if cost.by_purpose:
        print("\nBy purpose:")
        for purpose, pc in cost.by_purpose.items():
            print(
                f"  {purpose}: ${pc.total_cost:.6f} ({pc.tokens} tokens, {pc.calls} calls)"
            )


if __name__ == "__main__":
    asyncio.run(main())
