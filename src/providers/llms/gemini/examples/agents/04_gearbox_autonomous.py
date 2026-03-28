"""Gearbox Gear 3: AUTONOMOUS mode with Gemini.

Autonomous mode is the most powerful gear — it adds task decomposition,
a Critic component for quality verification, and a Refiner for iterative
improvement. Best for complex, multi-step tasks.

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.tools.decorators import tool
from nucleusiq_gemini import BaseGemini, GeminiLLMParams


@tool
def search_database(query: str) -> str:
    """Search a knowledge database for information.

    Args:
        query: Search query string.
    """
    knowledge = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "rust": "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "javascript": "JavaScript is a dynamic scripting language primarily used for web development.",
        "ai": "AI (Artificial Intelligence) refers to systems that can perform tasks typically requiring human intelligence.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g. '2 + 2').
    """
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")

    config = AgentConfig(
        execution_mode=ExecutionMode.AUTONOMOUS,
        require_quality_check=True,
        max_iterations=5,
        llm_max_output_tokens=2048,
        llm_params=GeminiLLMParams(
            temperature=0.7,
            max_output_tokens=2048,
        ),
    )

    agent = Agent(
        llm=llm,
        config=config,
        name="gemini-autonomous",
        instructions=(
            "You are an expert research assistant. Break complex questions "
            "into sub-tasks, use tools to gather data, and synthesize a "
            "comprehensive answer. Always verify your findings."
        ),
        model="gemini-2.5-flash",
        tools=[search_database, calculate],
    )

    result = await agent.execute(
        "Compare Python and Rust for building AI applications. "
        "Consider performance, ecosystem, and ease of use."
    )
    print("=== AUTONOMOUS MODE ===")
    print(result.content)

    usage = agent.last_usage
    print(f"\nTokens: {usage.total.total_tokens}")
    print(f"LLM calls: {usage.total.calls}")
    print(f"By purpose: {list(usage.by_purpose.keys())}")
    print(f"By origin: {list(usage.by_origin.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
