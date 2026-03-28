"""Gearbox Gear 2: STANDARD mode with Gemini + tools.

Standard mode enables the tool-calling loop — the agent can call tools
iteratively (up to 30 calls by default) to gather information and
produce a final answer. This is the default mode.

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
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to look up weather for.
    """
    weather_data = {
        "paris": "22°C, Partly cloudy",
        "london": "15°C, Rainy",
        "tokyo": "28°C, Sunny",
        "new york": "18°C, Clear",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def get_population(city: str) -> str:
    """Get the population of a city.

    Args:
        city: The city name to look up population for.
    """
    population_data = {
        "paris": "2.1 million (city), 12.4 million (metro)",
        "london": "8.8 million (city), 14.4 million (metro)",
        "tokyo": "13.9 million (city), 37.4 million (metro)",
        "new york": "8.3 million (city), 20.1 million (metro)",
    }
    return population_data.get(
        city.lower(), f"Population data not available for {city}"
    )


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")

    config = AgentConfig(
        execution_mode=ExecutionMode.STANDARD,
        llm_params=GeminiLLMParams(
            temperature=0.5,
            max_output_tokens=1024,
        ),
    )

    agent = Agent(
        llm=llm,
        config=config,
        name="gemini-standard",
        instructions=(
            "You are a helpful travel assistant. Use available tools to "
            "gather real data before answering questions."
        ),
        model="gemini-2.5-flash",
        tools=[get_weather, get_population],
    )

    result = await agent.execute(
        "Tell me about Tokyo — what's the weather like and how many people live there?"
    )
    print("=== STANDARD MODE (with tools) ===")
    print(result.content)

    usage = agent.last_usage
    print(f"\nTokens: {usage.total.total_tokens}")
    print(f"LLM calls: {usage.total.calls}")
    print(f"By purpose: {list(usage.by_purpose.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
