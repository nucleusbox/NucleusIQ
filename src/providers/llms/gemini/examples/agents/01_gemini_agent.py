"""Using Gemini with NucleusIQ Agent framework.

Demonstrates how BaseGemini plugs into the Agent + AgentConfig
pattern — the same way BaseOpenAI does.

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq_gemini import BaseGemini, GeminiLLMParams


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")

    config = AgentConfig(
        name="gemini-assistant",
        instructions="You are a helpful AI assistant powered by Google Gemini.",
        model="gemini-2.5-flash",
        llm_params=GeminiLLMParams(
            temperature=0.5,
            top_k=40,
            max_output_tokens=1024,
        ),
    )

    agent = Agent(llm=llm, config=config)

    result = await agent.execute("Explain quantum computing in simple terms.")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
