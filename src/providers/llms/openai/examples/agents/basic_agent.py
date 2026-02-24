"""
Example: Basic Agent

This example demonstrates how to create and use a basic agent.
The LLM provider (OpenAI) is an internal detail - you just create Agents!

Run with: python examples/agents/basic_agent.py

Requires OPENAI_API_KEY environment variable.
"""

import asyncio
import logging
import os
import sys

# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq_openai import BaseOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """
    Main example: Creating and using an Agent.

    Key points:
    1. Create an LLM provider (internal detail)
    2. Create an Agent (primary interface)
    3. Use the Agent to execute tasks
    """

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.error("Please set it in your .env file or environment.")
        return

    logger.info("=" * 60)
    logger.info("NucleusIQ Basic Agent Example")
    logger.info("=" * 60)
    logger.info("\n💡 Remember: You create Agents, not LLM calls!")
    logger.info("   The LLM provider is just a configuration detail.\n")

    # Step 1: Create LLM provider (internal detail, not exposed to users)
    logger.info("1. Creating LLM provider (internal)...")
    llm = BaseOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_retries=3,
    )
    logger.info("✅ LLM provider created")

    # Step 2: Create prompt
    logger.info("\n2. Creating prompt...")
    prompt = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT).configure(
        system="You are a helpful assistant.",
        user="Answer the user's question clearly and concisely.",
    )
    logger.info("✅ Prompt created")

    # Step 3: Create Agent (THIS IS THE PRIMARY INTERFACE)
    logger.info("\n3. Creating Agent (primary interface)...")
    agent = Agent(
        name="BasicBot",
        role="Assistant",
        objective="Help users with their questions.",
        narrative="BasicBot is a helpful assistant that answers questions.",
        llm=llm,  # LLM is just a parameter
        prompt=prompt,
        config=AgentConfig(verbose=True),
    )
    logger.info("✅ Agent created")

    # Step 4: Initialize agent
    logger.info("\n4. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")

    # Step 5: Use the Agent (this is what users do)
    logger.info("\n5. Using Agent to execute tasks...")
    tasks = [
        {"id": "task1", "objective": "What is the capital of France?"},
        {"id": "task2", "objective": "Explain what Python is in one sentence."},
        {"id": "task3", "objective": "What is 2 + 2?"},
    ]

    for i, task in enumerate(tasks, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"Task {i}: {task['objective']}")
        logger.info("=" * 60)

        try:
            # Execute task using Agent (primary interface)
            result = await agent.execute(task)
            logger.info(f"\n✅ Result: {result}")
        except Exception as e:
            logger.error(f"\n❌ Error: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Example completed!")
    logger.info("=" * 60)
    logger.info("\n💡 Key takeaway: You work with Agents, not LLM APIs!")
    logger.info("   The LLM provider is an implementation detail.")


if __name__ == "__main__":
    asyncio.run(main())
