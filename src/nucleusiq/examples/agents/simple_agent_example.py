"""
Example: Simple Agent with Tools

This example demonstrates how to create and use an Agent directly
(without subclassing) with MockLLM and tools, similar to echo_agent.py.

Run with: python src/examples/agents/simple_agent_example.py
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
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools.base_tool import BaseTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main example: Using Agent directly with MockLLM and tools."""

    logger.info("=" * 60)
    logger.info("NucleusIQ Simple Agent Example")
    logger.info("=" * 60)
    logger.info("\n💡 This example uses Agent directly (no subclassing)")
    logger.info("   Similar to echo_agent.py but simpler!\n")

    # Step 1: Create MockLLM
    logger.info("1. Creating MockLLM...")
    llm = MockLLM()
    logger.info("✅ MockLLM created")

    # Step 2: Create prompt
    logger.info("\n2. Creating prompt...")
    prompt = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT).configure(
        system="You are a helpful assistant.",
        user="Compute the sum of two numbers or repeat the request.",
    )
    logger.info("✅ Prompt created")

    # Step 3: Create a tool from a function
    logger.info("\n3. Creating tool from function...")

    def add(a: int, b: int) -> int:
        """Add two integers together."""
        return a + b

    adder_tool = BaseTool.from_function(add, description="Add two integers together.")
    logger.info(f"✅ Tool created: {adder_tool.name}")

    # Step 4: Create Agent directly (no subclassing!)
    logger.info("\n4. Creating Agent directly...")
    agent = Agent(
        name="SimpleBot",
        role="Calculator",
        objective="Help users with calculations.",
        narrative="SimpleBot can add numbers using tools.",
        llm=llm,
        prompt=prompt,
        tools=[adder_tool],
        config=AgentConfig(verbose=True),
    )
    logger.info("✅ Agent created")

    # Step 5: Initialize agent
    logger.info("\n5. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")

    # Step 6: Execute tasks
    logger.info("\n6. Executing tasks...")

    # Task 1: Function call task (should use the add tool)
    logger.info("\n" + "=" * 60)
    logger.info("Task 1: Add 7 and 8 (should use tool)")
    logger.info("=" * 60)
    task1 = {"id": "task1", "objective": "Add 7 and 8."}
    try:
        result1 = await agent.execute(task1)
        logger.info(f"\n✅ Result: {result1}")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Task 2: Simple echo task (no tool needed)
    logger.info("\n" + "=" * 60)
    logger.info("Task 2: Simple question (should echo)")
    logger.info("=" * 60)
    task2 = {"id": "task2", "objective": "What is the weather today?"}
    try:
        result2 = await agent.execute(task2)
        logger.info(f"\n✅ Result: {result2}")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Task 3: Another calculation
    logger.info("\n" + "=" * 60)
    logger.info("Task 3: Add 15 and 25 (should use tool)")
    logger.info("=" * 60)
    task3 = {"id": "task3", "objective": "Add 15 and 25."}
    try:
        result3 = await agent.execute(task3)
        logger.info(f"\n✅ Result: {result3}")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info("Example completed!")
    logger.info("=" * 60)
    logger.info("\n💡 Key takeaway: You can use Agent directly without subclassing!")
    logger.info("   The Agent class handles all the function-calling logic for you.")


if __name__ == "__main__":
    asyncio.run(main())
