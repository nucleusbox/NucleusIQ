"""
Example: Math Agent with Tools

This example demonstrates how to create an agent that uses tools for calculations.
The LLM provider (OpenAI) is an internal detail - you just create Agents!

Run with: python examples/agents/math_agent.py

Requires OPENAI_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
from nucleusiq.core.tools import BaseTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run a math agent example with tools."""
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.error("Please set it in your .env file or environment.")
        return
    
    logger.info("=" * 60)
    logger.info("NucleusIQ Math Agent Example")
    logger.info("=" * 60)
    logger.info("\n💡 Remember: You create Agents, not LLM calls!")
    logger.info("   The LLM provider is just a configuration detail.\n")
    
    # Step 1: Create LLM provider (internal detail)
    logger.info("1. Creating LLM provider (internal)...")
    llm = BaseOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_retries=3,
    )
    logger.info("✅ LLM provider created")
    
    # Step 2: Create prompt
    logger.info("\n2. Creating prompt...")
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful math assistant that can perform calculations.",
        user="Help the user with their mathematical questions. Use tools when appropriate."
    )
    logger.info("✅ Prompt created")
    
    # Step 3: Create tools
    logger.info("\n3. Creating tools...")
    
    def add(a: int, b: int) -> int:
        """Add two integers together."""
        return a + b
    
    def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
    
    def multiply(a: int, b: int) -> int:
        """Multiply two integers together."""
        return a * b
    
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    adder = BaseTool.from_function(add, description="Add two integers.")
    subtractor = BaseTool.from_function(subtract, description="Subtract two integers.")
    multiplier = BaseTool.from_function(multiply, description="Multiply two integers.")
    divider = BaseTool.from_function(divide, description="Divide two numbers.")
    
    tools = [adder, subtractor, multiplier, divider]
    logger.info(f"✅ Tools created: {[t.name for t in tools]}")
    
    # Step 4: Create Agent (THIS IS THE PRIMARY INTERFACE)
    logger.info("\n4. Creating Agent (primary interface)...")
    agent = Agent(
        name="MathBot",
        role="Math Assistant",
        objective="Help users with mathematical calculations.",
        narrative="MathBot can perform basic arithmetic operations using tools.",
        llm=llm,  # LLM is just a parameter
        prompt=prompt,
        tools=tools,
        config=AgentConfig(verbose=True)
    )
    logger.info("✅ Agent created")
    
    # Step 5: Initialize agent
    logger.info("\n5. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")
    
    # Step 6: Use the Agent (this is what users do)
    logger.info("\n6. Using Agent to execute tasks...")
    tasks = [
        {"id": "task1", "objective": "What is 15 + 27?"},
        {"id": "task2", "objective": "What is 7 times 8?"},
        {"id": "task3", "objective": "Calculate (5 + 3) * 4"},
        {"id": "task4", "objective": "What is 100 divided by 4?"},
        {"id": "task5", "objective": "What is 50 - 23?"},
    ]
    
    for i, task in enumerate(tasks, 1):
        logger.info("\n" + "=" * 60)
        logger.info(f"Task {i}: {task['objective']}")
        logger.info("=" * 60)
        
        try:
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

