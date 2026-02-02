"""
OpenAI Quick Start Guide

A simple, quick-start example showing the most common OpenAI integration patterns.
Perfect for getting started with NucleusIQ and OpenAI.

Run with: python src/examples/agents/openai_quick_start.py

Requires OPENAI_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
from pydantic import BaseModel, Field

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
from nucleusiq.core.tools import BaseTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Quick Start Examples
# ============================================================================

async def quick_start_1_simple_chat():
    """Quick Start 1: Simple chat agent (DIRECT mode)."""
    logger.info("\n" + "=" * 60)
    logger.info("Quick Start 1: Simple Chat Agent")
    logger.info("=" * 60)
    
    # Step 1: Create OpenAI LLM
    llm = BaseOpenAI(model_name="gpt-4o-mini")
    
    # Step 2: Create agent
    agent = Agent(
        name="ChatBot",
        role="Assistant",
        objective="Answer questions helpfully",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT)
    )
    
    # Step 3: Initialize
    await agent.initialize()
    
    # Step 4: Execute
    result = await agent.execute({
        "id": "1",
        "objective": "What is the capital of France?"
    })
    
    logger.info(f"Answer: {result}\n")


async def quick_start_2_with_tools():
    """Quick Start 2: Agent with tools (STANDARD mode)."""
    logger.info("\n" + "=" * 60)
    logger.info("Quick Start 2: Agent with Tools")
    logger.info("=" * 60)
    
    # Create tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    tool = BaseTool.from_function(add)
    
    # Create LLM and agent
    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="Calculator",
        role="Calculator",
        objective="Perform calculations",
        llm=llm,
        tools=[tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD)
    )
    
    await agent.initialize()
    
    result = await agent.execute({
        "id": "2",
        "objective": "What is 15 + 27?"
    })
    
    logger.info(f"Result: {result}\n")


async def quick_start_3_structured_output():
    """Quick Start 3: Structured output extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("Quick Start 3: Structured Output")
    logger.info("=" * 60)
    
    # Define schema
    class Person(BaseModel):
        name: str
        age: int
        city: str
    
    # Create agent with structured output
    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="Extractor",
        role="Data Extractor",
        objective="Extract structured data",
        llm=llm,
        response_format=Person,  # Just pass the schema!
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT)
    )
    
    await agent.initialize()
    
    result = await agent.execute({
        "id": "3",
        "objective": "Extract: John, 30, New York"
    })
    
    if isinstance(result, dict) and "output" in result:
        person = result["output"]
        logger.info(f"Extracted: {person.name}, {person.age}, {person.city}\n")


async def quick_start_4_autonomous_planning():
    """Quick Start 4: Autonomous mode with planning."""
    logger.info("\n" + "=" * 60)
    logger.info("Quick Start 4: Autonomous Planning")
    logger.info("=" * 60)
    
    # Create tools
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    tool = BaseTool.from_function(multiply)
    
    # Create agent in AUTONOMOUS mode
    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="Planner",
        role="Planning Assistant",
        objective="Plan and execute multi-step tasks",
        llm=llm,
        tools=[tool],
        config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
    )
    
    await agent.initialize()
    
    result = await agent.execute({
        "id": "4",
        "objective": "Calculate 5 * 3, then multiply that result by 2"
    })
    
    logger.info(f"Final Result: {result}\n")


async def main():
    """Run all quick start examples."""
    logger.info("=" * 60)
    logger.info("NucleusIQ OpenAI Quick Start")
    logger.info("=" * 60)
    logger.info("\nThese examples show the most common patterns:")
    logger.info("  1. Simple chat (DIRECT mode)")
    logger.info("  2. Tools (STANDARD mode)")
    logger.info("  3. Structured output")
    logger.info("  4. Autonomous planning")
    logger.info("\n" + "=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        logger.error("Please set it in your .env file or environment.")
        return
    
    examples = [
        ("Simple Chat", quick_start_1_simple_chat),
        ("With Tools", quick_start_2_with_tools),
        ("Structured Output", quick_start_3_structured_output),
        ("Autonomous Planning", quick_start_4_autonomous_planning),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
            logger.info(f"✅ {name} completed\n")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}\n", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("Quick start examples completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

