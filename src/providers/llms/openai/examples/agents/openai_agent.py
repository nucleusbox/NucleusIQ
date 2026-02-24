"""
Example: Full Agent Example with Tools

This example demonstrates a complete agent setup with multiple tools.
The LLM provider (OpenAI) is an internal detail - you just create Agents!

Run with: python examples/agents/openai_agent.py

Requires OPENAI_API_KEY environment variable.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import BaseTool
from nucleusiq_openai import BaseOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    """Run an example agent with OpenAI."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in your .env file or environment.")
        return

    print("=" * 60)
    print("NucleusIQ Full Agent Example")
    print("=" * 60)
    print("\nRemember: You create Agents, not LLM calls!")
    print("   The LLM provider is just a configuration detail.\n")

    # Step 1: Create LLM provider (internal detail)
    print("1. Creating LLM provider (internal)...")
    llm = BaseOpenAI(
        model_name="gpt-5-nano",
        temperature=0.7,
        max_retries=3,
    )
    print("[OK] LLM provider created")

    # Create prompt
    print("\n2. Creating prompt...")
    prompt = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT).configure(
        system="You are a helpful assistant that can perform calculations.",
        user="Help the user with their request. Use tools when appropriate.",
    )
    print("[OK] Prompt created")

    # Create a tool
    print("\n3. Creating tools...")

    def add(a: int, b: int) -> int:
        """Add two integers together."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two integers together."""
        return a * b

    adder = BaseTool.from_function(add, description="Add two integers.")
    multiplier = BaseTool.from_function(multiply, description="Multiply two integers.")
    print("[OK] Tools created: add, multiply")

    # Step 4: Create Agent (THIS IS THE PRIMARY INTERFACE)
    print("\n4. Creating Agent (primary interface)...")
    agent = Agent(
        name="MathBot",
        role="Math Assistant",
        objective="Help users with mathematical calculations.",
        narrative="MathBot can add and multiply numbers using tools.",
        llm=llm,  # LLM is just a parameter
        prompt=prompt,
        tools=[adder, multiplier],
        config=AgentConfig(verbose=True),
    )
    print("[OK] Agent created")

    # Initialize agent
    print("\n5. Initializing agent...")
    await agent.initialize()
    print(f"[OK] Agent initialized (state: {agent.state})")

    # Test 1: Simple addition
    print("\n" + "=" * 60)
    print("Test 1: Addition Task")
    print("=" * 60)
    task1 = {"id": "task1", "objective": "What is 15 + 27?"}
    try:
        result1 = await agent.execute(task1)
        print(f"\n[OK] Result: {result1}")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")

    # Test 2: Multiplication
    print("\n" + "=" * 60)
    print("Test 2: Multiplication Task")
    print("=" * 60)
    task2 = {"id": "task2", "objective": "What is 7 times 8?"}
    try:
        result2 = await agent.execute(task2)
        print(f"\n[OK] Result: {result2}")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")

    # Test 3: Complex calculation
    print("\n" + "=" * 60)
    print("Test 3: Complex Calculation")
    print("=" * 60)
    task3 = {"id": "task3", "objective": "Calculate (5 + 3) * 4"}
    try:
        result3 = await agent.execute(task3)
        print(f"\n[OK] Result: {result3}")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")

    # Test 4: Non-math question (should use LLM directly)
    print("\n" + "=" * 60)
    print("Test 4: Non-Math Question")
    print("=" * 60)
    task4 = {"id": "task4", "objective": "What is the capital of France?"}
    try:
        result4 = await agent.execute(task4)
        print(f"\n[OK] Result: {result4}")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nKey takeaway: You work with Agents, not LLM APIs!")
    print("   The LLM provider is an implementation detail.")


if __name__ == "__main__":
    asyncio.run(main())
