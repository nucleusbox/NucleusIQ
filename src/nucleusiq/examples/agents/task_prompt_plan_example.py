# src/examples/agents/task_prompt_plan_example.py
"""
Example: Task, Prompt, and Plan Relationship

This example demonstrates how Task, Prompt, and Plan work together:
- Task: User's request (what to do)
- Prompt: Agent's instructions (how to behave)
- Plan: Task decomposition (how to break down) - Optional
"""

import asyncio
import logging
import os
import sys

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import BaseTool
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """Simple calculator tool."""
    
    def __init__(self):
        super().__init__(
            name="add",
            description="Add two numbers together"
        )
    
    async def initialize(self) -> None:
        pass
    
    async def execute(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            }
        }


async def example_1_simple_execution():
    """Example 1: Simple execution without planning."""
    logger.info("=" * 60)
    logger.info("Example 1: Simple Execution (No Plan)")
    logger.info("=" * 60)
    
    # Create prompt
    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
    prompt.configure(
        system="You are a helpful assistant that performs calculations.",
        user="Answer questions accurately."
    )
    
    # Create agent WITHOUT planning
    agent = Agent(
        name="SimpleAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent",
        llm=MockLLM(),
        prompt=prompt,
        tools=[CalculatorTool()],
        config=AgentConfig(use_planning=False, verbose=True)
    )
    
    await agent.initialize()
    
    # Create task
    task = {"id": "task1", "objective": "What is 5 + 3?"}
    
    logger.info(f"\nTask: {task['objective']}")
    logger.info("Prompt: System + User template")
    logger.info("Plan: None (use_planning=False)")
    logger.info("\nExecuting...")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Result: {result}")


async def example_2_with_planning():
    """Example 2: Execution with planning enabled."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Execution with Planning")
    logger.info("=" * 60)
    
    # Create prompt
    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
    prompt.configure(
        system="You are a helpful assistant that performs calculations.",
        user="Answer questions accurately and show your work."
    )
    
    # Create agent WITH planning enabled
    agent = Agent(
        name="PlanningAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent with planning",
        llm=MockLLM(),
        prompt=prompt,
        tools=[CalculatorTool()],
        config=AgentConfig(use_planning=True, verbose=True)
    )
    
    await agent.initialize()
    
    # Create task
    task = {"id": "task2", "objective": "What is 15 + 27?"}
    
    logger.info(f"\nTask: {task['objective']}")
    logger.info("Prompt: System + User template")
    logger.info("Plan: Enabled (use_planning=True)")
    logger.info("\nExecuting...")
    
    # Plan will be created automatically
    result = await agent.execute(task)
    logger.info(f"\n✅ Result: {result}")


async def example_3_manual_plan():
    """Example 3: Manual plan creation and execution."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Manual Plan Creation")
    logger.info("=" * 60)
    
    # Create agent
    agent = Agent(
        name="ManualPlanAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent",
        llm=MockLLM(),
        tools=[CalculatorTool()],
        config=AgentConfig(verbose=True)
    )
    
    await agent.initialize()
    
    # Create task
    task = {"id": "task3", "objective": "Calculate 10 + 20"}
    
    # Create plan manually
    logger.info(f"\nTask: {task['objective']}")
    logger.info("Creating plan manually...")
    
    plan = await agent.plan(task)
    logger.info(f"Plan created: {len(plan)} step(s)")
    for step in plan.steps:
        logger.info(f"  Step {step.step}: {step.action}")
    
    # Execute with plan (use public API - execute() handles planning internally)
    logger.info("\nExecuting with plan...")
    # Note: execute() will use the plan if use_planning=True
    # For manual plan execution, we can still use execute() which will create its own plan
    # or we can enable planning and let execute() handle it
    agent.config.use_planning = True
    result = await agent.execute(task)
    logger.info(f"\n✅ Result: {result}")


async def example_4_all_three_together():
    """Example 4: Task, Prompt, and Plan all working together."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Task + Prompt + Plan Together")
    logger.info("=" * 60)
    
    # Create prompt
    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
    prompt.configure(
        system="You are a helpful assistant that performs calculations.",
        user="Answer questions accurately and show your work step by step."
    )
    
    # Create agent with planning
    agent = Agent(
        name="FullAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent with full features",
        llm=MockLLM(),
        prompt=prompt,
        tools=[CalculatorTool()],
        config=AgentConfig(use_planning=True, verbose=True)
    )
    
    await agent.initialize()
    
    # Create task
    task = {"id": "task4", "objective": "What is 25 + 35?"}
    
    logger.info(f"\nTask: {task['objective']}")
    logger.info("Prompt: System='You are a helpful assistant...'")
    logger.info("Prompt: User='Answer questions accurately...'")
    logger.info("Plan: Enabled (will be created automatically)")
    logger.info("\nExecuting...")
    
    result = await agent.execute(task)
    logger.info(f"\n✅ Result: {result}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info("=" * 60)
    logger.info("✅ Task: User's request (what to do)")
    logger.info("✅ Prompt: Agent's instructions (how to behave)")
    logger.info("✅ Plan: Task decomposition (how to break down)")
    logger.info("✅ All three work together in execute()!")


async def main():
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("Task, Prompt, and Plan Relationship Examples")
    logger.info("=" * 60)
    
    await example_1_simple_execution()
    await example_2_with_planning()
    await example_3_manual_plan()
    await example_4_all_three_together()
    
    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)
    logger.info("\n💡 Key Takeaways:")
    logger.info("   - Task = What the user wants (required)")
    logger.info("   - Prompt = How the agent behaves (optional but recommended)")
    logger.info("   - Plan = How to break down the task (optional, via config)")
    logger.info("   - All three work together in agent.execute()!")


if __name__ == "__main__":
    asyncio.run(main())

