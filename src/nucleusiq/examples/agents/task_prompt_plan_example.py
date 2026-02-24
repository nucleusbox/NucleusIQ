# src/examples/agents/task_prompt_plan_example.py
"""
Example: Task, Prompt, and Execution Modes

This example demonstrates how Task and Prompt work together across modes:
- Task: User's request (what to do)
- Prompt: Agent's instructions (how to behave)
- Execution Mode: How to execute (Direct, Standard, Autonomous)
"""

import asyncio
import logging
import os
import sys

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from typing import Any, Dict

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import BaseTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """Simple calculator tool."""

    def __init__(self):
        super().__init__(name="add", description="Add two numbers together")

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
            },
        }


async def example_1_direct_mode():
    """Example 1: Direct mode — fast, simple."""
    logger.info("=" * 60)
    logger.info("Example 1: Direct Mode (Fast, Simple)")
    logger.info("=" * 60)

    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
    prompt.configure(
        system="You are a helpful assistant that performs calculations.",
        user="Answer questions accurately.",
    )

    agent = Agent(
        name="DirectAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent",
        llm=MockLLM(),
        prompt=prompt,
        tools=[CalculatorTool()],
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=True),
    )

    await agent.initialize()

    task = {"id": "task1", "objective": "What is 5 + 3?"}

    logger.info(f"\nTask: {task['objective']}")
    logger.info("Mode: DIRECT (max 5 tool calls)")
    logger.info("\nExecuting...")

    result = await agent.execute(task)
    logger.info(f"\nResult: {result}")


async def example_2_standard_mode():
    """Example 2: Standard mode — tool-driven workflows."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Standard Mode (Tool-Driven)")
    logger.info("=" * 60)

    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
    prompt.configure(
        system="You are a helpful assistant that performs calculations.",
        user="Answer questions accurately and show your work.",
    )

    agent = Agent(
        name="StandardAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent with tool support",
        llm=MockLLM(),
        prompt=prompt,
        tools=[CalculatorTool()],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=True),
    )

    await agent.initialize()

    task = {"id": "task2", "objective": "What is 15 + 27?"}

    logger.info(f"\nTask: {task['objective']}")
    logger.info("Mode: STANDARD (max 30 tool calls)")
    logger.info("\nExecuting...")

    result = await agent.execute(task)
    logger.info(f"\nResult: {result}")


async def example_3_manual_plan():
    """Example 3: Manual plan creation and execution."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Manual Plan Creation")
    logger.info("=" * 60)

    agent = Agent(
        name="ManualPlanAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent",
        llm=MockLLM(),
        tools=[CalculatorTool()],
        config=AgentConfig(verbose=True),
    )

    await agent.initialize()

    task = {"id": "task3", "objective": "Calculate 10 + 20"}

    logger.info(f"\nTask: {task['objective']}")
    logger.info("Creating plan manually...")

    plan = await agent.plan(task)
    logger.info(f"Plan created: {len(plan)} step(s)")
    for step in plan.steps:
        logger.info(f"  Step {step.step}: {step.action}")

    logger.info("\nExecuting task...")
    result = await agent.execute(task)
    logger.info(f"\nResult: {result}")


async def example_4_autonomous_mode():
    """Example 4: Autonomous mode — orchestration + verification."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Autonomous Mode (Orchestration + Verification)")
    logger.info("=" * 60)

    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
    prompt.configure(
        system="You are a helpful assistant that performs calculations.",
        user="Answer questions accurately and show your work step by step.",
    )

    agent = Agent(
        name="AutonomousAgent",
        role="Calculator",
        objective="Perform calculations",
        narrative="A calculator agent with full features",
        llm=MockLLM(),
        prompt=prompt,
        tools=[CalculatorTool()],
        config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, verbose=True),
    )

    await agent.initialize()

    task = {"id": "task4", "objective": "What is 25 + 35?"}

    logger.info(f"\nTask: {task['objective']}")
    logger.info("Mode: AUTONOMOUS (max 100 tool calls, Critic + Refiner)")
    logger.info("\nExecuting...")

    result = await agent.execute(task)
    logger.info(f"\nResult: {result}")

    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info("=" * 60)
    logger.info("Task: User's request (what to do)")
    logger.info("Prompt: Agent's instructions (how to behave)")
    logger.info("Mode: Execution strategy (Direct, Standard, Autonomous)")


async def main():
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("Task, Prompt, and Execution Mode Examples")
    logger.info("=" * 60)

    await example_1_direct_mode()
    await example_2_standard_mode()
    await example_3_manual_plan()
    await example_4_autonomous_mode()

    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)
    logger.info("\nKey Takeaways:")
    logger.info("   - Task = What the user wants (required)")
    logger.info("   - Prompt = How the agent behaves (optional but recommended)")
    logger.info("   - Mode = DIRECT (fast), STANDARD (tools), AUTONOMOUS (verify)")


if __name__ == "__main__":
    asyncio.run(main())
