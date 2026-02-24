"""
Example: Autonomous Mode — Generate → Verify → Revise Loop

Demonstrates NucleusIQ's autonomous execution with self-correction:

    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ GENERATE │ ──→ │  VERIFY  │ ──→ │  REVISE  │
    │ (Plan +  │     │ (Critic) │     │(Refiner) │
    │ Execute) │     │          │     │          │
    └──────────┘     └────┬─────┘     └────┬─────┘
                     PASS │                │
                       ↓  │     ←──────────┘
                     Done └── FAIL → loop back

Architecture (Generate → Verify → Revise):
  - Generator: Proposes candidate solutions via tool-enabled execution
  - Verifier (Critic): Reviews results independently with fresh context
  - Reviser (Refiner): Makes directed corrections using Critic's feedback

Run with: python src/nucleusiq/examples/agents/autonomous_mode_example.py
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.base_tool import BaseTool

logging.basicConfig(
    level=logging.INFO,
    format="%(name)-20s %(levelname)-7s %(message)s",
)
logger = logging.getLogger("autonomous_demo")


# ------------------------------------------------------------------ #
# Tools                                                               #
# ------------------------------------------------------------------ #


class AddTool(BaseTool):
    def __init__(self):
        super().__init__(name="add", description="Add two numbers")

    async def initialize(self) -> None:
        pass

    async def execute(self, a: int, b: int) -> int:
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


class MultiplyTool(BaseTool):
    def __init__(self):
        super().__init__(name="multiply", description="Multiply two numbers")

    async def initialize(self) -> None:
        pass

    async def execute(self, a: int, b: int) -> int:
        return a * b

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


# ------------------------------------------------------------------ #
# Demo runner                                                         #
# ------------------------------------------------------------------ #


async def main():
    logger.info("=" * 60)
    logger.info("NucleusIQ Autonomous Mode (Generate → Verify → Revise)")
    logger.info("=" * 60)

    llm = MockLLM()
    tools = [AddTool(), MultiplyTool()]

    config = AgentConfig(
        execution_mode=ExecutionMode.AUTONOMOUS,
        critique_rounds=3,  # max Generate→Verify→Revise cycles
        require_quality_check=True,  # enable quality gate
        verbose=True,
    )

    agent = Agent(
        name="MathAgent",
        role="Calculator",
        objective="Solve multi-step math problems with self-correction",
        llm=llm,
        tools=tools,
        config=config,
    )
    await agent.initialize()

    logger.info("")
    logger.info("Agent: %s", agent.name)
    logger.info("Mode:  AUTONOMOUS (Generate -> Verify -> Revise)")
    logger.info("Config:")
    logger.info("  critique_rounds:       %d", config.critique_rounds)
    logger.info("  require_quality_check: %s", config.require_quality_check)
    logger.info("")

    task = Task(id="math-1", objective="Calculate (5 + 3) * 2")
    logger.info("Task: %s", task.objective)
    logger.info("-" * 60)

    result = await agent.execute(task)

    logger.info("-" * 60)
    logger.info("Result: %s", result)
    logger.info("State:  %s", agent.state)
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    asyncio.run(main())
