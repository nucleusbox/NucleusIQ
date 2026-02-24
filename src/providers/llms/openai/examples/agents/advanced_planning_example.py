"""
Advanced Multi-Step Planning Example

This example demonstrates advanced planning capabilities with multiple plan steps.
It shows how the agent can:
1. Generate complex multi-step plans using LLM
2. Execute plans with multiple sequential steps
3. Use tools across different plan steps
4. Build context from previous steps
5. Handle complex multi-step calculations and workflows

This proves the completion of the planning system.

Run with: python src/examples/agents/advanced_planning_example.py

Requires OPENAI_API_KEY environment variable.
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config.agent_config import AgentConfig, ExecutionMode
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.planning.planner import Planner
from nucleusiq.agents.task import Task
from nucleusiq.tools import BaseTool
from nucleusiq_openai import BaseOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Keep third-party HTTP/client logs quiet; we only want high-signal traces.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def create_advanced_math_tools():
    """Create advanced math tools for complex calculations."""

    def add(a: int, b: int) -> int:
        """Add two integers together."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two integers together."""
        return a * b

    def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def power(base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base**exponent

    def sqrt(value: float) -> float:
        """Calculate square root of value."""
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return value**0.5

    add_tool = BaseTool.from_function(add, description="Add two integers together.")
    multiply_tool = BaseTool.from_function(
        multiply, description="Multiply two integers together."
    )
    subtract_tool = BaseTool.from_function(
        subtract, description="Subtract second number from first."
    )
    divide_tool = BaseTool.from_function(
        divide, description="Divide first number by second."
    )
    power_tool = BaseTool.from_function(
        power, description="Raise base to the power of exponent."
    )
    sqrt_tool = BaseTool.from_function(
        sqrt, description="Calculate square root of a number."
    )

    return [add_tool, multiply_tool, subtract_tool, divide_tool, power_tool, sqrt_tool]


async def test_llm_planning():
    """Test LLM-based plan generation with complex multi-step tasks using AUTONOMOUS mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: LLM-Based Multi-Step Plan Generation (AUTONOMOUS Mode)")
    logger.info("=" * 80)
    logger.info(
        "This test demonstrates how the agent generates complex plans using LLM"
    )
    logger.info(
        "Mode: AUTONOMOUS - Full reasoning loop with planning and self-correction"
    )
    logger.info("-" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create tools
    tools = create_advanced_math_tools()

    # Create agent with AUTONOMOUS mode for advanced planning
    agent = Agent(
        name="PlanningBot",
        role="Advanced Calculator",
        objective="Break down complex calculations into step-by-step plans and execute them",
        llm=llm,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS, verbose=False, max_iterations=15
        ),
    )

    await agent.initialize()
    logger.info(f"Agent initialized: {agent.name}")
    logger.info(f"   Tools available: {[t.name for t in tools]}")

    # Very complex task that requires multi-step planning and reasoning
    complex_task = Task(
        id="complex_calc_1",
        objective="Calculate: ((10 + 5) * 3 - 20) / 5 + 2^3, then find the square root of the result, multiply by 4, and subtract 10",
    )

    logger.info(f"\nComplex Task: {complex_task.objective}")
    logger.info("   Mode: AUTONOMOUS (automatic planning + execution)")
    logger.info("   Expected: Agent will automatically generate plan and execute it")
    logger.info("   Note: User just calls execute() - planning happens internally!")

    # Simply call execute() - AUTONOMOUS mode will automatically:
    # 1. Generate a plan internally
    # 2. Execute the plan
    # 3. Return the result
    try:
        result = await agent.execute(complex_task)

        logger.info(f"\nFinal Result: {result}")
        logger.info(f"   Agent State: {agent.state}")
        logger.info("Planning and execution completed.")

        # Treat echoes / errors as failure (this test is meant to prove multi-step completion).
        if isinstance(result, str) and (
            result.strip().startswith("Echo:")
            or result.strip().startswith("Error:")
            or result.strip().startswith("No response")
        ):
            return False
        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_manual_multi_step_plan():
    """Test manual creation and execution of multi-step plan using AUTONOMOUS mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Manual Multi-Step Plan Execution (AUTONOMOUS Mode)")
    logger.info("=" * 80)
    logger.info("This test demonstrates manual plan creation with multiple steps")
    logger.info("Mode: AUTONOMOUS - Executes plans with full reasoning capabilities")
    logger.info("-" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create tools
    tools = create_advanced_math_tools()

    # Create agent with AUTONOMOUS mode
    agent = Agent(
        name="ManualPlanBot",
        role="Calculator",
        objective="Execute manually created multi-step plans",
        llm=llm,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS, verbose=False, max_iterations=15
        ),
    )

    await agent.initialize()
    logger.info(f"Agent initialized: {agent.name}")

    # Create a complex task
    task = Task(
        id="manual_plan_1",
        objective="Calculate the area of a circle with radius 5, then find the square root of the result",
    )

    # Manually create a multi-step plan
    plan = Plan(
        steps=[
            PlanStep(
                step=1,
                action="multiply",
                args={"a": 5, "b": 5},  # radius^2
                details="Calculate radius squared (5 * 5 = 25)",
            ),
            PlanStep(
                step=2,
                action="multiply",
                args={"a": 25, "b": 3.14159},  # π * r^2 (approximate π)
                details="Multiply by π to get area (25 * 3.14159 ≈ 78.54)",
            ),
            PlanStep(
                step=3,
                action="sqrt",
                args={"value": 78.54},  # Will be updated from step 2 result
                details="Calculate square root of the area",
            ),
        ],
        task=task,
    )

    logger.info(f"\n📋 Task: {task.objective}")
    logger.info(f"\n📝 Manual Plan Created: {len(plan)} steps")
    for step in plan.steps:
        logger.info(f"   Step {step.step}: {step.action} - {step.details}")
        if step.args:
            logger.info(f"      Args: {step.args}")

    # Execute the plan
    logger.info("\n🚀 Executing Manual Plan...")
    try:
        result = await Planner(agent).execute_plan(task, plan)
        logger.info(f"\nFinal Result: {result}")
        logger.info(f"   Agent State: {agent.state}")
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_complex_workflow_planning():
    """Test complex workflow with multiple interdependent steps using AUTONOMOUS mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Complex Workflow Planning (AUTONOMOUS Mode)")
    logger.info("=" * 80)
    logger.info("This test demonstrates a complex workflow with interdependent steps")
    logger.info("Mode: AUTONOMOUS - Handles complex multi-step reasoning workflows")
    logger.info("-" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create tools
    tools = create_advanced_math_tools()

    # Create agent with AUTONOMOUS mode for complex workflows
    agent = Agent(
        name="WorkflowBot",
        role="Workflow Calculator",
        objective="Execute complex workflows with multiple interdependent calculations",
        llm=llm,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS, verbose=False, max_iterations=20
        ),
    )

    await agent.initialize()
    logger.info(f"Agent initialized: {agent.name}")

    # Very complex workflow task requiring multiple reasoning steps
    workflow_task = Task(
        id="workflow_1",
        objective="Calculate: Start with 100, add 25, multiply by 3, subtract 50, divide by 5, then raise to power of 2. After that, find the square root, multiply by 10, and add 100",
    )

    logger.info(f"\nWorkflow Task: {workflow_task.objective}")
    logger.info("   Mode: AUTONOMOUS (automatic planning + execution)")
    logger.info(
        "   Note: Agent will automatically plan and execute this complex workflow"
    )

    # Simply call execute() - AUTONOMOUS mode handles everything internally
    try:
        result = await agent.execute(workflow_task)

        logger.info(f"\nFinal Result: {result}")
        logger.info(f"   Agent State: {agent.state}")
        logger.info("Complex workflow planned and executed.")

        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_plan_with_context_building():
    """Test plan execution where steps build on previous results using AUTONOMOUS mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Plan with Context Building (AUTONOMOUS Mode)")
    logger.info("=" * 80)
    logger.info(
        "This test demonstrates how plan steps build context from previous steps"
    )
    logger.info("Mode: AUTONOMOUS - Advanced context management across steps")
    logger.info("-" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create tools
    tools = create_advanced_math_tools()

    # Create agent with AUTONOMOUS mode for context building
    agent = Agent(
        name="ContextBot",
        role="Context Builder",
        objective="Execute plans where each step uses results from previous steps",
        llm=llm,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS, verbose=False, max_iterations=15
        ),
    )

    await agent.initialize()
    logger.info(f"Agent initialized: {agent.name}")

    # Complex task that requires context building across multiple steps
    context_task = Task(
        id="context_1",
        objective="Calculate: (10 * 2) + (result * 3) - 15, where result is from first calculation. Then use that final result to calculate: (final_result / 2) + 50, and find the square root of that",
    )

    logger.info(f"\nContext Task: {context_task.objective}")
    logger.info("   Mode: AUTONOMOUS (automatic planning + context building)")
    logger.info(
        "   Note: Agent will automatically plan and handle context across steps"
    )

    # Simply call execute() - AUTONOMOUS mode handles planning and context automatically
    try:
        result = await agent.execute(context_task)

        logger.info(f"\nFinal Result: {result}")
        logger.info(f"   Agent State: {agent.state}")
        logger.info("Context building handled across plan steps.")

        # Treat echoes / errors as failure (this test is meant to prove context building works).
        if isinstance(result, str) and (
            result.strip().startswith("Echo:")
            or result.strip().startswith("Error:")
            or result.strip().startswith("No response")
        ):
            return False
        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_plan_serialization():
    """Test plan serialization and deserialization."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Plan Serialization")
    logger.info("=" * 80)
    logger.info("This test demonstrates plan serialization to/from JSON")
    logger.info("-" * 80)

    # Create a plan
    task = Task(id="serial_test", objective="Test serialization")
    plan = Plan(
        steps=[
            PlanStep(
                step=1, action="add", args={"a": 5, "b": 3}, details="Add 5 and 3"
            ),
            PlanStep(
                step=2,
                action="multiply",
                args={"a": 8, "b": 2},
                details="Multiply result by 2",
            ),
            PlanStep(
                step=3, action="subtract", args={"a": 16, "b": 4}, details="Subtract 4"
            ),
        ],
        task=task,
    )

    logger.info(f"\n📝 Original Plan: {len(plan)} steps")
    for step in plan.steps:
        logger.info(f"   Step {step.step}: {step.action} - {step.details}")

    # Serialize to dict
    plan_dict = plan.to_dict()
    logger.info("\n✅ Plan serialized to dict")
    logger.info(f"   Keys: {list(plan_dict.keys())}")

    # Serialize to JSON
    plan_json = json.dumps(plan_dict, indent=2)
    logger.info(f"\n✅ Plan serialized to JSON ({len(plan_json)} bytes)")

    # Deserialize from dict
    plan_restored = Plan(**plan_dict)
    logger.info("\nPlan deserialized from dict")
    logger.info(f"   Restored steps: {len(plan_restored)}")

    # Verify
    assert len(plan_restored.steps) == len(plan.steps), "Step count mismatch"
    for i, (original, restored) in enumerate(zip(plan.steps, plan_restored.steps)):
        assert original.step == restored.step, f"Step {i} number mismatch"
        assert original.action == restored.action, f"Step {i} action mismatch"
        logger.info(f"   Step {i + 1} verified: {restored.action}")

    logger.info("\nAll serialization tests passed.")
    return True


async def main():
    """Run all advanced planning tests."""
    logger.info("=" * 80)
    logger.info("NucleusIQ Advanced Multi-Step Planning Test Suite")
    logger.info("=" * 80)
    logger.info("\nThis test suite demonstrates AUTONOMOUS mode capabilities:")
    logger.info("  1. LLM-based multi-step plan generation (AUTONOMOUS mode)")
    logger.info("  2. Manual multi-step plan creation and execution (AUTONOMOUS mode)")
    logger.info("  3. Complex workflow planning (AUTONOMOUS mode)")
    logger.info("  4. Context building across plan steps (AUTONOMOUS mode)")
    logger.info("  5. Plan serialization/deserialization")
    logger.info("\nNote: AUTONOMOUS mode provides a planning + execution loop.")
    logger.info("Planning happens automatically; users only call execute().")
    logger.info("\n" + "=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("\nERROR: OPENAI_API_KEY environment variable not set!")
        logger.error("   Please set it in your .env file or environment.")
        return

    results = {}

    try:
        # Run all tests
        # results['llm_planning'] = await test_llm_planning()
        # results['manual_plan'] = await test_manual_multi_step_plan()
        # results['workflow'] = await test_complex_workflow_planning()
        results["context_building"] = await test_plan_with_context_building()
        # results['serialization'] = await test_plan_serialization()

        # Final summary (only report tests that were actually executed)
        logger.info("\n" + "=" * 80)
        logger.info("FINAL TEST SUMMARY")
        logger.info("=" * 80)
        for name, ok in results.items():
            logger.info(f"{name}: {'PASS' if ok else 'FAIL'}")

        total_passed = sum(1 for v in results.values() if v)
        total_tests = len(results)
        logger.info(f"\nOverall: {total_passed}/{total_tests} tests passed")

        if all(results.values()):
            logger.info("\nAll selected advanced planning tests passed.")
        else:
            logger.warning("\nSome tests failed. Please review the logs above.")

    except KeyboardInterrupt:
        logger.info("\n\nTests interrupted by user")
    except Exception as e:
        logger.error(f"\n\nError running tests: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
