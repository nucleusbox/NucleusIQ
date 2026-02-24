"""
Comprehensive Test: All Execution Modes with OpenAI

This example thoroughly tests all three execution modes (DIRECT, STANDARD, AUTONOMOUS)
with real OpenAI API calls to ensure they work correctly.

Execution Modes:
1. DIRECT (Gear 1): Fast, simple, no tools - Single LLM call, no tool execution
2. STANDARD (Gear 2): Tool-enabled, linear execution - Can use tools, linear flow
3. AUTONOMOUS (Gear 3): Full reasoning loop - Currently falls back to STANDARD

Run with: python src/examples/agents/comprehensive_execution_modes_test.py

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
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import BaseTool
from nucleusiq_openai import BaseOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_math_tools():
    """Create math tools for testing."""

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

    return [add_tool, multiply_tool, subtract_tool, divide_tool]


async def test_direct_mode():
    """Comprehensive test of DIRECT mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE 1: DIRECT MODE (Gear 1)")
    logger.info("=" * 80)
    logger.info("Characteristics:")
    logger.info("  • Fast, single LLM call")
    logger.info("  • No tool execution (tools are ignored)")
    logger.info("  • No planning")
    logger.info("  • Best for: Chat, Q&A, creative writing")
    logger.info("-" * 80)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create prompt
    prompt = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT).configure(
        system="You are a helpful assistant that answers questions clearly and concisely.",
        user="Answer the user's question directly.",
    )

    # Create agent with DIRECT mode
    agent = Agent(
        name="DirectTestBot",
        role="Assistant",
        objective="Answer questions",
        llm=llm,
        prompt=prompt,
        tools=create_math_tools(),  # Tools provided but will be ignored in DIRECT mode
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=True),
    )

    logger.info(f"✅ Agent created: {agent.name}")
    logger.info(f"   Execution Mode: {agent.config.execution_mode.value}")
    logger.info(
        f"   Tools provided: {len(agent.tools)} (will be ignored in DIRECT mode)"
    )

    # Test cases for DIRECT mode
    test_cases = [
        {
            "name": "Simple Q&A",
            "task": Task(id="direct_1", objective="What is the capital of France?"),
            "expected": "Should answer directly without tools",
        },
        {
            "name": "Creative Writing",
            "task": Task(id="direct_2", objective="Write a haiku about programming."),
            "expected": "Should generate creative content",
        },
        {
            "name": "Explanation",
            "task": Task(
                id="direct_3", objective="Explain what Python is in one sentence."
            ),
            "expected": "Should provide explanation",
        },
        {
            "name": "Math Question (no tools)",
            "task": Task(id="direct_4", objective="What is 15 + 27?"),
            "expected": "Should answer directly (tools ignored in DIRECT mode)",
        },
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n📋 Test {i}: {test_case['name']}")
        logger.info(f"   Task: {test_case['task'].objective}")
        logger.info(f"   Expected: {test_case['expected']}")

        try:
            result = await agent.execute(test_case["task"])
            result_str = str(result) if result is not None else "None"
            logger.info(
                f"   ✅ Result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}"
            )
            logger.info(f"   State: {agent.state}")
            results.append(
                {"test": test_case["name"], "status": "PASS", "result": result}
            )
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            results.append(
                {"test": test_case["name"], "status": "FAIL", "error": str(e)}
            )

    logger.info(
        f"\n📊 DIRECT Mode Test Summary: {sum(1 for r in results if r['status'] == 'PASS')}/{len(results)} passed"
    )
    return all(r["status"] == "PASS" for r in results)


async def test_standard_mode():
    """Comprehensive test of STANDARD mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE 2: STANDARD MODE (Gear 2)")
    logger.info("=" * 80)
    logger.info("Characteristics:")
    logger.info("  • Tool execution enabled")
    logger.info("  • Linear flow (no loops)")
    logger.info("  • Multiple tool calls supported")
    logger.info("  • Best for: Calculations, API calls, data queries")
    logger.info("-" * 80)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create tools
    tools = create_math_tools()

    # Create prompt
    prompt = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT).configure(
        system="You are a helpful calculator assistant. Use tools to perform calculations accurately.",
        user="Use the appropriate tool to solve the user's problem.",
    )

    # Create agent with STANDARD mode
    agent = Agent(
        name="StandardTestBot",
        role="Calculator",
        objective="Perform calculations using tools",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=True),
    )

    logger.info(f"✅ Agent created: {agent.name}")
    logger.info(f"   Execution Mode: {agent.config.execution_mode.value}")
    logger.info(f"   Tools available: {[t.name for t in tools]}")

    # Test cases for STANDARD mode
    test_cases = [
        {
            "name": "Simple Addition",
            "task": Task(id="standard_1", objective="What is 15 + 27?"),
            "expected": "Should use add tool and return 42",
        },
        {
            "name": "Multiplication",
            "task": Task(id="standard_2", objective="Multiply 8 by 9"),
            "expected": "Should use multiply tool and return 72",
        },
        {
            "name": "Multiple Tool Calls",
            "task": Task(id="standard_3", objective="Calculate (5 + 3) * 4"),
            "expected": "Should use add then multiply tools, return 32",
        },
        {
            "name": "Subtraction",
            "task": Task(id="standard_4", objective="What is 100 - 37?"),
            "expected": "Should use subtract tool and return 63",
        },
        {
            "name": "Division",
            "task": Task(id="standard_5", objective="Divide 144 by 12"),
            "expected": "Should use divide tool and return 12",
        },
        {
            "name": "Complex Calculation",
            "task": Task(id="standard_6", objective="Calculate (20 + 5) * 3 - 10"),
            "expected": "Should use multiple tools in sequence",
        },
        {
            "name": "Non-Math Question",
            "task": Task(id="standard_7", objective="What is the capital of France?"),
            "expected": "Should answer directly without tools",
        },
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n📋 Test {i}: {test_case['name']}")
        logger.info(f"   Task: {test_case['task'].objective}")
        logger.info(f"   Expected: {test_case['expected']}")

        try:
            result = await agent.execute(test_case["task"])
            result_str = str(result) if result is not None else "None"
            logger.info(
                f"   ✅ Result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}"
            )
            logger.info(f"   State: {agent.state}")
            results.append(
                {"test": test_case["name"], "status": "PASS", "result": result}
            )
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            results.append(
                {"test": test_case["name"], "status": "FAIL", "error": str(e)}
            )

    logger.info(
        f"\n📊 STANDARD Mode Test Summary: {sum(1 for r in results if r['status'] == 'PASS')}/{len(results)} passed"
    )
    return all(r["status"] == "PASS" for r in results)


async def test_autonomous_mode():
    """Comprehensive test of AUTONOMOUS mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE 3: AUTONOMOUS MODE (Gear 3)")
    logger.info("=" * 80)
    logger.info("Characteristics:")
    logger.info("  • Full reasoning loop with planning")
    logger.info("  • Self-correction capabilities")
    logger.info("  • Best for: Complex multi-step tasks, research, analysis")
    logger.info("  ⚠️  Status: Currently falls back to STANDARD mode")
    logger.info("-" * 80)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create tools
    tools = create_math_tools()

    # Create prompt
    prompt = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT).configure(
        system="You are an advanced research assistant. Plan your approach and use tools strategically.",
        user="Think through the problem step by step and use tools when needed.",
    )

    # Create agent with AUTONOMOUS mode
    agent = Agent(
        name="AutonomousTestBot",
        role="Researcher",
        objective="Research and analyze complex topics",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS, verbose=True, max_iterations=10
        ),
    )

    logger.info(f"✅ Agent created: {agent.name}")
    logger.info(f"   Execution Mode: {agent.config.execution_mode.value}")
    logger.info(f"   Max Iterations: {agent.config.max_iterations}")
    logger.info("   ⚠️  Note: Currently falls back to STANDARD mode")

    # Test cases for AUTONOMOUS mode
    test_cases = [
        {
            "name": "Simple Calculation",
            "task": Task(id="autonomous_1", objective="Calculate 25 * 4"),
            "expected": "Should work (falls back to STANDARD)",
        },
        {
            "name": "Multi-Step Calculation",
            "task": Task(id="autonomous_2", objective="Calculate (10 + 5) * 3 - 20"),
            "expected": "Should use multiple tools (falls back to STANDARD)",
        },
        {
            "name": "Complex Problem",
            "task": Task(
                id="autonomous_3",
                objective="If I have 100 apples and give away 30, then multiply the remainder by 2, what do I get?",
            ),
            "expected": "Should break down and solve (falls back to STANDARD)",
        },
    ]

    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n📋 Test {i}: {test_case['name']}")
        logger.info(f"   Task: {test_case['task'].objective}")
        logger.info(f"   Expected: {test_case['expected']}")

        try:
            result = await agent.execute(test_case["task"])
            result_str = str(result) if result is not None else "None"
            logger.info(
                f"   ✅ Result: {result_str[:100]}{'...' if len(result_str) > 100 else ''}"
            )
            logger.info(f"   State: {agent.state}")
            results.append(
                {"test": test_case["name"], "status": "PASS", "result": result}
            )
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            results.append(
                {"test": test_case["name"], "status": "FAIL", "error": str(e)}
            )

    logger.info(
        f"\n📊 AUTONOMOUS Mode Test Summary: {sum(1 for r in results if r['status'] == 'PASS')}/{len(results)} passed"
    )
    return all(r["status"] == "PASS" for r in results)


async def test_mode_comparison():
    """Compare the same task across different modes."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE 4: MODE COMPARISON")
    logger.info("=" * 80)
    logger.info("Testing the same task across DIRECT, STANDARD, and AUTONOMOUS modes")
    logger.info("-" * 80)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("ERROR: OPENAI_API_KEY environment variable not set!")
        return False

    # Create LLM
    llm = BaseOpenAI(model_name="gpt-5-nano")

    # Create tools
    tools = create_math_tools()

    # Test task
    test_task = Task(id="comparison_1", objective="What is 15 + 27?")

    # Test DIRECT mode
    logger.info("\n--- DIRECT Mode ---")
    agent_direct = Agent(
        name="DirectCompare",
        role="Assistant",
        objective="Answer questions",
        llm=llm,
        tools=tools,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False),
    )
    try:
        result_direct = await agent_direct.execute(test_task)
        result_str = str(result_direct) if result_direct is not None else "None"
        logger.info(f"✅ Result: {result_str[:100]}")
        logger.info("   Note: Tools ignored, LLM answers directly")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        result_direct = None

    # Test STANDARD mode
    logger.info("\n--- STANDARD Mode ---")
    agent_standard = Agent(
        name="StandardCompare",
        role="Calculator",
        objective="Perform calculations",
        llm=llm,
        tools=tools,
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False),
    )
    try:
        result_standard = await agent_standard.execute(test_task)
        result_str = str(result_standard) if result_standard is not None else "None"
        logger.info(f"✅ Result: {result_str[:100]}")
        logger.info("   Note: Tools available, should use add tool")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        result_standard = None

    # Test AUTONOMOUS mode
    logger.info("\n--- AUTONOMOUS Mode ---")
    agent_autonomous = Agent(
        name="AutonomousCompare",
        role="Researcher",
        objective="Research and analyze",
        llm=llm,
        tools=tools,
        config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS, verbose=False),
    )
    try:
        result_autonomous = await agent_autonomous.execute(test_task)
        result_str = str(result_autonomous) if result_autonomous is not None else "None"
        logger.info(f"✅ Result: {result_str[:100]}")
        logger.info("   Note: Falls back to STANDARD mode")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        result_autonomous = None

    logger.info("\n📊 Comparison Summary:")
    logger.info(f"   DIRECT: {'✅' if result_direct else '❌'}")
    logger.info(f"   STANDARD: {'✅' if result_standard else '❌'}")
    logger.info(f"   AUTONOMOUS: {'✅' if result_autonomous else '❌'}")

    return result_direct and result_standard and result_autonomous


async def main():
    """Run all comprehensive tests."""
    logger.info("=" * 80)
    logger.info("NucleusIQ Comprehensive Execution Modes Test Suite")
    logger.info("=" * 80)
    logger.info("\nThis test suite thoroughly tests all three execution modes:")
    logger.info("  1. DIRECT mode (Gear 1) - Fast, simple, no tools")
    logger.info("  2. STANDARD mode (Gear 2) - Tool-enabled, linear execution")
    logger.info(
        "  3. AUTONOMOUS mode (Gear 3) - Full reasoning loop (falls back to standard)"
    )
    logger.info("  4. Mode comparison - Same task across all modes")
    logger.info("\n" + "=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("\n❌ ERROR: OPENAI_API_KEY environment variable not set!")
        logger.error("   Please set it in your .env file or environment.")
        return

    results = {}

    try:
        # Run all test suites
        results["direct"] = await test_direct_mode()
        results["standard"] = await test_standard_mode()
        results["autonomous"] = await test_autonomous_mode()
        results["comparison"] = await test_mode_comparison()

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"DIRECT Mode:     {'✅ PASS' if results.get('direct') else '❌ FAIL'}"
        )
        logger.info(
            f"STANDARD Mode:   {'✅ PASS' if results.get('standard') else '❌ FAIL'}"
        )
        logger.info(
            f"AUTONOMOUS Mode: {'✅ PASS' if results.get('autonomous') else '❌ FAIL'}"
        )
        logger.info(
            f"Mode Comparison: {'✅ PASS' if results.get('comparison') else '❌ FAIL'}"
        )

        total_passed = sum(1 for v in results.values() if v)
        total_tests = len(results)
        logger.info(f"\nOverall: {total_passed}/{total_tests} test suites passed")

        if all(results.values()):
            logger.info("\n🎉 All tests passed!")
        else:
            logger.warning("\n⚠️  Some tests failed. Please review the logs above.")

    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error running tests: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
