"""
Example: Agent Execution Modes (Gearbox Strategy)

This example demonstrates the three execution modes available in NucleusIQ:
1. DIRECT mode (Gear 1): Fast, simple, no tools - for chat/conversation
2. STANDARD mode (Gear 2): Tool-enabled, linear execution - for tool usage
3. AUTONOMOUS mode (Gear 3): Full reasoning loop with planning - for complex tasks

Key Features Demonstrated:
- ExecutionMode enum usage
- Auto-initialization (no need to call initialize() manually)
- Task creation and execution
- Tool integration
- Prompt precedence

Run with: python src/examples/agents/execution_modes_example.py

Requires OPENAI_API_KEY environment variable for real LLM, or uses MockLLM.
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
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.llms.mock_llm import MockLLM

# Try to import OpenAI LLM, fallback to MockLLM if not available
try:
    from nucleusiq_openai import BaseOpenAI
    USE_REAL_LLM = os.getenv("OPENAI_API_KEY") is not None
except ImportError:
    USE_REAL_LLM = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_calculator_tool():
    """Create a simple calculator tool for examples."""
    def add(a: int, b: int) -> int:
        """Add two integers together."""
        return a + b
    
    def multiply(a: int, b: int) -> int:
        """Multiply two integers together."""
        return a * b
    
    add_tool = BaseTool.from_function(add, description="Add two integers together.")
    multiply_tool = BaseTool.from_function(multiply, description="Multiply two integers together.")
    
    return [add_tool, multiply_tool]


async def example_direct_mode():
    """Example 1: DIRECT mode - Fast, simple, no tools."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: DIRECT MODE (Gear 1)")
    logger.info("=" * 80)
    logger.info("Purpose: Fast, simple conversation without tools")
    logger.info("Use Case: Chatbots, Q&A, creative writing")
    logger.info("-" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)")
    
    # Create prompt
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful assistant that answers questions clearly and concisely.",
        user="Answer the user's question."
    )
    
    # Create agent with DIRECT mode
    agent = Agent(
        name="ChatBot",
        role="Assistant",
        objective="Answer questions",
        llm=llm,
        prompt=prompt,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,  # Gear 1
            verbose=True
        )
    )
    
    logger.info(f"✅ Agent created: {agent.name}")
    logger.info(f"   Execution Mode: {agent.config.execution_mode.value}")
    logger.info(f"   Note: Tools are ignored in DIRECT mode")
    
    # Execute tasks (auto-initialization happens automatically!)
    tasks = [
        Task(id="task1", objective="What is the capital of France?"),
        Task(id="task2", objective="Explain quantum computing in one sentence."),
        Task(id="task3", objective="Write a haiku about programming."),
    ]
    
    for task in tasks:
        logger.info(f"\n📋 Task: {task.objective}")
        try:
            result = await agent.execute(task)  # Auto-initializes if needed!
            logger.info(f"✅ Result: {result}")
            logger.info(f"   Agent State: {agent.state}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def example_standard_mode():
    """Example 2: STANDARD mode - Tool-enabled, linear execution."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: STANDARD MODE (Gear 2)")
    logger.info("=" * 80)
    logger.info("Purpose: Tool-enabled tasks with linear execution")
    logger.info("Use Case: Calculations, API calls, data queries")
    logger.info("-" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)")
    
    # Create tools
    tools = create_calculator_tool()
    logger.info(f"✅ Created {len(tools)} tools: {[t.name for t in tools]}")
    
    # Create prompt
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful calculator assistant. Use tools to perform calculations.",
        user="Use the appropriate tool to solve the user's problem."
    )
    
    # Create agent with STANDARD mode (default)
    agent = Agent(
        name="CalculatorBot",
        role="Calculator",
        objective="Perform calculations using tools",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.STANDARD,  # Gear 2 (default)
            verbose=True
        )
    )
    
    logger.info(f"✅ Agent created: {agent.name}")
    logger.info(f"   Execution Mode: {agent.config.execution_mode.value}")
    logger.info(f"   Tools: {len(agent.tools)} available")
    
    # Execute tasks
    tasks = [
        Task(id="task1", objective="Add 15 and 27"),
        Task(id="task2", objective="Multiply 8 by 9"),
        Task(id="task3", objective="What is 100 + 200?"),
    ]
    
    for task in tasks:
        logger.info(f"\n📋 Task: {task.objective}")
        try:
            result = await agent.execute(task)  # Auto-initializes and uses tools!
            logger.info(f"✅ Result: {result}")
            logger.info(f"   Agent State: {agent.state}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def example_autonomous_mode():
    """Example 3: AUTONOMOUS mode - Full reasoning loop (currently falls back to standard)."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: AUTONOMOUS MODE (Gear 3)")
    logger.info("=" * 80)
    logger.info("Purpose: Full reasoning loop with planning and self-correction")
    logger.info("Use Case: Complex multi-step tasks, research, analysis")
    logger.info("Status: Currently falls back to STANDARD mode (full implementation in Week 2)")
    logger.info("-" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)")
    
    # Create tools
    tools = create_calculator_tool()
    
    # Create agent with AUTONOMOUS mode
    agent = Agent(
        name="ResearchBot",
        role="Researcher",
        objective="Research and analyze complex topics",
        llm=llm,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,  # Gear 3
            verbose=True,
            max_iterations=10
        )
    )
    
    logger.info(f"✅ Agent created: {agent.name}")
    logger.info(f"   Execution Mode: {agent.config.execution_mode.value}")
    logger.info(f"   ⚠️  Note: Currently falls back to STANDARD mode")
    
    # Execute tasks
    tasks = [
        Task(id="task1", objective="Calculate (5 + 3) * 2"),
        Task(id="task2", objective="What is 10 * 10 + 5?"),
    ]
    
    for task in tasks:
        logger.info(f"\n📋 Task: {task.objective}")
        try:
            result = await agent.execute(task)
            logger.info(f"✅ Result: {result}")
            logger.info(f"   Agent State: {agent.state}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def example_prompt_precedence():
    """Example 4: Prompt precedence - prompt takes precedence over role/objective."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: PROMPT PRECEDENCE")
    logger.info("=" * 80)
    logger.info("Key Concept: If prompt is provided, it takes precedence over role/objective")
    logger.info("             for LLM message construction during execution.")
    logger.info("             role/objective are still used for planning context.")
    logger.info("-" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)")
    
    # Create agent WITH prompt (prompt takes precedence)
    agent_with_prompt = Agent(
        name="PromptBot",
        role="Calculator",  # This is used for planning context only
        objective="Perform calculations",  # This is used for planning context only
        llm=llm,
        prompt=PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="You are a creative poet. Write beautiful poems.",
            user="Write a poem about the user's request."
        ),
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            verbose=True
        )
    )
    
    logger.info("✅ Agent WITH prompt created")
    logger.info(f"   Role: {agent_with_prompt.role} (used for planning only)")
    logger.info(f"   Prompt System: 'You are a creative poet...' (used for execution)")
    
    # Create agent WITHOUT prompt (role/objective used)
    agent_without_prompt = Agent(
        name="RoleBot",
        role="Calculator",  # This is used to build system message
        objective="Perform calculations",  # This is used to build system message
        llm=llm,
        prompt=None,  # No prompt, so role/objective are used
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            verbose=True
        )
    )
    
    logger.info("\n✅ Agent WITHOUT prompt created")
    logger.info(f"   Role: {agent_without_prompt.role} (used for execution)")
    logger.info(f"   Objective: {agent_without_prompt.objective} (used for execution)")
    
    # Test both agents with same task
    task = Task(id="task1", objective="What is 2 + 2?")
    
    logger.info(f"\n📋 Testing with task: {task.objective}")
    logger.info("\n--- Agent WITH prompt (should write a poem) ---")
    try:
        result1 = await agent_with_prompt.execute(task)
        logger.info(f"✅ Result: {result1}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    logger.info("\n--- Agent WITHOUT prompt (should calculate) ---")
    try:
        result2 = await agent_without_prompt.execute(task)
        logger.info(f"✅ Result: {result2}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


async def example_auto_initialization():
    """Example 5: Auto-initialization feature."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: AUTO-INITIALIZATION")
    logger.info("=" * 80)
    logger.info("Key Feature: No need to call initialize() manually!")
    logger.info("             Agent auto-initializes on first execute() call.")
    logger.info("-" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)")
    
    # Create agent
    agent = Agent(
        name="AutoInitBot",
        role="Assistant",
        objective="Help users",
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            verbose=True
        )
    )
    
    logger.info(f"✅ Agent created: {agent.name}")
    logger.info(f"   Initial State: {agent.state}")
    logger.info(f"   Executor: {agent._executor} (not initialized yet)")
    
    # Execute without manual initialization (auto-initializes!)
    task = Task(id="task1", objective="Hello, how are you?")
    
    logger.info(f"\n📋 Executing task WITHOUT calling initialize()...")
    logger.info("   (Agent will auto-initialize on first execute() call)")
    
    try:
        result = await agent.execute(task)  # Auto-initializes here!
        logger.info(f"✅ Result: {result}")
        logger.info(f"   Final State: {agent.state}")
        logger.info(f"   Executor: {agent._executor} (now initialized)")
        logger.info("\n💡 Key takeaway: You can call execute() directly without initialize()!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


async def main():
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("NucleusIQ Agent Execution Modes Examples")
    logger.info("=" * 80)
    logger.info("\nThis example demonstrates:")
    logger.info("1. DIRECT mode - Fast, simple, no tools")
    logger.info("2. STANDARD mode - Tool-enabled, linear execution")
    logger.info("3. AUTONOMOUS mode - Full reasoning loop (falls back to standard)")
    logger.info("4. Prompt precedence - prompt vs role/objective")
    logger.info("5. Auto-initialization - no need to call initialize() manually")
    logger.info("\n" + "=" * 80)
    
    try:
        # Run all examples
        await example_direct_mode()
        await example_standard_mode()
        await example_autonomous_mode()
        await example_prompt_precedence()
        await example_auto_initialization()
        
        logger.info("\n" + "=" * 80)
        logger.info("All examples completed!")
        logger.info("=" * 80)
        logger.info("\n💡 Key Takeaways:")
        logger.info("   • Use DIRECT mode for simple chat/conversation")
        logger.info("   • Use STANDARD mode for tool-enabled tasks")
        logger.info("   • Use AUTONOMOUS mode for complex multi-step tasks (coming soon)")
        logger.info("   • Prompt takes precedence over role/objective for execution")
        logger.info("   • Auto-initialization means you can call execute() directly")
        logger.info("   • ExecutionMode enum provides type safety and clarity")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())




