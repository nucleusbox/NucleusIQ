"""
Example: Gearbox Strategy - Choosing the Right Execution Mode

This example helps you understand when to use each execution mode (gear)
based on your task requirements.

Gearbox Strategy:
- Gear 1 (DIRECT): Fast, simple, no tools - for chat/conversation
- Gear 2 (STANDARD): Tool-enabled, linear execution - for tool usage
- Gear 3 (AUTONOMOUS): Full reasoning loop - for complex tasks

Run with: python src/examples/agents/gearbox_strategy_example.py
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.llms.mock_llm import MockLLM

# Try to import OpenAI LLM
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


def create_tools():
    """Create example tools."""
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny, 72°F"
    
    def search_database(query: str) -> str:
        """Search a database."""
        return f"Database results for '{query}': Found 5 records"
    
    weather_tool = BaseTool.from_function(get_weather, description="Get weather for a city")
    db_tool = BaseTool.from_function(search_database, description="Search a database")
    
    return [weather_tool, db_tool]


async def demonstrate_gear_selection():
    """Demonstrate how to choose the right gear for different tasks."""
    
    logger.info("=" * 80)
    logger.info("Gearbox Strategy: Choosing the Right Execution Mode")
    logger.info("=" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)\n")
    
    # Create tools
    tools = create_tools()
    
    # ========================================================================
    # SCENARIO 1: Simple Q&A - Use DIRECT mode (Gear 1)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 1: Simple Q&A → Use DIRECT mode (Gear 1)")
    logger.info("=" * 80)
    logger.info("Task Type: Simple questions, conversation, creative writing")
    logger.info("Characteristics: No tools needed, single LLM call, fast")
    logger.info("-" * 80)
    
    agent_direct = Agent(
        name="QABot",
        role="Assistant",
        objective="Answer questions",
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,  # Gear 1
            verbose=False
        )
    )
    
    task1 = Task(id="task1", objective="What is the capital of France?")
    logger.info(f"📋 Task: {task1.objective}")
    try:
        result = await agent_direct.execute(task1)
        logger.info(f"✅ Result: {result}")
        logger.info(f"   Mode: {agent_direct.config.execution_mode.value}")
        logger.info(f"   Tools Used: None (DIRECT mode ignores tools)")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # ========================================================================
    # SCENARIO 2: Tool Usage - Use STANDARD mode (Gear 2)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 2: Tool Usage → Use STANDARD mode (Gear 2)")
    logger.info("=" * 80)
    logger.info("Task Type: Calculations, API calls, data queries")
    logger.info("Characteristics: Needs tools, linear execution, reliable")
    logger.info("-" * 80)
    
    agent_standard = Agent(
        name="ToolBot",
        role="Assistant",
        objective="Use tools to help users",
        llm=llm,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.STANDARD,  # Gear 2 (default)
            verbose=False
        )
    )
    
    task2 = Task(id="task2", objective="What's the weather in New York?")
    logger.info(f"📋 Task: {task2.objective}")
    try:
        result = await agent_standard.execute(task2)
        logger.info(f"✅ Result: {result}")
        logger.info(f"   Mode: {agent_standard.config.execution_mode.value}")
        logger.info(f"   Tools Available: {len(agent_standard.tools)}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # ========================================================================
    # SCENARIO 3: Complex Multi-Step Task - Use AUTONOMOUS mode (Gear 3)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 3: Complex Multi-Step Task → Use AUTONOMOUS mode (Gear 3)")
    logger.info("=" * 80)
    logger.info("Task Type: Research, analysis, multi-step reasoning")
    logger.info("Characteristics: Needs planning, self-correction, context")
    logger.info("Status: Currently falls back to STANDARD (full implementation coming)")
    logger.info("-" * 80)
    
    agent_autonomous = Agent(
        name="ResearchBot",
        role="Researcher",
        objective="Research and analyze complex topics",
        llm=llm,
        tools=tools,
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,  # Gear 3

            max_iterations=10,
            verbose=False
        )
    )
    
    task3 = Task(id="task3", objective="Research the weather in New York and search the database for related information")
    logger.info(f"📋 Task: {task3.objective}")
    try:
        result = await agent_autonomous.execute(task3)
        logger.info(f"✅ Result: {result}")
        logger.info(f"   Mode: {agent_autonomous.config.execution_mode.value}")
        logger.info(f"   ⚠️  Note: Currently falls back to STANDARD mode")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # ========================================================================
    # DECISION TREE
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DECISION TREE: How to Choose the Right Gear")
    logger.info("=" * 80)
    logger.info("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    START: Choose Execution Mode              │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
        ┌───────▼────────┐              ┌────────▼────────┐
        │ Need tools?    │              │ No tools needed │
        │                │              │                 │
        │   YES          │              │      YES        │
        └───────┬────────┘              └────────┬────────┘
                │                                 │
    ┌───────────┴───────────┐                    │
    │                       │                    │
┌───▼────┐          ┌───────▼──────┐            │
│ Simple │          │ Complex      │            │
│ tool   │          │ multi-step   │            │
│ usage? │          │ task?        │            │
└───┬────┘          └───────┬──────┘            │
    │                       │                    │
    │ YES                   │ YES                │
    │                       │                    │
    │              ┌────────▼────────┐           │
    │              │ AUTONOMOUS      │           │
    │              │ (Gear 3)        │           │
    │              └─────────────────┘           │
    │                                             │
    │              ┌─────────────────┐           │
    │              │ STANDARD        │           │
    │              │ (Gear 2)        │           │
    │              └─────────────────┘           │
    │                                             │
    │              ┌─────────────────┐           │
    │              │ DIRECT          │           │
    │              │ (Gear 1)        │           │
    │              └─────────────────┘           │
    │                                             │
    └─────────────────────────────────────────────┘
    """)
    
    logger.info("\n💡 Quick Reference:")
    logger.info("   • DIRECT: Chat, Q&A, creative writing → Fast, no tools")
    logger.info("   • STANDARD: Calculations, API calls → Tool-enabled, linear")
    logger.info("   • AUTONOMOUS: Research, analysis → Planning, self-correction")
    logger.info("\n" + "=" * 80)


async def main():
    """Run the gearbox strategy example."""
    try:
        await demonstrate_gear_selection()
        logger.info("\n✅ Example completed!")
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Example interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())




