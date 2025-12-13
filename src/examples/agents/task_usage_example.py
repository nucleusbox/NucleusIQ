"""
Example: Task Usage Patterns

This example demonstrates different ways to create and use Tasks with Agents:
1. Task object creation
2. Dictionary-based tasks
3. Task serialization/deserialization
4. Task with different execution modes

Run with: python src/examples/agents/task_usage_example.py
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
from nucleusiq.core.llms.mock_llm import MockLLM

# Try to import OpenAI LLM
try:
    from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
    USE_REAL_LLM = os.getenv("OPENAI_API_KEY") is not None
except ImportError:
    USE_REAL_LLM = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_task_object():
    """Example 1: Using Task objects."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: Using Task Objects")
    logger.info("=" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)\n")
    
    # Create agent
    agent = Agent(
        name="TaskBot",
        role="Assistant",
        objective="Help users",
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            verbose=False
        )
    )
    
    # Create Task object
    task = Task(
        id="task1",
        objective="What is the capital of France?",
        context={"user_id": "123", "session": "abc"}
    )
    
    logger.info(f"✅ Task created:")
    logger.info(f"   ID: {task.id}")
    logger.info(f"   Objective: {task.objective}")
    logger.info(f"   Context: {task.context}")
    
    # Execute task
    logger.info(f"\n📋 Executing task...")
    try:
        result = await agent.execute(task)
        logger.info(f"✅ Result: {result}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


async def example_task_dict():
    """Example 2: Using dictionary-based tasks."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Using Dictionary-Based Tasks")
    logger.info("=" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)\n")
    
    # Create agent
    agent = Agent(
        name="DictBot",
        role="Assistant",
        objective="Help users",
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            verbose=False
        )
    )
    
    # Create task as dictionary
    task_dict = {
        "id": "task2",
        "objective": "Explain Python in one sentence.",
        "context": {"source": "web"}
    }
    
    logger.info(f"✅ Task dictionary created:")
    logger.info(f"   {task_dict}")
    
    # Execute task (Agent accepts both Task objects and dicts)
    logger.info(f"\n📋 Executing task dictionary...")
    try:
        result = await agent.execute(task_dict)
        logger.info(f"✅ Result: {result}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


async def example_task_serialization():
    """Example 3: Task serialization and deserialization."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Task Serialization/Deserialization")
    logger.info("=" * 80)
    
    # Create a Task object
    original_task = Task(
        id="task3",
        objective="What is 2 + 2?",
        context={"user": "alice"}
    )
    
    logger.info(f"✅ Original Task:")
    logger.info(f"   ID: {original_task.id}")
    logger.info(f"   Objective: {original_task.objective}")
    
    # Convert to dictionary
    task_dict = original_task.to_dict()
    logger.info(f"\n✅ Task converted to dictionary:")
    logger.info(f"   {task_dict}")
    
    # Convert back to Task object
    restored_task = Task.from_dict(task_dict)
    logger.info(f"\n✅ Task restored from dictionary:")
    logger.info(f"   ID: {restored_task.id}")
    logger.info(f"   Objective: {restored_task.objective}")
    logger.info(f"   Context: {restored_task.context}")
    
    # Verify they're equivalent
    assert original_task.id == restored_task.id
    assert original_task.objective == restored_task.objective
    logger.info(f"\n✅ Verification: Original and restored tasks match!")


async def example_task_with_different_modes():
    """Example 4: Same task with different execution modes."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Same Task with Different Execution Modes")
    logger.info("=" * 80)
    
    # Create LLM
    if USE_REAL_LLM:
        llm = BaseOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    else:
        llm = MockLLM()
        logger.info("ℹ️  Using MockLLM (set OPENAI_API_KEY for real LLM)\n")
    
    # Same task
    task = Task(id="task4", objective="What is the weather today?")
    
    # Test with DIRECT mode
    logger.info("\n--- DIRECT Mode ---")
    agent_direct = Agent(
        name="DirectBot",
        role="Assistant",
        objective="Answer questions",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT, verbose=False)
    )
    try:
        result = await agent_direct.execute(task)
        logger.info(f"✅ Result: {result}")
        logger.info(f"   Mode: {agent_direct.config.execution_mode.value}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # Test with STANDARD mode
    logger.info("\n--- STANDARD Mode ---")
    agent_standard = Agent(
        name="StandardBot",
        role="Assistant",
        objective="Answer questions",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD, verbose=False)
    )
    try:
        result = await agent_standard.execute(task)
        logger.info(f"✅ Result: {result}")
        logger.info(f"   Mode: {agent_standard.config.execution_mode.value}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")


async def main():
    """Run all task usage examples."""
    logger.info("=" * 80)
    logger.info("NucleusIQ Task Usage Examples")
    logger.info("=" * 80)
    logger.info("\nThis example demonstrates:")
    logger.info("1. Task object creation and usage")
    logger.info("2. Dictionary-based tasks")
    logger.info("3. Task serialization/deserialization")
    logger.info("4. Same task with different execution modes")
    logger.info("\n" + "=" * 80)
    
    try:
        await example_task_object()
        await example_task_dict()
        await example_task_serialization()
        await example_task_with_different_modes()
        
        logger.info("\n" + "=" * 80)
        logger.info("All examples completed!")
        logger.info("=" * 80)
        logger.info("\n💡 Key Takeaways:")
        logger.info("   • You can use Task objects or dictionaries")
        logger.info("   • Tasks can be serialized to/from dictionaries")
        logger.info("   • Same task works with different execution modes")
        logger.info("   • Task context is preserved during serialization")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())




