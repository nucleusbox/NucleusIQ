"""Quick test to diagnose hanging issue."""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import asyncio
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM

async def test_simple():
    """Simple test to check if agent works."""
    print("Creating MockLLM...")
    llm = MockLLM()
    
    print("Creating Agent...")
    agent = Agent(
        name="TestAgent",
        role="Assistant",
        objective="Help users",
        llm=llm,
        config=AgentConfig(
            execution_mode=ExecutionMode.DIRECT,
            verbose=False
        )
    )
    
    print("Initializing agent...")
    await agent.initialize()
    
    print("Creating task...")
    task = Task(id="task1", objective="Hello")
    
    print("Executing task...")
    result = await agent.execute(task)
    
    print(f"Result: {result}")
    print("Test completed successfully!")

if __name__ == "__main__":
    print("Starting test...")
    asyncio.run(test_simple())
    print("Done!")

