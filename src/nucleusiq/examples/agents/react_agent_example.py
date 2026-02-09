# src/examples/agents/react_agent_example.py
"""
Example: ReAct (Reasoning + Acting) Agent

This example demonstrates how to create and use a ReAct agent that:
- Reasons about what to do next (Thought)
- Takes actions using tools (Action)
- Observes results and continues (Observation)
- Loops until final answer
"""

import asyncio
import logging
import os
import sys

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents.react_agent import ReActAgent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools import BaseTool
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    """Simple calculator tool for ReAct example."""
    
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


class WeatherTool(BaseTool):
    """Mock weather tool for ReAct example."""
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get weather information for a location"
        )
        self._weather_data = {
            "Paris": "Sunny, 22°C",
            "London": "Cloudy, 15°C",
            "New York": "Rainy, 18°C"
        }
    
    async def initialize(self) -> None:
        pass
    
    async def execute(self, location: str) -> str:
        """Get weather for a location."""
        return self._weather_data.get(location, f"Weather data not available for {location}")
    
    def get_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            }
        }


async def main():
    """Main example: Creating and using a ReAct agent."""
    
    logger.info("=" * 60)
    logger.info("ReAct Agent Example")
    logger.info("=" * 60)
    
    # Step 1: Create LLM
    logger.info("\n1. Creating LLM...")
    llm = MockLLM()
    logger.info("✅ LLM created (MockLLM)")
    
    # Step 2: Create tools
    logger.info("\n2. Creating tools...")
    calculator = CalculatorTool()
    weather = WeatherTool()
    logger.info(f"✅ {calculator.name} tool created")
    logger.info(f"✅ {weather.name} tool created")
    
    # Step 3: Create ReAct agent
    logger.info("\n3. Creating ReAct agent...")
    agent = ReActAgent(
        name="ReActAgent",
        role="Assistant",
        objective="Answer questions using reasoning and tools",
        narrative="A ReAct agent that reasons before acting",
        llm=llm,
        tools=[calculator, weather],
        max_iterations=10,
        config=AgentConfig(verbose=True)
    )
    logger.info("✅ ReAct agent created")
    
    # Step 4: Initialize
    logger.info("\n4. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")
    
    # Step 5: Execute task with ReAct pattern
    logger.info("\n5. Executing task with ReAct pattern...")
    logger.info("=" * 60)
    
    task = {
        "id": "task1",
        "objective": "What is 15 + 27? Also, what's the weather in Paris?"
    }
    
    logger.info(f"Task: {task['objective']}")
    logger.info("=" * 60)
    
    try:
        result = await agent.execute(task)
        logger.info(f"\n✅ Final Result: {result}")
        
        # Show ReAct history
        logger.info("\n" + "=" * 60)
        logger.info("ReAct Execution History:")
        logger.info("=" * 60)
        
        history = agent.get_react_history()
        for step in history:
            logger.info(f"\nIteration {step['iteration']}:")
            logger.info(f"  Thought: {step.get('thought', 'N/A')}")
            logger.info(f"  Action: {step.get('action', 'N/A')}")
            if 'observation' in step:
                logger.info(f"  Observation: {step['observation']}")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed!")
    logger.info("=" * 60)
    logger.info("\n💡 Key takeaway: ReAct agents reason before acting,")
    logger.info("   creating a Thought -> Action -> Observation loop!")


if __name__ == "__main__":
    asyncio.run(main())

