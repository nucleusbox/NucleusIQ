import asyncio
import logging

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique

from typing import Dict, Any, List

class EchoAgent(Agent):
    """
    A simple agent that echoes the task objective.
    """
    
    async def execute(self, task: Dict[str, Any]) -> Any:
        """
        Executes the task by echoing the objective.
        """
        self._logger.info(f"Executing task with objective: {task['objective']}")
        # Simulate some processing delay
        await asyncio.sleep(2)
        result = f"Echo: {task['objective']}"
        return result

    async def plan(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Creates a simple execution plan with one step.
        """
        self._logger.debug("Creating a simple execution plan.")
        return [{
            "step": 1,
            "action": "execute",
            "task": task
        }]

async def main():
    # Configure logging at the root level
    logging.basicConfig(level=logging.DEBUG)

    # Initialize the mock LLM and simple prompt
    llm = MockLLM()

    zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="You are a helpful assistant.",
            user="Translate the following English text to French: 'Hello, how are you?'"
        )

    # Format the prompt
    prompt = zero_shot

    # Create an instance of EchoAgent
    agent = EchoAgent(
        name="EchoBot",
        role="Echo Service",
        objective="Echo the provided objective.",
        narrative="EchoBot is designed to repeat back the objective of any given task.",
        llm=llm,
        prompt=prompt,
        config=AgentConfig(verbose=True)
    )

    # Initialize the agent
    await agent.initialize()

    # Define a simple task
    task = {
        "id": "task1",
        "objective": "Hello, NucleusIQ!"
    }

    # Execute the task
    try:
        result = await agent.execute(task)
        print(f"Task Result: {result}")
    except Exception as e:
        print(f"Task failed with error: {e}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())