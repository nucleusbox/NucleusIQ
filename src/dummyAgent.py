# quick example (same as you did with EchoAgent)
from nucleusiq.core.llms.mock_llm import MockLLM
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.core.tools.base_tool import BaseTool
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.agent import Agent
import asyncio

async def demo():
    llm = MockLLM()
    prompt = (
        PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)
        .configure(system="You are a helpful assistant.",
                   user="Compute the sum of two numbers or repeat.")
    )

    def add(a: int, b: int) -> int: return a + b
    add_tool = BaseTool.from_function(add, description="Add two integers.")

    agent = Agent(
        name="CoreBot",
        role="Math & Echo",
        objective="Add numbers or echo.",
        narrative="First NucleusIQ core agent.",
        llm=llm,
        prompt=prompt,
        tools=[add_tool],
        config=AgentConfig(verbose=True),
    )
    await agent.initialize()
    result = await agent.execute({"id": "t1", "objective": "Add 7 and 8."})
    print(result)          # → “Dummy Model final answer incorporating function output is: 15”

if __name__ == "__main__":
    asyncio.run(demo())