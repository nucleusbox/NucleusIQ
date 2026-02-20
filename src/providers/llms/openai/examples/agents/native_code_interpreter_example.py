"""
Example: Code Interpreter with OpenAI Responses API

Demonstrates using OpenAI's built-in code interpreter tool.
The model can write and execute Python code server-side.

Usage:
    python native_code_interpreter_example.py
"""

import os
import sys
import asyncio
import logging

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)
from dotenv import load_dotenv

load_dotenv()

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq_openai import BaseOpenAI, OpenAITool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set — skipping.")
        return

    logger.info("=" * 60)
    logger.info("Native Code Interpreter Example")
    logger.info("=" * 60)

    # 1. Create LLM
    llm = BaseOpenAI(model_name="gpt-4o", temperature=0)

    # 2. Create code interpreter tool
    code_interp = OpenAITool.code_interpreter()
    logger.info(f"Tool: {code_interp.name}  |  type: {code_interp.tool_type}  |  native: {code_interp.is_native}")

    spec = code_interp.get_spec()
    logger.info(f"Spec: {spec}")
    assert spec["type"] == "code_interpreter"

    # 3. Create Agent
    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
        system="You are a data analyst. Use the code interpreter to run Python code and compute results.",
        user="Solve the problem by writing and executing code.",
    )

    agent = Agent(
        name="CodeBot",
        role="Data Analyst",
        objective="Solve problems by writing and executing code.",
        llm=llm,
        prompt=prompt,
        tools=[code_interp],
        config=AgentConfig(execution_mode="standard"),
    )
    await agent.initialize()

    # 4. Execute tasks requiring code execution
    tasks = [
        {"id": "ci1", "objective": "Calculate the 20th Fibonacci number."},
        {"id": "ci2", "objective": "Generate a list of the first 10 prime numbers."},
    ]

    for task in tasks:
        logger.info(f"\nTask: {task['objective']}")
        logger.info("-" * 50)
        try:
            result = await agent.execute(task)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"Error: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Code Interpreter Example Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
