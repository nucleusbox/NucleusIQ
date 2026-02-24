"""
Example: Mixed Tools — Native + Custom Function Calling

Demonstrates using BOTH native OpenAI tools (web_search, code_interpreter)
AND custom function-calling tools (BaseTool) in the same Agent.

The routing is fully automatic:
- call() detects native tools → uses Responses API
- Responses API handles native tools server-side
- If a custom function_call is returned, the Agent executes it locally
- Results flow back seamlessly

Usage:
    python native_mixed_tools_example.py
"""

import asyncio
import logging
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)
from dotenv import load_dotenv

load_dotenv()

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.tools import BaseTool
from nucleusiq_openai import BaseOpenAI, OpenAITool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set — skipping.")
        return

    logger.info("=" * 60)
    logger.info("Mixed Tools Example (Native + Custom)")
    logger.info("=" * 60)

    # 1. Create LLM
    llm = BaseOpenAI(model_name="gpt-4o", temperature=0)

    # 2. Create custom function-calling tools
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    add_tool = BaseTool.from_function(add, description="Add two integers")
    multiply_tool = BaseTool.from_function(
        multiply, description="Multiply two integers"
    )

    # 3. Create native OpenAI tools
    web_search = OpenAITool.web_search()
    code_interp = OpenAITool.code_interpreter()

    # 4. Mix them all together
    all_tools = [add_tool, multiply_tool, web_search, code_interp]

    logger.info("Tools:")
    for t in all_tools:
        is_native = getattr(t, "is_native", False)
        logger.info(f"  - {t.name:20s}  native={is_native}")

    # 5. Verify routing detection
    tool_specs = llm.convert_tool_specs(all_tools)
    has_native = llm._has_native_tools(tool_specs)
    logger.info(f"\nHas native tools: {has_native}  →  Will use Responses API")

    # 6. Create Agent
    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
        system=(
            "You are a versatile assistant with access to:\n"
            "- add(a, b): Add two numbers\n"
            "- multiply(a, b): Multiply two numbers\n"
            "- web_search: Search the web for current information\n"
            "- code_interpreter: Execute Python code\n"
            "Use the appropriate tool for each task."
        ),
        user="Help the user with their request.",
    )

    agent = Agent(
        name="MixedToolsBot",
        role="Versatile Assistant",
        objective="Handle diverse tasks using the right tool.",
        llm=llm,
        prompt=prompt,
        tools=all_tools,
        config=AgentConfig(execution_mode="standard"),
    )
    await agent.initialize()

    # 7. Execute tasks — some use custom tools, some use native tools
    tasks = [
        {"id": "m1", "objective": "What is 42 + 58? Use the add tool."},
        {"id": "m2", "objective": "What is today's top news headline? Use web search."},
        {
            "id": "m3",
            "objective": "Calculate the factorial of 10 using code interpreter.",
        },
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
    logger.info("Mixed Tools Example Complete")
    logger.info("=" * 60)
    logger.info("\nKey takeaway: Native + Custom tools work seamlessly together!")
    logger.info("The Agent doesn't know or care which API backend is used.")


if __name__ == "__main__":
    asyncio.run(main())
