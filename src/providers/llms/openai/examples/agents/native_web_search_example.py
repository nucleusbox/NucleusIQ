"""
Example: Web Search with OpenAI Responses API

Demonstrates using OpenAI's built-in web search tool.
The routing is automatic — call() detects the native tool and
uses the Responses API behind the scenes.

Usage:
    python native_web_search_example.py
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
    logger.info("Native Web Search Example")
    logger.info("=" * 60)

    # 1. Create LLM
    llm = BaseOpenAI(model_name="gpt-4o", temperature=0.3)

    # 2. Create native web search tool (one-liner!)
    web_search = OpenAITool.web_search()
    logger.info(f"Tool: {web_search.name}  |  type: {web_search.tool_type}  |  native: {web_search.is_native}")

    # 3. Verify spec is in Responses API format
    spec = web_search.get_spec()
    logger.info(f"Spec: {spec}")
    assert spec["type"] == "web_search_preview", "Spec should use web_search_preview"

    # 4. Create Agent
    prompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
        system="You are a helpful assistant. Use web search to find current information.",
        user="Answer the user's question with up-to-date facts.",
    )

    agent = Agent(
        name="WebSearchBot",
        role="Researcher",
        objective="Find current information using web search.",
        llm=llm,
        prompt=prompt,
        tools=[web_search],
        config=AgentConfig(execution_mode="standard"),
    )
    await agent.initialize()

    # 5. Execute — the Agent automatically routes through Responses API
    tasks = [
        {"id": "ws1", "objective": "What is the current population of Tokyo?"},
        {"id": "ws2", "objective": "Who won the latest Nobel Prize in Physics?"},
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
    logger.info("Web Search Example Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
