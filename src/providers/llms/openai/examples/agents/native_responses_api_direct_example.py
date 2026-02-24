"""
Example: Direct Responses API Access (Advanced)

Demonstrates using responses_call() for advanced users who need:
- Explicit previous_response_id control
- The include parameter for rich annotations (e.g., web search sources)
- Raw Responses API response format

Most users should just use Agent (which calls llm.call() internally).
This example is for power users who need low-level control.

Usage:
    python native_responses_api_direct_example.py
"""

import asyncio
import logging
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)
from dotenv import load_dotenv

load_dotenv()

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
    logger.info("Direct Responses API Example (Advanced)")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o", temperature=0.3)

    # ----- Example 1: Web search with rich annotations -----
    logger.info("\n--- Example 1: Web search with annotations ---")

    web_spec = OpenAITool.web_search().get_spec()
    logger.info(f"Web search spec: {web_spec}")

    try:
        response = await llm.responses_call(
            model="gpt-4o",
            input="What are the latest developments in quantum computing?",
            tools=[web_spec],
        )

        logger.info(f"Response ID: {response.id}")
        logger.info(f"Status: {response.status}")
        logger.info(f"Output items: {len(response.output)}")

        for item in response.output:
            item_type = getattr(item, "type", "unknown")
            if item_type == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", "") == "output_text":
                        logger.info(f"Text: {block.text[:200]}...")
            elif item_type == "web_search_call":
                logger.info(f"Web search executed (id={getattr(item, 'id', 'N/A')})")
            else:
                logger.info(f"Output type: {item_type}")

    except Exception as e:
        logger.error(f"Error: {e}")

    # ----- Example 2: Multi-turn with previous_response_id -----
    logger.info("\n--- Example 2: Multi-turn conversation ---")

    try:
        # Turn 1
        resp1 = await llm.responses_call(
            model="gpt-4o",
            input="What is 25 * 4?",
        )
        resp1_id = resp1.id
        logger.info(f"Turn 1 response ID: {resp1_id}")

        # Extract text from turn 1
        for item in resp1.output:
            if getattr(item, "type", "") == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", "") == "output_text":
                        logger.info(f"Turn 1 answer: {block.text}")

        # Turn 2 — continues from turn 1
        resp2 = await llm.responses_call(
            model="gpt-4o",
            input="Now double that result.",
            previous_response_id=resp1_id,
        )
        logger.info(f"Turn 2 response ID: {resp2.id}")

        for item in resp2.output:
            if getattr(item, "type", "") == "message":
                for block in getattr(item, "content", []):
                    if getattr(block, "type", "") == "output_text":
                        logger.info(f"Turn 2 answer: {block.text}")

    except Exception as e:
        logger.error(f"Error: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Direct Responses API Example Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
