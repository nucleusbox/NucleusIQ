"""
Example: All OpenAI Tool Types

This demonstrates how to create tools for all OpenAI-supported tool types:
1. Function calling (BaseTool)
2. Web search (native)
3. Code interpreter (native)
4. File search (native)
5. Image generation (native)
6. MCP servers (native)
7. Computer use (native)
"""

import asyncio
import logging
import os
import sys

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)
from dotenv import load_dotenv

# Load environment variables from .env file
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

# ============================================================================
# Example Usage
# ============================================================================


async def main():
    """Demonstrate all OpenAI tool types."""

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        return

    logger.info("=" * 60)
    logger.info("OpenAI All Tools Example")
    logger.info("=" * 60)
    logger.info("\n💡 Demonstrating all OpenAI tool types\n")

    # Step 1: Create LLM
    logger.info("1. Creating OpenAI LLM...")
    llm = BaseOpenAI(model_name="gpt-5-mini", temperature=0.7)
    logger.info("✅ LLM created")

    # Step 2: Create BaseTool (function calling)
    logger.info("\n2. Creating BaseTool (function calling)...")

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    calculator = BaseTool.from_function(add, description="Add two integers")
    logger.info(f"✅ BaseTool created: {calculator.name}")

    # Step 3: Create all OpenAI native tools using factory methods
    logger.info("\n3. Creating OpenAI native tools using factory methods...")
    tools = [
        calculator,  # BaseTool (function calling)
        OpenAITool.web_search(),  # Web search
        OpenAITool.code_interpreter(),  # Code interpreter
        OpenAITool.file_search(),  # File search
        OpenAITool.image_generation(),  # Image generation
        # OpenAITool.mcp(
        #     server_label="dmcp",
        #     server_description="A Dungeons and Dragons MCP server",
        #     server_url="https://dmcp-server.deno.dev/sse",
        #     require_approval="never",
        # ),  # MCP server (commented - needs config)
        # OpenAITool.computer_use(),        # Computer use (commented - advanced)
    ]

    logger.info("✅ Created tools:")
    for tool in tools:
        is_native = getattr(tool, "is_native", False)
        if is_native:
            tool_type_str = getattr(tool, "tool_type", "native")
            tool_type = f"OpenAI ({tool_type_str})"
        else:
            tool_type = "BaseTool"
        logger.info(f"   - {tool.name} ({tool_type})")

    # Step 4: Create prompt
    logger.info("\n4. Creating prompt...")
    prompt = PromptFactory.create_prompt(technique=PromptTechnique.ZERO_SHOT).configure(
        system="You are a helpful assistant with access to various tools.",
        user="Use tools when needed.",
    )
    logger.info("✅ Prompt created")

    # Step 5: Create Agent
    logger.info("\n5. Creating Agent...")
    agent = Agent(
        name="AllToolsAgent",
        role="Assistant",
        objective="Help users with various capabilities.",
        narrative="Has access to web search, code execution, file search, and more.",
        llm=llm,
        prompt=prompt,
        tools=tools,
        config=AgentConfig(verbose=True),
    )
    logger.info("✅ Agent created")

    # Step 6: Initialize
    logger.info("\n6. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("\n✅ All OpenAI tool types supported:")
    logger.info("   1. Function calling (BaseTool.from_function())")
    logger.info("   2. Web search (OpenAITool.web_search())")
    logger.info("   3. Code interpreter (OpenAITool.code_interpreter())")
    logger.info("   4. File search (OpenAITool.file_search())")
    logger.info("   5. Image generation (OpenAITool.image_generation())")
    logger.info("   6. MCP servers (OpenAITool.mcp())")
    logger.info("   7. Computer use (OpenAITool.computer_use())")
    logger.info("\n💡 Use OpenAITool factory methods - no need to create subclasses!")


if __name__ == "__main__":
    asyncio.run(main())
