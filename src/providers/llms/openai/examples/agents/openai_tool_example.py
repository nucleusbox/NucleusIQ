"""
Example: Using OpenAI Tools (Factory Pattern)

This demonstrates how to use OpenAITool factory methods to create
all OpenAI native tools and use them in an Agent.
"""

import os
import sys
import asyncio
import logging

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq_openai import BaseOpenAI
from nucleusiq.tools import BaseTool
from nucleusiq_openai import OpenAITool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Demonstrate using OpenAI tools via factory pattern.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.info("💡 Set it in your .env file or environment")
        return
    
    logger.info("=" * 60)
    logger.info("OpenAI Tools Example (Factory Pattern)")
    logger.info("=" * 60)
    logger.info("\n💡 Using OpenAITool as a factory class\n")
    
    # Step 1: Create LLM
    logger.info("1. Creating OpenAI LLM...")
    llm = BaseOpenAI(model_name="gpt-4o", temperature=0)
    logger.info("✅ LLM created")
    
    # Step 2: Create a BaseTool (function calling)
    logger.info("\n2. Creating BaseTool (function calling)...")
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    calculator_tool = BaseTool.from_function(add, description="Add two integers")
    logger.info(f"✅ BaseTool created: {calculator_tool.name}")
    
    # Step 3: Create OpenAI native tools using factory methods
    logger.info("\n3. Creating OpenAI native tools using factory methods...")
    
    # Web search
    web_search = OpenAITool.web_search()
    logger.info(f"   ✅ {web_search.name} (via OpenAITool.web_search())")
    
    # Code interpreter
    code_interpreter = OpenAITool.code_interpreter()
    logger.info(f"   ✅ {code_interpreter.name} (via OpenAITool.code_interpreter())")
    
    # File search (with optional vector store IDs)
    file_search = OpenAITool.file_search()  # Can pass vector_store_ids=["vs_123"]
    logger.info(f"   ✅ {file_search.name} (via OpenAITool.file_search())")
    
    # Image generation
    image_gen = OpenAITool.image_generation()
    logger.info(f"   ✅ {image_gen.name} (via OpenAITool.image_generation())")
    
    # MCP server (commented - needs actual server)
    # mcp_tool = OpenAITool.mcp(
    #     server_label="dmcp",
    #     server_description="A Dungeons and Dragons MCP server to assist with dice rolling.",
    #     server_url="https://dmcp-server.deno.dev/sse",
    #     require_approval="never",
    # )
    # logger.info(f"   ✅ {mcp_tool.name} (via OpenAITool.mcp())")
    
    # Computer use
    computer_use = OpenAITool.computer_use()
    logger.info(f"   ✅ {computer_use.name} (via OpenAITool.computer_use())")
    
    # Step 4: Create prompt
    logger.info("\n4. Creating prompt...")
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful assistant with access to various tools including web search, code execution, and calculations.",
        user="Use tools when needed to answer questions accurately."
    )
    logger.info("✅ Prompt created")
    
    # Step 5: Create Agent with all tools
    logger.info("\n5. Creating Agent with BaseTool + OpenAI tools...")
    agent = Agent(
        name="OpenAIAgent",
        role="Assistant",
        objective="Help users with accurate information using available tools.",
        narrative="Can search the web, execute code, perform calculations, and more.",
        llm=llm,
        prompt=prompt,
        tools=[
            calculator_tool,      # BaseTool (function calling)
            web_search,           # OpenAITool.web_search()
            code_interpreter,     # OpenAITool.code_interpreter()
            # file_search,        # OpenAITool.file_search()
            # image_gen,          # OpenAITool.image_generation()
            # computer_use,       # OpenAITool.computer_use()
        ],
        config=AgentConfig(verbose=True)
    )
    logger.info("✅ Agent created")
    
    # Step 6: Initialize
    logger.info("\n6. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")
    
    # Step 7: Execute task
    logger.info("\n7. Executing task...")
    logger.info("=" * 60)
    logger.info("Task: What is 15 + 27?")
    logger.info("=" * 60)
    
    task = {
        "id": "task1",
        "objective": "What is 15 + 27? Use the add function."
    }
    
    try:
        result = await agent.execute(task)
        logger.info(f"\n✅ Result: {result}")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed!")
    logger.info("=" * 60)
    logger.info("\n💡 Key points:")
    logger.info("   - Use OpenAITool as a factory class (no need to create subclasses)")
    logger.info("   - OpenAITool.web_search() - Web search")
    logger.info("   - OpenAITool.code_interpreter() - Code execution")
    logger.info("   - OpenAITool.file_search(vector_store_ids=[...]) - File search")
    logger.info("   - OpenAITool.image_generation() - Image generation")
    logger.info("   - OpenAITool.mcp(server_label='...', server_url='...') - MCP servers")
    logger.info("   - OpenAITool.computer_use() - Computer control")
    logger.info("   - Mix BaseTool and OpenAITool in Agent!")
    logger.info("\n📚 All OpenAI tool types supported:")
    logger.info("   1. Function calling (BaseTool)")
    logger.info("   2. Web search (OpenAITool.web_search())")
    logger.info("   3. Code interpreter (OpenAITool.code_interpreter())")
    logger.info("   4. File search (OpenAITool.file_search())")
    logger.info("   5. Image generation (OpenAITool.image_generation())")
    logger.info("   6. MCP servers (OpenAITool.mcp())")
    logger.info("   7. Computer use (OpenAITool.computer_use())")


if __name__ == "__main__":
    asyncio.run(main())
