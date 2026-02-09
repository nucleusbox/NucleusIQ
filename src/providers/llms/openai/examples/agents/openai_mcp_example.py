"""
Example: Using OpenAI MCP (Model Context Protocol) Tools

This demonstrates how to use OpenAI's MCP server tool to connect
to remote MCP servers and access their capabilities.
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
from nucleusiq_openai import OpenAITool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Demonstrate using OpenAI MCP tools.
    
    MCP (Model Context Protocol) allows connecting to remote servers
    that provide additional capabilities to the LLM.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.info("💡 Set it in your .env file or environment")
        return
    
    logger.info("=" * 60)
    logger.info("OpenAI MCP (Model Context Protocol) Example")
    logger.info("=" * 60)
    logger.info("\n💡 This example shows how to use MCP servers with OpenAI\n")
    
    # Step 1: Create LLM
    logger.info("1. Creating OpenAI LLM...")
    llm = BaseOpenAI(model_name="gpt-4o", temperature=0)
    logger.info("✅ LLM created")
    
    # Step 2: Create MCP tool
    logger.info("\n2. Creating MCP tool...")
    logger.info("   MCP servers provide additional capabilities to the LLM")
    logger.info("   Example: D&D dice rolling server")
    
    # Example 1: D&D Dice Rolling MCP Server (Remote MCP)
    logger.info("\n   Example 1: Remote MCP Server")
    mcp_tool = OpenAITool.mcp(
        server_label="dmcp",
        server_description="A Dungeons and Dragons MCP server to assist with dice rolling.",
        server_url="https://dmcp-server.deno.dev/sse",
        require_approval="never",  # Options: "never", "always", or dict
        # allowed_tools=["roll"],  # Optional: filter to specific tools
    )
    
    logger.info(f"   ✅ MCP tool created: {mcp_tool.name}")
    logger.info(f"      Server Label: dmcp")
    logger.info(f"      Server URL: https://dmcp-server.deno.dev/sse")
    
    # Example 2: With tool filtering (commented - uncomment to use)
    # logger.info("\n   Example 2: MCP with tool filtering")
    # mcp_filtered = OpenAITool.mcp(
    #     server_label="dmcp",
    #     server_description="D&D dice rolling server (filtered).",
    #     server_url="https://dmcp-server.deno.dev/sse",
    #     require_approval="never",
    #     allowed_tools=["roll"],  # Only import the "roll" tool
    # )
    # logger.info(f"   ✅ MCP tool created with filtering: {mcp_filtered.name}")
    
    # Example 3: OpenAI Connector (commented - needs OAuth token)
    # logger.info("\n   Example 3: OpenAI Connector (Google Calendar)")
    # calendar_connector = OpenAITool.connector(
    #     connector_id="connector_googlecalendar",
    #     server_label="google_calendar",
    #     server_description="Access Google Calendar events.",
    #     authorization="ya29.A0AS3H6...",  # OAuth token
    #     require_approval="never",
    # )
    # logger.info(f"   ✅ Connector created: {calendar_connector.name}")
    
    # Step 3: Create prompt
    logger.info("\n3. Creating prompt...")
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful assistant with access to a D&D dice rolling server. Use the MCP server to roll dice when requested.",
        user="Help users with dice rolling requests."
    )
    logger.info("✅ Prompt created")
    
    # Step 4: Create Agent with MCP tool
    logger.info("\n4. Creating Agent with MCP tool...")
    agent = Agent(
        name="MCPAgent",
        role="D&D Assistant",
        objective="Help users with dice rolling using the MCP server.",
        narrative="Has access to a D&D dice rolling MCP server.",
        llm=llm,
        prompt=prompt,
        tools=[mcp_tool],  # MCP tool
        config=AgentConfig(verbose=True)
    )
    logger.info("✅ Agent created")
    
    # Step 5: Initialize
    logger.info("\n5. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")
    
    # Step 6: Execute task
    logger.info("\n6. Executing task...")
    logger.info("=" * 60)
    logger.info("Task: Roll 2d4+1")
    logger.info("=" * 60)
    
    task = {
        "id": "task1",
        "objective": "Roll 2d4+1"
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
    logger.info("\n💡 Key points about MCP:")
    logger.info("   - MCP servers provide additional capabilities to the LLM")
    logger.info("   - Use OpenAITool.mcp() for remote MCP servers")
    logger.info("   - Use OpenAITool.connector() for OpenAI connectors")
    logger.info("\n📋 Parameters:")
    logger.info("   - server_label: Label/name for the server/connector")
    logger.info("   - server_description: What the server/connector does")
    logger.info("   - server_url: URL of remote MCP server (for remote MCP)")
    logger.info("   - connector_id: ID of OpenAI connector (for connectors)")
    logger.info("   - require_approval: 'never', 'always', or dict with tool names")
    logger.info("   - allowed_tools: Optional list to filter tools (reduces cost/latency)")
    logger.info("   - authorization: OAuth token for authenticated servers/connectors")
    logger.info("\n🔧 Features:")
    logger.info("   - LLM automatically discovers tools from MCP server")
    logger.info("   - Tools are handled natively by OpenAI - no execute() method")
    logger.info("   - Can filter tools with allowed_tools parameter")
    logger.info("   - Supports OAuth authentication")
    logger.info("\n📚 Available Connectors:")
    logger.info("   - connector_dropbox, connector_gmail")
    logger.info("   - connector_googlecalendar, connector_googledrive")
    logger.info("   - connector_microsoftteams")
    logger.info("   - connector_outlookcalendar, connector_outlookemail")
    logger.info("   - connector_sharepoint")
    logger.info("\n⚠️  Security Warning:")
    logger.info("   - Only use MCP servers you trust")
    logger.info("   - Malicious servers can exfiltrate sensitive data")
    logger.info("   - Review the server code and security before using")
    logger.info("   - Use require_approval='always' for sensitive operations")
    logger.info("   - Log and review data being shared with MCP servers")


if __name__ == "__main__":
    asyncio.run(main())

