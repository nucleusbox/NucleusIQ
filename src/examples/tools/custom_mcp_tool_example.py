"""
Example: Custom MCP Tool (BaseTool) with OpenAI LLM

This demonstrates how to create a custom MCP client tool as a BaseTool
that works with OpenAI LLM, without using OpenAI's native MCP tool.

Scenario:
- Using OpenAI LLM
- Creating a custom MCP client tool (BaseTool)
- This tool implements MCP protocol client-side
- OpenAI LLM uses function calling to invoke this custom tool
- The tool connects to MCP server, calls tools, and returns results
"""

import os
import sys
import asyncio
import logging
import json
from typing import Any, Dict, List, Optional

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
from nucleusiq.core.tools import BaseTool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomMCPTool(BaseTool):
    """
    Custom MCP client tool that implements MCP protocol client-side.
    
    This is a BaseTool, so it works with ANY LLM via function calling.
    It's not using OpenAI's native MCP tool - it's a custom implementation.
    
    The tool:
    1. Connects to an MCP server
    2. Lists available tools from the server
    3. Calls MCP tools on the server
    4. Returns results to the LLM
    """
    
    def __init__(
        self,
        server_url: str,
        server_label: str = "custom_mcp",
        authorization: Optional[str] = None,
    ):
        """
        Initialize custom MCP tool.
        
        Args:
            server_url: URL of the MCP server
            server_label: Label for the server
            authorization: Optional OAuth token
        """
        super().__init__(
            name=f"mcp_client_{server_label}",
            description=f"Custom MCP client tool that connects to {server_url} and calls MCP tools",
        )
        self.server_url = server_url
        self.server_label = server_label
        self.authorization = authorization
        self._available_tools: Optional[List[Dict[str, Any]]] = None
    
    async def initialize(self) -> None:
        """Initialize by fetching available tools from MCP server."""
        logger.info(f"Initializing custom MCP tool for {self.server_url}")
        # In a real implementation, you would:
        # 1. Connect to MCP server
        # 2. Call mcp_list_tools
        # 3. Store available tools
        # For this example, we'll simulate it
        self._available_tools = [
            {
                "name": "roll",
                "description": "Roll dice",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "diceRollExpression": {"type": "string"}
                    },
                    "required": ["diceRollExpression"]
                }
            }
        ]
        logger.info(f"Found {len(self._available_tools)} tools from MCP server")
    
    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Execute an MCP tool on the remote server.
        
        Args:
            tool_name: Name of the MCP tool to call
            **kwargs: Arguments for the tool
        
        Returns:
            Result from the MCP tool
        """
        logger.info(f"Calling MCP tool '{tool_name}' with args: {kwargs}")
        
        # In a real implementation, you would:
        # 1. Make HTTP request to MCP server
        # 2. Call the tool via MCP protocol
        # 3. Return the result
        
        # Simulated response for this example
        if tool_name == "roll":
            dice_expr = kwargs.get("diceRollExpression", "")
            # Simulate dice roll result
            return f"Dice roll result for '{dice_expr}': 7"
        
        raise ValueError(f"Unknown MCP tool: {tool_name}")
    
    def get_spec(self) -> Dict[str, Any]:
        """
        Return tool spec for function calling.
        
        This creates a function that the LLM can call.
        The function takes tool_name and arguments, then calls the MCP server.
        """
        # Get available tools from MCP server
        if not self._available_tools:
            # If not initialized, return a generic spec
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the MCP tool to call"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the MCP tool"
                        }
                    },
                    "required": ["tool_name", "arguments"]
                }
            }
        
        # For each MCP tool, create a separate function spec
        # In this example, we'll create one function that can call any tool
        # In a more advanced implementation, you could create separate BaseTool
        # instances for each MCP tool
        
        tool_descriptions = [f"- {t['name']}: {t['description']}" for t in self._available_tools]
        
        return {
            "name": self.name,
            "description": f"{self.description}\n\nAvailable MCP tools:\n" + "\n".join(tool_descriptions),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the MCP tool to call",
                        "enum": [t["name"] for t in self._available_tools]
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments for the MCP tool (varies by tool)",
                        "additionalProperties": True
                    }
                },
                "required": ["tool_name", "arguments"]
            }
        }


async def main():
    """
    Demonstrate using a custom MCP tool (BaseTool) with OpenAI LLM.
    
    This shows that you can:
    - Use OpenAI LLM
    - Create custom MCP client as BaseTool
    - Use function calling (not native MCP tool)
    - Still connect to MCP servers
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.info("💡 Set it in your .env file or environment")
        return
    
    logger.info("=" * 60)
    logger.info("Custom MCP Tool (BaseTool) with OpenAI LLM")
    logger.info("=" * 60)
    logger.info("\n💡 This demonstrates using a custom MCP client tool")
    logger.info("   with OpenAI LLM via function calling\n")
    
    # Step 1: Create OpenAI LLM
    logger.info("1. Creating OpenAI LLM...")
    llm = BaseOpenAI(model_name="gpt-4o", temperature=0)
    logger.info("✅ LLM created")
    
    # Step 2: Create custom MCP tool (BaseTool, not native OpenAI tool)
    logger.info("\n2. Creating custom MCP tool (BaseTool)...")
    logger.info("   This is NOT using OpenAI's native MCP tool")
    logger.info("   This is a custom BaseTool that implements MCP client")
    
    custom_mcp = CustomMCPTool(
        server_url="https://dmcp-server.deno.dev/sse",
        server_label="dmcp",
    )
    
    # Initialize to fetch available tools
    await custom_mcp.initialize()
    
    logger.info(f"✅ Custom MCP tool created: {custom_mcp.name}")
    logger.info(f"   Server URL: {custom_mcp.server_url}")
    logger.info(f"   Available tools: {len(custom_mcp._available_tools or [])}")
    
    # Step 3: Create prompt
    logger.info("\n3. Creating prompt...")
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful assistant with access to a custom MCP client tool. Use the tool to call MCP server functions.",
        user="Help users by calling MCP tools when needed."
    )
    logger.info("✅ Prompt created")
    
    # Step 4: Create Agent with custom MCP tool
    logger.info("\n4. Creating Agent with custom MCP tool...")
    logger.info("   Using OpenAI LLM + Custom MCP Tool (BaseTool)")
    logger.info("   NOT using OpenAI's native MCP tool")
    
    agent = Agent(
        name="CustomMCPAgent",
        role="MCP Assistant",
        objective="Help users by calling MCP tools via custom client.",
        narrative="Has access to a custom MCP client tool that connects to MCP servers.",
        llm=llm,
        prompt=prompt,
        tools=[custom_mcp],  # Custom BaseTool, not OpenAITool.mcp()
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
    logger.info("Task: Roll 2d4+1 using custom MCP tool")
    logger.info("=" * 60)
    
    task = {
        "id": "task1",
        "objective": "Roll 2d4+1 using the MCP client tool"
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
    logger.info("   ✅ Custom MCP tool works as BaseTool")
    logger.info("   ✅ OpenAI LLM uses function calling (not native MCP)")
    logger.info("   ✅ Tool connects to MCP server client-side")
    logger.info("   ✅ You have full control over MCP protocol implementation")
    logger.info("\n📊 Comparison:")
    logger.info("   Native OpenAI MCP Tool:")
    logger.info("     - OpenAI API handles MCP protocol")
    logger.info("     - No client-side code needed")
    logger.info("     - Less control, simpler")
    logger.info("   Custom MCP Tool (BaseTool):")
    logger.info("     - You implement MCP client")
    logger.info("     - Full control over protocol")
    logger.info("     - Works with any LLM (not just OpenAI)")
    logger.info("     - More flexible, more code")


if __name__ == "__main__":
    asyncio.run(main())

