"""
Example: Using OpenAI Connectors

This demonstrates how to use OpenAI's built-in connectors (like Google Calendar,
Gmail, Dropbox, etc.) via the MCP tool interface.
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
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
from nucleusiq.providers.llms.openai.tools import OpenAITool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Demonstrate using OpenAI connectors.
    
    Connectors are OpenAI-maintained MCP wrappers for popular services
    like Google Workspace, Dropbox, Microsoft Teams, etc.
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.info("💡 Set it in your .env file or environment")
        return
    
    logger.info("=" * 60)
    logger.info("OpenAI Connectors Example")
    logger.info("=" * 60)
    logger.info("\n💡 This example shows how to use OpenAI connectors\n")
    
    # Step 1: Create LLM
    logger.info("1. Creating OpenAI LLM...")
    llm = BaseOpenAI(model_name="gpt-4o", temperature=0)
    logger.info("✅ LLM created")
    
    # Step 2: Create connector tool
    logger.info("\n2. Creating connector tool...")
    logger.info("   Connectors require OAuth access tokens")
    logger.info("   For testing, you can use OAuth 2.0 Playground")
    
    # Example: Google Calendar Connector
    # You need to get an OAuth token first
    oauth_token = os.getenv("GOOGLE_OAUTH_TOKEN")
    
    if not oauth_token:
        logger.warning("⚠️  GOOGLE_OAUTH_TOKEN not set - using placeholder")
        logger.info("   To get a token:")
        logger.info("   1. Go to https://developers.google.com/oauthplayground/")
        logger.info("   2. Select Google Calendar API scope")
        logger.info("   3. Authorize and get access token")
        logger.info("   4. Set GOOGLE_OAUTH_TOKEN environment variable")
        oauth_token = "ya29.PLACEHOLDER_TOKEN"  # Placeholder
    
    # Method 1: Using connector() convenience method
    calendar_connector = OpenAITool.connector(
        connector_id="connector_googlecalendar",
        server_label="google_calendar",
        server_description="Access Google Calendar events and manage calendar.",
        authorization=oauth_token,
        require_approval="never",
        # allowed_tools=["search_events"],  # Optional: filter tools
    )
    
    logger.info(f"✅ Connector created: {calendar_connector.name}")
    logger.info(f"   Connector ID: connector_googlecalendar")
    logger.info(f"   Server Label: google_calendar")
    
    # Method 2: Using mcp() with connector_id (same result)
    # calendar_connector = OpenAITool.mcp(
    #     server_label="google_calendar",
    #     server_description="Access Google Calendar events.",
    #     connector_id="connector_googlecalendar",
    #     authorization=oauth_token,
    #     require_approval="never",
    # )
    
    # Step 3: Create prompt
    logger.info("\n3. Creating prompt...")
    prompt = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful assistant with access to Google Calendar. Use the calendar connector to help users manage their schedule.",
        user="Help users with calendar-related requests."
    )
    logger.info("✅ Prompt created")
    
    # Step 4: Create Agent with connector
    logger.info("\n4. Creating Agent with connector...")
    agent = Agent(
        name="CalendarAgent",
        role="Calendar Assistant",
        objective="Help users manage their Google Calendar.",
        narrative="Has access to Google Calendar via OpenAI connector.",
        llm=llm,
        prompt=prompt,
        tools=[calendar_connector],
        config=AgentConfig(verbose=True)
    )
    logger.info("✅ Agent created")
    
    # Step 5: Initialize
    logger.info("\n5. Initializing agent...")
    await agent.initialize()
    logger.info(f"✅ Agent initialized (state: {agent.state})")
    
    # Step 6: Execute task (only if token is valid)
    if oauth_token and oauth_token != "ya29.PLACEHOLDER_TOKEN":
        logger.info("\n6. Executing task...")
        logger.info("=" * 60)
        logger.info("Task: What's on my calendar today?")
        logger.info("=" * 60)
        
        task = {
            "id": "task1",
            "objective": "What's on my Google Calendar for today?"
        }
        
        try:
            result = await agent.execute(task)
            logger.info(f"\n✅ Result: {result}")
        except Exception as e:
            logger.error(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("\n⚠️  Skipping execution - OAuth token not configured")
        logger.info("   Set GOOGLE_OAUTH_TOKEN environment variable to test")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed!")
    logger.info("=" * 60)
    logger.info("\n💡 Available Connectors:")
    logger.info("   - connector_dropbox")
    logger.info("   - connector_gmail")
    logger.info("   - connector_googlecalendar")
    logger.info("   - connector_googledrive")
    logger.info("   - connector_microsoftteams")
    logger.info("   - connector_outlookcalendar")
    logger.info("   - connector_outlookemail")
    logger.info("   - connector_sharepoint")
    logger.info("\n📝 Usage:")
    logger.info("   # Method 1: Using connector() convenience method")
    logger.info("   connector = OpenAITool.connector(")
    logger.info("       connector_id='connector_googlecalendar',")
    logger.info("       server_label='google_calendar',")
    logger.info("       server_description='Access Google Calendar.',")
    logger.info("       authorization='ya29.ACCESS_TOKEN',")
    logger.info("       require_approval='never',")
    logger.info("   )")
    logger.info("\n   # Method 2: Using mcp() with connector_id")
    logger.info("   connector = OpenAITool.mcp(")
    logger.info("       server_label='google_calendar',")
    logger.info("       server_description='Access Google Calendar.',")
    logger.info("       connector_id='connector_googlecalendar',")
    logger.info("       authorization='ya29.ACCESS_TOKEN',")
    logger.info("   )")


if __name__ == "__main__":
    asyncio.run(main())

