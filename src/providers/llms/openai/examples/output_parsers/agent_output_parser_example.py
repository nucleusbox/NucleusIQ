#!/usr/bin/env python3
"""
NucleusIQ Structured Output Examples

Demonstrates how to get validated structured responses from agents
using Pydantic models, dataclasses, and TypedDict.

NucleusIQ Philosophy:
- Simple: Just pass your schema, NucleusIQ handles the rest
- Intelligent: Auto-selects best method based on model
- Consistent: Same API regardless of underlying implementation

Basic Usage:
    agent = Agent(response_format=MySchema)
    result = await agent.execute(task)
    data = result["output"]  # Validated instance
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Load environment variables
load_dotenv()

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config.agent_config import AgentConfig, ExecutionMode
from nucleusiq.agents.structured_output import OutputMode, OutputSchema
from nucleusiq.agents.task import Task
from nucleusiq_openai import BaseOpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================


class ContactInfo(BaseModel):
    """Contact information extracted from text."""

    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str | None = Field(default=None, description="Phone number")


class ProductReview(BaseModel):
    """Analyzed product review."""

    rating: int | None = Field(description="Rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "negative", "neutral"]
    key_points: List[str] = Field(description="Main points from review")


@dataclass
class MeetingAction:
    """Action item from a meeting."""

    task: str
    assignee: str
    priority: str  # low, medium, high


class ServerConfig(TypedDict):
    """Server configuration."""

    host: str
    port: int
    debug: bool


# ============================================================================
# EXAMPLES
# ============================================================================


async def example_simple():
    """Simplest usage - just pass the schema."""
    logger.info("=" * 60)
    logger.info("Example 1: Simple Usage (Just Pass Schema)")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    # Just pass the schema - that's it!
    agent = Agent(
        name="ContactExtractor",
        role="Information Extractor",
        objective="Extract contact information",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        response_format=ContactInfo,  # <-- Just pass the Pydantic model
    )

    task = Task(
        id="contact-1", objective="Extract: John Smith, john.smith@email.com, 555-1234"
    )

    result = await agent.execute(task)
    contact = result["output"]

    logger.info(f"Type: {type(contact).__name__}")
    logger.info(f"Name: {contact.name}")
    logger.info(f"Email: {contact.email}")
    logger.info(f"Phone: {contact.phone}")
    return True


async def example_explicit_config():
    """Explicit configuration with OutputSchema."""
    logger.info("=" * 60)
    logger.info("Example 2: Explicit Config (OutputSchema)")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    # Explicit configuration for more control
    agent = Agent(
        name="ReviewAnalyzer",
        role="Product Analyst",
        objective="Analyze product reviews",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        response_format=OutputSchema(
            schema=ProductReview,
            mode=OutputMode.NATIVE,  # Use native structured output
            strict=True,  # Strict schema adherence
            retry_on_error=True,  # Retry on validation failure
            max_retries=2,
        ),
    )

    task = Task(
        id="review-1",
        objective="Analyze: 'Great product! 5 stars. Fast shipping, great quality, but expensive.'",
    )

    result = await agent.execute(task)
    review = result["output"]

    logger.info(f"Rating: {review.rating}/5")
    logger.info(f"Sentiment: {review.sentiment}")
    logger.info(f"Key Points: {review.key_points}")
    return True


async def example_dataclass():
    """Using Python dataclass as schema."""
    logger.info("=" * 60)
    logger.info("Example 3: Dataclass Schema")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    agent = Agent(
        name="MeetingParser",
        role="Meeting Assistant",
        objective="Extract action items from meeting notes",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        response_format=MeetingAction,
    )

    task = Task(
        id="meeting-1",
        objective="Action: Sarah needs to update the project timeline by Friday - high priority",
    )

    result = await agent.execute(task)
    action = result["output"]

    logger.info(f"Type: {type(action).__name__}")
    logger.info(f"Task: {action.task}")
    logger.info(f"Assignee: {action.assignee}")
    logger.info(f"Priority: {action.priority}")
    return True


async def example_typeddict():
    """Using TypedDict as schema."""
    logger.info("=" * 60)
    logger.info("Example 4: TypedDict Schema")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    agent = Agent(
        name="ConfigParser",
        role="Config Parser",
        objective="Parse server configuration",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        response_format=ServerConfig,
    )

    task = Task(
        id="config-1",
        objective="Parse: server at api.example.com on port 8080, debug mode on",
    )

    result = await agent.execute(task)
    config = result["output"]

    logger.info(f"Type: {type(config).__name__}")
    logger.info(f"Host: {config['host']}")
    logger.info(f"Port: {config['port']}")
    logger.info(f"Debug: {config['debug']}")
    return True


async def example_no_schema():
    """Without response_format - returns raw text."""
    logger.info("=" * 60)
    logger.info("Example 5: No Schema (Raw Text)")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    # No response_format = raw text response
    agent = Agent(
        name="ChatBot",
        role="Assistant",
        objective="Answer questions",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    task = Task(id="chat-1", objective="What is Python?")

    result = await agent.execute(task)

    logger.info(f"Type: {type(result).__name__}")
    logger.info(f"Response: {result[:100]}...")
    return isinstance(result, str)


# ============================================================================
# MAIN
# ============================================================================


async def main():
    """Run all examples."""
    logger.info("NucleusIQ Structured Output Examples")
    logger.info("=" * 60)
    logger.info("""
NucleusIQ makes structured output simple:

1. Just pass your schema:
   agent = Agent(response_format=MyModel)

2. Or configure explicitly:
   agent = Agent(response_format=OutputSchema(
       schema=MyModel,
       mode=OutputMode.NATIVE,
       strict=True
   ))

Supported schemas:
- Pydantic models (BaseModel)
- Python dataclasses
- TypedDict
- JSON Schema (dict)
""")

    results = {}

    try:
        results["Simple (Pydantic)"] = await example_simple()
        results["Explicit Config"] = await example_explicit_config()
        results["Dataclass"] = await example_dataclass()
        results["TypedDict"] = await example_typeddict()
        results["No Schema"] = await example_no_schema()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        logger.info(f"  {name}: {'PASS' if passed else 'FAIL'}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    logger.info(f"\nOverall: {passed}/{total} passed")


if __name__ == "__main__":
    asyncio.run(main())
