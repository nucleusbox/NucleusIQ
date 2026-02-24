"""
OpenAI Structured Output Examples

Comprehensive examples showing all structured output features with OpenAI:
- AUTO mode (recommended)
- NATIVE mode (explicit)
- Pydantic models
- Dataclasses
- TypedDict
- Error handling and retries
- Strict vs non-strict modes

Run with: python src/examples/agents/openai_structured_output_examples.py

Requires OPENAI_API_KEY environment variable.
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field

# Load environment variables (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.structured_output import OutputMode, OutputSchema
from nucleusiq.agents.task import Task
from nucleusiq_openai import BaseOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Schema Definitions
# ============================================================================


# Pydantic Model
class Person(BaseModel):
    """Person information."""

    name: str = Field(description="Full name")
    age: int = Field(description="Age in years", ge=0, le=150)
    email: str = Field(description="Email address")
    city: str = Field(description="City of residence")


# Pydantic Model with nested structure
class Company(BaseModel):
    """Company information."""

    name: str = Field(description="Company name")
    employees: int = Field(description="Number of employees", ge=0)
    founded: int = Field(description="Year founded")
    headquarters: Person = Field(description="CEO information")


# Dataclass
@dataclass
class Product:
    """Product information."""

    name: str
    price: float
    category: str
    in_stock: bool


# TypedDict
class WeatherInfo(TypedDict):
    """Weather information."""

    location: str
    temperature: float
    condition: str
    humidity: int


# ============================================================================
# Example 1: AUTO Mode with Pydantic (Recommended)
# ============================================================================


async def example_auto_mode_pydantic():
    """Example: AUTO mode with Pydantic model (recommended approach)."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: AUTO Mode with Pydantic Model (Recommended)")
    logger.info("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return

    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    # Just pass the schema - AUTO mode handles everything
    agent = Agent(
        name="Extractor",
        role="Data Extractor",
        objective="Extract structured data",
        llm=llm,
        response_format=Person,  # AUTO mode → NATIVE for OpenAI
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()

    task = Task(
        id="extract1",
        objective="Extract person info: John Doe, 30 years old, john@example.com, lives in New York",
    )

    logger.info(f"Task: {task.objective}")
    logger.info("Using AUTO mode - framework selects NATIVE for OpenAI")

    result = await agent.execute(task)

    if isinstance(result, dict) and "output" in result:
        person = result["output"]
        logger.info("\n✅ Extracted Person:")
        logger.info(f"   Name: {person.name}")
        logger.info(f"   Age: {person.age}")
        logger.info(f"   Email: {person.email}")
        logger.info(f"   City: {person.city}")
        logger.info(f"\n   Mode used: {result.get('mode', 'unknown')}")
        logger.info(f"   Schema: {result.get('schema', 'unknown')}")


# ============================================================================
# Example 2: Explicit NATIVE Mode with Strict Validation
# ============================================================================


async def example_native_mode_strict():
    """Example: Explicit NATIVE mode with strict validation."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: NATIVE Mode with Strict Validation")
    logger.info("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return

    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    # Explicit NATIVE mode with strict=True
    agent = Agent(
        name="StrictExtractor",
        role="Strict Data Extractor",
        objective="Extract data with strict validation",
        llm=llm,
        response_format=OutputSchema(
            schema=Person,
            mode=OutputMode.NATIVE,
            strict=True,  # OpenAI will enforce strict schema adherence
            retry_on_error=True,
            max_retries=2,
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()

    task = Task(
        id="extract2",
        objective="Extract: Jane Smith, 25, jane@example.com, San Francisco",
    )

    logger.info(f"Task: {task.objective}")
    logger.info("Using NATIVE mode with strict=True")
    logger.info("OpenAI will enforce that all required fields are present")

    result = await agent.execute(task)

    if isinstance(result, dict) and "output" in result:
        person = result["output"]
        logger.info("\n✅ Strict Validation Result:")
        logger.info(f"   {person}")


# ============================================================================
# Example 3: Nested Pydantic Models
# ============================================================================


async def example_nested_pydantic():
    """Example: Nested Pydantic models."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Nested Pydantic Models")
    logger.info("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return

    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    agent = Agent(
        name="CompanyExtractor",
        role="Company Data Extractor",
        objective="Extract company information",
        llm=llm,
        response_format=Company,  # Nested structure
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()

    task = Task(
        id="extract3",
        objective="Extract company: TechCorp, 500 employees, founded 2010, CEO: Alice Johnson, 45, alice@techcorp.com, Seattle",
    )

    logger.info(f"Task: {task.objective}")
    logger.info("Extracting nested structure (Company with Person CEO)")

    result = await agent.execute(task)

    if isinstance(result, dict) and "output" in result:
        company = result["output"]
        logger.info("\n✅ Extracted Company:")
        logger.info(f"   Name: {company.name}")
        logger.info(f"   Employees: {company.employees}")
        logger.info(f"   Founded: {company.founded}")
        logger.info(f"   CEO Name: {company.headquarters.name}")
        logger.info(f"   CEO Age: {company.headquarters.age}")
        logger.info(f"   CEO Email: {company.headquarters.email}")
        logger.info(f"   CEO City: {company.headquarters.city}")


# ============================================================================
# Example 4: Dataclass Schema
# ============================================================================


async def example_dataclass_schema():
    """Example: Using dataclass as schema."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Dataclass Schema")
    logger.info("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return

    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    agent = Agent(
        name="ProductExtractor",
        role="Product Data Extractor",
        objective="Extract product information",
        llm=llm,
        response_format=Product,  # Dataclass
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()

    task = Task(
        id="extract4",
        objective="Extract product: Laptop, $999.99, Electronics, in stock",
    )

    logger.info(f"Task: {task.objective}")
    logger.info("Using dataclass as schema")

    result = await agent.execute(task)

    if isinstance(result, dict) and "output" in result:
        product = result["output"]
        logger.info("\n✅ Extracted Product:")
        logger.info(f"   Name: {product.name}")
        logger.info(f"   Price: ${product.price}")
        logger.info(f"   Category: {product.category}")
        logger.info(f"   In Stock: {product.in_stock}")


# ============================================================================
# Example 5: TypedDict Schema
# ============================================================================


async def example_typeddict_schema():
    """Example: Using TypedDict as schema."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: TypedDict Schema")
    logger.info("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return

    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    agent = Agent(
        name="WeatherExtractor",
        role="Weather Data Extractor",
        objective="Extract weather information",
        llm=llm,
        response_format=WeatherInfo,  # TypedDict
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()

    task = Task(
        id="extract5",
        objective="Extract weather: New York, 72 degrees, sunny, 60% humidity",
    )

    logger.info(f"Task: {task.objective}")
    logger.info("Using TypedDict as schema")

    result = await agent.execute(task)

    if isinstance(result, dict) and "output" in result:
        weather = result["output"]
        logger.info("\n✅ Extracted Weather:")
        logger.info(f"   Location: {weather['location']}")
        logger.info(f"   Temperature: {weather['temperature']}°F")
        logger.info(f"   Condition: {weather['condition']}")
        logger.info(f"   Humidity: {weather['humidity']}%")


# ============================================================================
# Example 6: Error Handling and Retries
# ============================================================================


async def example_error_handling():
    """Example: Error handling and retry mechanism."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 6: Error Handling and Retries")
    logger.info("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return

    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    # Configure with retry on error
    agent = Agent(
        name="RetryExtractor",
        role="Data Extractor with Retry",
        objective="Extract data with automatic retry on errors",
        llm=llm,
        response_format=OutputSchema(
            schema=Person,
            mode=OutputMode.NATIVE,
            strict=True,
            retry_on_error=True,  # Enable retry
            max_retries=3,  # Max 3 retry attempts
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()

    # Task with potentially incomplete information
    task = Task(
        id="extract6",
        objective="Extract person info: Bob, 35, bob@example.com (city missing - should still work)",
    )

    logger.info(f"Task: {task.objective}")
    logger.info("Note: City is missing, but schema requires it")
    logger.info("System will retry with error feedback if validation fails")

    result = await agent.execute(task)

    if isinstance(result, dict) and "output" in result:
        person = result["output"]
        logger.info("\n✅ Result after retries:")
        logger.info(f"   {person}")


# ============================================================================
# Example 7: Non-Strict Mode
# ============================================================================


async def example_non_strict_mode():
    """Example: Non-strict mode (allows extra fields)."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 7: Non-Strict Mode")
    logger.info("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not set!")
        return

    llm = BaseOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    # Non-strict mode allows extra fields
    agent = Agent(
        name="FlexibleExtractor",
        role="Flexible Data Extractor",
        objective="Extract data with flexible validation",
        llm=llm,
        response_format=OutputSchema(
            schema=Person,
            mode=OutputMode.NATIVE,
            strict=False,  # Non-strict: allows extra fields
            retry_on_error=True,
            max_retries=2,
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    await agent.initialize()

    task = Task(
        id="extract7",
        objective="Extract: Charlie, 28, charlie@example.com, Boston, phone: 555-1234",
    )

    logger.info(f"Task: {task.objective}")
    logger.info("Using non-strict mode (strict=False)")
    logger.info("LLM can include extra fields like 'phone' even if not in schema")

    result = await agent.execute(task)

    if isinstance(result, dict) and "output" in result:
        person = result["output"]
        logger.info("\n✅ Result:")
        logger.info(f"   {person}")


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all structured output examples."""
    logger.info("=" * 80)
    logger.info("NucleusIQ OpenAI Structured Output Examples")
    logger.info("=" * 80)
    logger.info("\nThis example demonstrates:")
    logger.info("  ✅ AUTO mode (recommended)")
    logger.info("  ✅ NATIVE mode with strict validation")
    logger.info("  ✅ Nested Pydantic models")
    logger.info("  ✅ Dataclass schemas")
    logger.info("  ✅ TypedDict schemas")
    logger.info("  ✅ Error handling and retries")
    logger.info("  ✅ Non-strict mode")
    logger.info("\n" + "=" * 80)

    examples = [
        ("AUTO Mode with Pydantic", example_auto_mode_pydantic),
        ("NATIVE Mode with Strict", example_native_mode_strict),
        ("Nested Pydantic Models", example_nested_pydantic),
        ("Dataclass Schema", example_dataclass_schema),
        ("TypedDict Schema", example_typeddict_schema),
        ("Error Handling", example_error_handling),
        ("Non-Strict Mode", example_non_strict_mode),
    ]

    for name, example_func in examples:
        try:
            await example_func()
            logger.info(f"\n✅ {name} completed\n")
        except Exception as e:
            logger.error(f"\n❌ {name} failed: {e}\n", exc_info=True)

    logger.info("=" * 80)
    logger.info("All structured output examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
