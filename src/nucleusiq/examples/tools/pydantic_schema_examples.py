"""
Tool Examples: Function Calling and Pydantic Schema Definition

This demonstrates:
1. Simple function calling (auto-generated from function signature)
2. Pydantic schema definition (advanced schema with validation)
"""

import os
import sys
import asyncio
import logging
from typing import Any, Dict, Literal
import json

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.tools.base_tool import BaseTool
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Simple Function Calling (Auto-generated Schema)
# ============================================================================

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Simple function tool - schema auto-generated from function signature
simple_tool = BaseTool.from_function(add, description="Add two integers")


# ============================================================================
# EXAMPLE 2: Pydantic Schema Definition (Advanced)
# ============================================================================

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )


def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result

# Tool with Pydantic schema - automatically generates detailed schema
weather_tool = BaseTool.from_function(
    get_weather,
    description="Get current weather and optional forecast for a location",
    args_schema=WeatherInput
)


# ============================================================================
# EXAMPLE 3: Calculator with Pydantic Schema
# ============================================================================

class CalculatorInput(BaseModel):
    """Input for calculator operations."""
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="The mathematical operation to perform"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


def calculate(operation: str, a: float, b: float) -> Dict[str, Any]:
    """Perform a mathematical calculation."""
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else None
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    result = operations[operation](a, b)
    return {"operation": operation, "a": a, "b": b, "result": result}

calculator_tool = BaseTool.from_function(
    calculate,
    description="Perform mathematical calculations",
    args_schema=CalculatorInput
)


# ============================================================================
# EXAMPLE 4: Search Tool with Pydantic Schema
# ============================================================================

class SearchInput(BaseModel):
    """Input for search queries."""
    query: str = Field(description="The search query string")
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of results to return"
    )
    sort_by: Literal["relevance", "date", "popularity"] = Field(
        default="relevance",
        description="Sort order for results"
    )


def search(query: str, max_results: int = 5, sort_by: str = "relevance") -> Dict[str, Any]:
    """Search for information."""
    return {
        "query": query,
        "max_results": max_results,
        "sort_by": sort_by,
        "results": [f"Result {i} for {query}" for i in range(1, max_results + 1)]
    }

search_tool = BaseTool.from_function(
    search,
    description="Search for information",
    args_schema=SearchInput
)


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demonstrate_tools():
    """Demonstrate function calling and Pydantic schema tools."""
    
    logger.info("=" * 60)
    logger.info("Tool Examples: Function Calling & Pydantic Schemas")
    logger.info("=" * 60)
    
    # Example 1: Simple function tool
    logger.info("\n1. Simple Function Tool (Auto-generated Schema):")
    logger.info(f"   Tool name: {simple_tool.name}")
    logger.info(f"   Description: {simple_tool.description}")
    spec1 = simple_tool.get_spec()
    logger.info(f"   Schema: {json.dumps(spec1, indent=2)}")
    result1 = await simple_tool.execute(a=5, b=3)
    logger.info(f"   Result: add(5, 3) = {result1}")
    
    # Example 2: Weather tool with Pydantic schema
    logger.info("\n2. Weather Tool (Pydantic Schema):")
    logger.info(f"   Tool name: {weather_tool.name}")
    logger.info(f"   Description: {weather_tool.description}")
    spec2 = weather_tool.get_spec()
    logger.info(f"   Schema: {json.dumps(spec2, indent=2)}")
    result2 = await weather_tool.execute(
        location="New York",
        units="fahrenheit",
        include_forecast=True
    )
    logger.info(f"   Result: {result2}")
    
    # Example 3: Calculator tool with Pydantic schema
    logger.info("\n3. Calculator Tool (Pydantic Schema):")
    logger.info(f"   Tool name: {calculator_tool.name}")
    spec3 = calculator_tool.get_spec()
    logger.info(f"   Schema: {json.dumps(spec3, indent=2)}")
    result3 = await calculator_tool.execute(operation="multiply", a=7, b=8)
    logger.info(f"   Result: {json.dumps(result3, indent=2)}")
    
    # Example 4: Search tool with Pydantic schema
    logger.info("\n4. Search Tool (Pydantic Schema):")
    logger.info(f"   Tool name: {search_tool.name}")
    spec4 = search_tool.get_spec()
    logger.info(f"   Schema: {json.dumps(spec4, indent=2)}")
    result4 = await search_tool.execute(
        query="Python async programming",
        max_results=3,
        sort_by="date"
    )
    logger.info(f"   Result: {json.dumps(result4, indent=2)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Key Takeaways:")
    logger.info("=" * 60)
    logger.info("1. Simple functions: Use BaseTool.from_function() - schema auto-generated")
    logger.info("2. Advanced schemas: Use args_schema parameter with Pydantic BaseModel")
    logger.info("3. Pydantic schemas provide:")
    logger.info("   - Field descriptions (used by LLM)")
    logger.info("   - Type validation")
    logger.info("   - Default values")
    logger.info("   - Constraints (min/max, enum, etc.)")
    logger.info("4. Both methods work with any LLM supporting function calling")


if __name__ == "__main__":
    asyncio.run(demonstrate_tools())

