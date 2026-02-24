"""
BaseTool Design and Extension Examples

This file demonstrates:
1. What BaseTool does
2. How to extend BaseTool for different use cases
3. Examples beyond simple function calls
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

import httpx

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.tools.base_tool import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Simple Function Tool (using from_function)
# ============================================================================


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# This is the simplest way - wrap a function
simple_tool = BaseTool.from_function(add_numbers, description="Add two integers")


# ============================================================================
# EXAMPLE 2: Custom Tool with State and Initialization
# ============================================================================


class CalculatorTool(BaseTool):
    """
    A calculator tool that maintains state (history).
    Shows how tools can have more than just function calls.
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations and maintain history",
        )
        self.history: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize the calculator - could load state, connect to DB, etc."""
        logger.info("Calculator tool initialized")
        self.history = []

    async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """
        Execute a calculation operation.

        Args:
            operation: One of 'add', 'subtract', 'multiply', 'divide'
            a: First number
            b: Second number
        """
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None,
        }

        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")

        result = operations[operation](a, b)

        # Store in history (stateful behavior)
        entry = {"operation": operation, "a": a, "b": b, "result": result}
        self.history.append(entry)

        return {"result": result, "history_count": len(self.history)}

    def get_spec(self) -> Dict[str, Any]:
        """Return OpenAI-style function spec."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform",
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["operation", "a", "b"],
                "additionalProperties": False,
            },
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Additional method - not part of BaseTool interface, but shows extensibility."""
        return self.history.copy()


# ============================================================================
# EXAMPLE 3: API Tool (HTTP requests)
# ============================================================================


class WeatherAPITool(BaseTool):
    """
    Tool that makes HTTP requests to an external API.
    Shows how tools can interact with external services.
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(
            name="get_weather", description="Get current weather for a city"
        )
        self.api_key = api_key or os.getenv("WEATHER_API_KEY")
        self.base_url = "https://api.weather.example.com"  # Example URL
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(timeout=10.0)
        logger.info("Weather API tool initialized")

    async def execute(self, city: str, units: str = "celsius") -> Dict[str, Any]:
        """
        Get weather for a city.

        Args:
            city: City name
            units: Temperature units ('celsius' or 'fahrenheit')
        """
        if not self._client:
            raise RuntimeError("Tool not initialized. Call initialize() first.")

        # Example API call (would be real in production)
        # response = await self._client.get(
        #     f"{self.base_url}/weather",
        #     params={"city": city, "units": units, "key": self.api_key}
        # )
        # return response.json()

        # Mock response for example
        return {
            "city": city,
            "temperature": 22 if units == "celsius" else 72,
            "units": units,
            "condition": "sunny",
            "humidity": 65,
        }

    def get_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                        "default": "celsius",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        }

    def shutdown(self) -> None:
        """Cleanup HTTP client."""
        if self._client:
            asyncio.create_task(self._client.aclose())


# ============================================================================
# EXAMPLE 4: Database Tool (persistent storage)
# ============================================================================


class DatabaseTool(BaseTool):
    """
    Tool that interacts with a database.
    Shows how tools can manage persistent state.
    """

    def __init__(self, db_path: str = "example.db"):
        super().__init__(
            name="database", description="Store and retrieve data from a database"
        )
        self.db_path = db_path
        self._connection = None

    async def initialize(self) -> None:
        """Initialize database connection."""
        # In real implementation, would connect to actual DB
        # import sqlite3
        # self._connection = sqlite3.connect(self.db_path)
        logger.info(f"Database tool initialized (path: {self.db_path})")

    async def execute(
        self, action: str, key: str, value: str | None = None
    ) -> Dict[str, Any]:
        """
        Store or retrieve data.

        Args:
            action: 'store' or 'retrieve'
            key: Data key
            value: Data value (required for 'store')
        """
        # Mock implementation
        if action == "store":
            if value is None:
                raise ValueError("Value required for store action")
            # Would actually store in DB
            return {"status": "stored", "key": key}
        elif action == "retrieve":
            # Would actually retrieve from DB
            return {"status": "retrieved", "key": key, "value": f"mock_value_for_{key}"}
        else:
            raise ValueError(f"Unknown action: {action}")

    def get_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "retrieve"],
                        "description": "Action to perform",
                    },
                    "key": {"type": "string", "description": "Data key"},
                    "value": {
                        "type": "string",
                        "description": "Data value (required for store)",
                    },
                },
                "required": ["action", "key"],
                "additionalProperties": False,
            },
        }


# ============================================================================
# EXAMPLE 5: File System Tool
# ============================================================================


class FileSystemTool(BaseTool):
    """
    Tool for file operations.
    Shows how tools can interact with the local file system.
    """

    def __init__(self, base_path: str = "."):
        super().__init__(name="filesystem", description="Read and write files")
        self.base_path = base_path

    async def initialize(self) -> None:
        """Validate base path exists."""
        if not os.path.exists(self.base_path):
            raise ValueError(f"Base path does not exist: {self.base_path}")
        logger.info(f"FileSystem tool initialized (base: {self.base_path})")

    async def execute(
        self, operation: str, path: str, content: str | None = None
    ) -> Dict[str, Any]:
        """
        Perform file operations.

        Args:
            operation: 'read' or 'write'
            path: File path (relative to base_path)
            content: File content (required for 'write')
        """
        full_path = os.path.join(self.base_path, path)

        if operation == "read":
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(full_path, encoding="utf-8") as f:
                return {"status": "read", "path": path, "content": f.read()}

        elif operation == "write":
            if content is None:
                raise ValueError("Content required for write operation")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "written", "path": path, "size": len(content)}

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def get_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write"],
                        "description": "File operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path (relative to base path)",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content (required for write)",
                    },
                },
                "required": ["operation", "path"],
                "additionalProperties": False,
            },
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================


async def demonstrate_tools():
    """Demonstrate different tool types."""

    logger.info("=" * 60)
    logger.info("BaseTool Extension Examples")
    logger.info("=" * 60)

    # Example 1: Simple function tool
    logger.info("\n1. Simple Function Tool:")
    result1 = await simple_tool.execute(a=5, b=3)
    logger.info(f"   add_numbers(5, 3) = {result1}")
    logger.info(f"   Spec: {json.dumps(simple_tool.get_spec(), indent=2)}")

    # Example 2: Stateful calculator
    logger.info("\n2. Stateful Calculator Tool:")
    calc = CalculatorTool()
    await calc.initialize()
    result2 = await calc.execute(operation="add", a=10, b=20)
    logger.info(f"   Result: {result2}")
    result3 = await calc.execute(operation="multiply", a=5, b=4)
    logger.info(f"   Result: {result3}")
    logger.info(f"   History: {calc.get_history()}")

    # Example 3: API tool
    logger.info("\n3. API Tool (Weather):")
    weather = WeatherAPITool()
    await weather.initialize()
    result4 = await weather.execute(city="New York", units="celsius")
    logger.info(f"   Weather: {result4}")

    # Example 4: Database tool
    logger.info("\n4. Database Tool:")
    db = DatabaseTool()
    await db.initialize()
    result5 = await db.execute(action="store", key="user1", value="John Doe")
    logger.info(f"   Store: {result5}")
    result6 = await db.execute(action="retrieve", key="user1")
    logger.info(f"   Retrieve: {result6}")

    logger.info("\n" + "=" * 60)
    logger.info("Key Takeaways:")
    logger.info("=" * 60)
    logger.info("1. BaseTool is NOT just for function calls!")
    logger.info("2. Tools can have state (CalculatorTool)")
    logger.info("3. Tools can make HTTP requests (WeatherAPITool)")
    logger.info("4. Tools can interact with databases (DatabaseTool)")
    logger.info("5. Tools can access file system (FileSystemTool)")
    logger.info("6. Tools can have initialization and cleanup")
    logger.info("7. Tools can have additional methods beyond BaseTool interface")
    logger.info("\nBaseTool provides a standard interface for LLM function-calling,")
    logger.info("but you can extend it for any capability your agent needs!")


if __name__ == "__main__":
    asyncio.run(demonstrate_tools())
