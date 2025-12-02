"""
Advanced Tool Examples: Web Search and More

This demonstrates that BaseTool can work with ANY LLM that supports
function calling and can serve ANY purpose (not just simple functions).
"""

import os
import sys
import asyncio
import logging
from typing import Any, Dict, List, Optional
import json
import httpx

# Add src directory to path
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.core.tools.base_tool import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Web Search Tool
# ============================================================================

class WebSearchTool(BaseTool):
    """
    Web search tool that can work with any LLM supporting function calling.
    Compatible with: OpenAI, Anthropic, Gemini, etc.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            name="web_search",
            description="Search the web for current information. Use this when you need up-to-date data, news, or information not in your training data."
        )
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize HTTP client for search API."""
        self._client = httpx.AsyncClient(timeout=10.0)
        logger.info("WebSearch tool initialized")
    
    async def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
        """
        if not self._client:
            raise RuntimeError("Tool not initialized")
        
        # Example: Using a real search API (e.g., SerpAPI, Google Custom Search, etc.)
        # In production, you would use:
        # response = await self._client.get(
        #     "https://serpapi.com/search",
        #     params={"q": query, "api_key": self.api_key, "num": max_results}
        # )
        # return response.json()
        
        # Mock response for demonstration
        return {
            "query": query,
            "results": [
                {
                    "title": f"Result {i} for {query}",
                    "url": f"https://example.com/result{i}",
                    "snippet": f"This is a snippet about {query}..."
                }
                for i in range(1, max_results + 1)
            ],
            "total_results": max_results
        }
    
    def get_spec(self) -> Dict[str, Any]:
        """Generate OpenAI-compatible function spec."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    
    def shutdown(self) -> None:
        """Cleanup HTTP client."""
        if self._client:
            asyncio.create_task(self._client.aclose())


# ============================================================================
# EXAMPLE 2: Code Execution Tool (like LangChain's code interpreter)
# ============================================================================

class CodeExecutionTool(BaseTool):
    """
    Code execution tool - similar to LangChain's code interpreter.
    Can execute Python code safely.
    """
    
    def __init__(self, safe_mode: bool = True):
        super().__init__(
            name="execute_code",
            description="Execute Python code and return results. Use this for calculations, data processing, or any computational tasks."
        )
        self.safe_mode = safe_mode
        self._execution_count = 0
    
    async def initialize(self) -> None:
        """Initialize code execution environment."""
        if self.safe_mode:
            # In production, you might set up a sandboxed environment
            logger.info("Code execution tool initialized (safe mode)")
        else:
            logger.warning("Code execution tool initialized (UNSAFE mode)")
    
    async def execute(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code and return results.
        
        Args:
            code: Code to execute
            language: Programming language (default: "python")
        """
        if language != "python":
            raise ValueError(f"Unsupported language: {language}")
        
        if self.safe_mode:
            # In production, you would use a sandboxed execution environment
            # For now, we'll do basic validation
            dangerous_keywords = ["import os", "import sys", "open(", "__import__"]
            if any(keyword in code for keyword in dangerous_keywords):
                return {
                    "success": False,
                    "error": "Code contains potentially unsafe operations",
                    "output": None
                }
        
        try:
            # In production, use exec() in a restricted environment
            # For demonstration, we'll just validate
            self._execution_count += 1
            return {
                "success": True,
                "output": f"Code executed successfully (execution #{self._execution_count})",
                "code": code
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": None
            }
    
    def get_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python"],
                        "default": "python",
                        "description": "Programming language"
                    }
                },
                "required": ["code"],
                "additionalProperties": False
            }
        }


# ============================================================================
# EXAMPLE 4: Database Query Tool
# ============================================================================

class DatabaseQueryTool(BaseTool):
    """
    Database query tool - can work with any SQL database.
    """
    
    def __init__(self, connection_string: str):
        super().__init__(
            name="query_database",
            description="Execute SQL queries on a database. Use this to retrieve, insert, update, or delete data."
        )
        self.connection_string = connection_string
        self._connection = None
    
    async def initialize(self) -> None:
        """Initialize database connection."""
        # In production: await asyncpg.connect(self.connection_string)
        logger.info("Database query tool initialized")
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
        """
        # In production:
        # result = await self._connection.fetch(query, *(params or {}).values())
        # return {"rows": [dict(row) for row in result]}
        
        return {
            "query": query,
            "params": params,
            "rows": [{"id": 1, "name": "Example"}],
            "row_count": 1
        }
    
    def get_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters for parameterized queries"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demonstrate_advanced_tools():
    """Demonstrate advanced tool capabilities."""
    
    logger.info("=" * 60)
    logger.info("Advanced Tool Examples")
    logger.info("=" * 60)
    logger.info("\nThese tools work with ANY LLM that supports function calling:")
    logger.info("  - OpenAI (GPT-3.5, GPT-4, etc.)")
    logger.info("  - Anthropic (Claude)")
    logger.info("  - Google (Gemini)")
    logger.info("  - Any model supporting OpenAI-style function calling\n")
    
    # Web Search
    logger.info("1. Web Search Tool:")
    search = WebSearchTool()
    await search.initialize()
    result1 = await search.execute(query="Python async programming", max_results=3)
    logger.info(f"   Results: {json.dumps(result1, indent=2)}")
    
    # Code Execution
    logger.info("\n2. Code Execution Tool:")
    code_tool = CodeExecutionTool(safe_mode=True)
    await code_tool.initialize()
    result3 = await code_tool.execute(code="result = 2 + 2\nprint(result)")
    logger.info(f"   Code Result: {json.dumps(result3, indent=2)}")
    
    # Database Query
    logger.info("\n4. Database Query Tool:")
    db = DatabaseQueryTool(connection_string="postgresql://localhost/db")
    await db.initialize()
    result4 = await db.execute(query="SELECT * FROM users LIMIT 5")
    logger.info(f"   DB Result: {json.dumps(result4, indent=2)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Key Points:")
    logger.info("=" * 60)
    logger.info("✅ BaseTool works with ANY LLM supporting function calling")
    logger.info("✅ Can serve ANY purpose (web search, code, DB, etc.)")
    logger.info("✅ Follows OpenAI function calling standard (universal compatibility)")
    logger.info("✅ Each tool generates its own function spec via get_spec()")
    logger.info("✅ Agent automatically sends tool specs to LLM")
    logger.info("✅ LLM decides when to use tools based on user request")


if __name__ == "__main__":
    asyncio.run(demonstrate_advanced_tools())

