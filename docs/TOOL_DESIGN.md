# Tool Design and Architecture

## Overview

NucleusIQ supports **two types of tools**:

1. **BaseTool** - Generic function calling tools that work with **ANY LLM**
   - Uses function calling protocol
   - Works with OpenAI, Anthropic, Gemini, etc.
   - You control execution via `execute()` method
   - Supports Pydantic schemas

2. **Native Tools** - LLM-specific built-in tools (native LLM features)
   - Uses native LLM capabilities (e.g., OpenAI's `web_search_preview`, `code_interpreter`)
   - Only works with specific LLM providers
   - LLM handles execution directly
   - Passed as simple dict specs (e.g., `{"type": "web_search_preview"}`)

## Tool Types

### BaseTool: Generic Function Calling

`BaseTool` is for tools that work with **ANY LLM** via function calling. It provides a standardized interface that allows agents to expose capabilities to LLMs through OpenAI-style function calling. The design is **LLM-agnostic** and works with any model that supports function calling (OpenAI, Anthropic, Gemini, etc.).

**Key Characteristics:**
- ✅ Works with any LLM supporting function calling
- ✅ Uses function calling protocol
- ✅ Has `execute()` method - you control execution
- ✅ Supports Pydantic schemas
- ✅ Generates OpenAI-compatible function specs

**Usage:**
```python
# Simple function
tool = BaseTool.from_function(add)

# With Pydantic schema
tool = BaseTool.from_function(get_weather, args_schema=WeatherInput)
```

### Native Tools: LLM-Specific Built-in Tools

Native tools are **LLM provider's built-in features** (not function calling). These are passed directly to the LLM provider's API as simple tool specs.

**Key Characteristics:**
- ⚠️ Only works with specific LLM providers
- ⚠️ Uses native LLM features (not function calling)
- ⚠️ No `execute()` method - LLM handles execution
- ⚠️ Passed as dict specs (e.g., `{"type": "web_search_preview"}`)
- ⚠️ Limited to built-in capabilities

**OpenAI Examples:**
- `{"type": "web_search_preview"}` - OpenAI's built-in web search
- `{"type": "code_interpreter"}` - OpenAI's built-in code interpreter

**Usage in Agent:**
```python
from nucleusiq.providers.llms.openai.tools import OpenAITool

# Create your own OpenAI native tool by extending OpenAITool
class MyOpenAISearchTool(OpenAITool):
    def __init__(self):
        super().__init__(
            name="web_search_preview",
            description="Search the web",
            is_native=True
        )
    
    def to_openai_spec(self):
        return {"type": "web_search_preview"}

agent = Agent(
    name="MyAgent",
    tools=[
        calculator_tool,      # BaseTool instance (function calling)
        MyOpenAISearchTool(), # OpenAI native tool (extends OpenAITool)
    ],
    ...
)
```

## Current Design Implementation

### Core Architecture

The tool system follows a simple but powerful design:

```
┌─────────────┐
│   Agent     │
│             │
│  ┌───────┐  │
│  │ Tools │  │───► get_spec() ───► OpenAI Function Spec
│  └───────┘  │
│      │      │
│      ▼      │
│  execute()  │───► Tool Execution
└─────────────┘
      │
      ▼
┌─────────────┐
│     LLM     │───► Decides when to call tools
└─────────────┘
```

### BaseTool Interface

Every tool must implement three core methods:

```python
class BaseTool(ABC):
    # Required attributes
    name: str              # Unique identifier
    description: str       # What the tool does (used by LLM)
    version: Optional[str] # Optional version metadata
    
    # Required methods
    async def initialize() -> None      # Setup before first use
    async def execute(**kwargs) -> Any  # Execute the tool logic
    def get_spec() -> Dict[str, Any]    # Generate OpenAI function spec
```

### How It Works

#### 1. Tool Registration

When an agent is created with tools, each tool is stored in the agent's `tools` list:

```python
agent = Agent(
    name="MyAgent",
    tools=[calculator_tool, web_search_tool],
    ...
)
```

#### 2. Tool Spec Generation

During agent execution, the agent collects function specifications from all tools:

```python
# In Agent.execute()
tool_specs = [t.get_spec() for t in self.tools] if self.tools else []
```

Each tool's `get_spec()` method returns an OpenAI-compatible function specification:

```python
{
    "type": "function",
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", ...]},
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False
    }
}
```

#### 3. LLM Function Calling

The agent sends tool specs to the LLM along with the user's request:

```python
resp = await self.llm.call(
    model=getattr(self.llm, "model_name", "default"),
    messages=messages,
    tools=tool_specs  # Tool specs sent to LLM
)
```

The LLM analyzes the user's request and available tools, then decides:
- Whether to call a tool
- Which tool to call
- What arguments to pass

#### 4. Tool Execution

If the LLM requests a tool, the agent:

1. **Parses the function call** from LLM response:
   ```python
   fn_call = first_msg.get("function_call")  # or getattr() for objects
   fn_name = fn_call.get("name")
   fn_args = json.loads(fn_call.get("arguments", "{}"))
   ```

2. **Finds the matching tool**:
   ```python
   tool = next((t for t in self.tools if t.name == fn_name), None)
   ```

3. **Executes the tool**:
   ```python
   tool_result = await tool.execute(**fn_args)
   ```

4. **Feeds result back to LLM**:
   ```python
   messages.extend([
       {"role": "assistant", "content": None, "function_call": fn_call},
       {"role": "function", "name": fn_name, "content": json.dumps(tool_result)}
   ])
   resp2 = await self.llm.call(model=..., messages=messages)
   ```

5. **Returns final LLM response** with tool results incorporated

## LLM Compatibility

### Supported LLMs

BaseTool works with **ANY LLM that supports OpenAI-style function calling**:

- ✅ **OpenAI** (GPT-3.5, GPT-4, GPT-4 Turbo) - Native support
- ✅ **Anthropic** (Claude 2, Claude 3) - Compatible via tool use
- ✅ **Google** (Gemini Pro, Gemini Ultra) - Compatible with OpenAI-style specs
- ✅ **Any OpenAI-Compatible API** - Works with compatible endpoints

### Why It Works

The design uses the **OpenAI function calling standard**, which has become the de facto standard for tool/function calling across LLM providers. By generating OpenAI-compatible specs, tools work with any LLM that supports this standard.

## Tool Creation Methods

### Method 1: Simple Function Wrapping

For simple Python functions, use `BaseTool.from_function()`:

```python
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tool = BaseTool.from_function(add, description="Add two integers")
```

**What it does:**
- Automatically extracts function signature using `inspect`
- Maps Python types to JSON Schema types (`int` → `"integer"`, `str` → `"string"`, etc.)
- Generates OpenAI function spec automatically
- Wraps function in async `execute()` method

**Limitations:**
- No state management
- No initialization logic
- No additional methods

### Method 2: Full Subclass (Recommended for Complex Tools)

For complex tools, subclass `BaseTool`:

```python
class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
        self.history = []  # Can maintain state!
    
    async def initialize(self) -> None:
        # Setup: connect to DB, load config, etc.
        self.history = []
    
    async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        # Your tool logic here
        result = perform_calculation(operation, a, b)
        self.history.append(result)  # Update state
        return {"result": result}
    
    def get_spec(self) -> Dict[str, Any]:
        # Return OpenAI function spec
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", ...]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"],
                "additionalProperties": False
            }
        }
```

## Tool Capabilities (Beyond Simple Functions)

BaseTool is **NOT limited to simple function calls**. It can serve **any purpose**:

### 1. State Management

Tools can maintain state across calls:

```python
class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(...)
        self.history = []  # State!
    
    async def execute(self, operation: str, a: float, b: float):
        result = perform_calculation(operation, a, b)
        self.history.append(result)  # Update state
        return result
```

### 2. External API Calls

Tools can make HTTP requests:

```python
class WebSearchTool(BaseTool):
    async def initialize(self):
        self.client = httpx.AsyncClient()
    
    async def execute(self, query: str):
        response = await self.client.get(f"https://api.search.com?q={query}")
        return response.json()
```

### 3. Database Operations

Tools can interact with databases:

```python
class DatabaseTool(BaseTool):
    async def initialize(self):
        self.conn = await connect_to_database()
    
    async def execute(self, query: str):
        return await self.conn.execute(query)
```

### 4. File System Access

Tools can read/write files:

```python
class FileTool(BaseTool):
    async def execute(self, operation: str, path: str, content: str = None):
        if operation == "read":
            return read_file(path)
        elif operation == "write":
            write_file(path, content)
```

### 5. MCP (Model Context Protocol)

Tools can integrate with MCP servers:

```python
class MCPTool(BaseTool):
    async def execute(self, resource_type: str, resource_id: str):
        # Access MCP resources
        return mcp_data
```

### 6. Code Execution

Tools can execute code (with safety measures):

```python
class CodeExecutionTool(BaseTool):
    async def execute(self, code: str, language: str = "python"):
        # Execute code in sandboxed environment
        return execution_result
```

### 7. Custom Business Logic

Tools can implement any custom functionality:

```python
class CustomTool(BaseTool):
    async def execute(self, **kwargs):
        # Any custom logic
        return result
```

## Design Principles

### 1. OpenAI Function Calling Standard

BaseTool follows OpenAI's function calling specification:
- Function name, description, and parameters
- JSON Schema for parameter validation
- Type mapping (int → integer, str → string, etc.)

This ensures compatibility across LLM providers.

### 2. Async-First

All tool operations are async to support:
- Network I/O (API calls, database queries)
- File I/O (reading/writing files)
- Long-running operations

### 3. Lifecycle Management

Tools have clear lifecycle hooks:
- `initialize()`: Setup before first use (connect to DB, load config, etc.)
- `execute()`: Main operation
- `shutdown()`: Cleanup (close connections, save state, etc.)

### 4. Extensibility

The design allows for:
- Simple function wrapping
- Complex stateful tools
- External service integration
- Custom methods beyond the interface

### 5. LLM-Agnostic

Tools don't know or care which LLM is being used. They just:
- Generate function specs
- Execute when called
- Return results

The agent handles all LLM interaction.

## Type Mapping

The `_parse_annotation()` function maps Python types to JSON Schema types:

```python
int      → "integer"
float    → "number"
bool     → "boolean"
str      → "string"
dict     → "object"
list     → "array"
Unknown  → "string" (default)
```

This allows automatic spec generation from Python function signatures.

## Current Design Limitations

### 1. Single Function Per Tool

Each tool exposes one function. For multiple related functions:
- Create multiple tools, OR
- Use an `action` parameter to differentiate operations

### 2. No Built-in Validation

Parameter validation must be done in `execute()`. Consider:
- Adding validation decorators
- Using Pydantic models for parameters
- Schema validation before execution

### 3. No Tool Composition

Tools can't directly call other tools. This must be done at the agent level.

### 4. No Streaming Support

Tools return complete results. No support for streaming responses yet.

### 5. No Runtime Context

Tools don't have access to:
- Agent state
- Execution context
- Long-term memory
- User session data

(Compare with LangChain's `ToolRuntime` parameter)

## Comparison with LangChain Tools

### Similarities

Both NucleusIQ BaseTool and LangChain tools:
- ✅ Support function calling
- ✅ Generate OpenAI-compatible specs
- ✅ Can wrap Python functions
- ✅ Support async execution
- ✅ Allow custom implementations

### Differences

| Feature | NucleusIQ BaseTool | LangChain Tool |
|---------|-------------------|----------------|
| **Runtime Context** | ❌ Not yet | ✅ ToolRuntime parameter |
| **State Management** | ✅ Via instance variables | ✅ Via ToolRuntime |
| **Stream Writer** | ❌ Not yet | ✅ Built-in |
| **Decorator Pattern** | ❌ Manual class | ✅ @tool decorator |
| **Pydantic Schemas** | ✅ Manual spec | ✅ Built-in support |
| **Tool Composition** | ❌ Not yet | ✅ Tool chains |
| **Tool Registry** | ❌ Not yet | ✅ Tool registry |

## Future Enhancements

### 1. Runtime Context (High Priority)

Add `ToolRuntime` parameter to access agent state:

```python
class BaseTool(ABC):
    async def execute(self, **kwargs, runtime: ToolRuntime = None):
        # Access state, context, store
        if runtime:
            user_id = runtime.context.get("user_id")
            state = runtime.state
            store = runtime.store
```

### 2. Tool Decorator

Add `@tool` decorator for simpler creation:

```python
@tool("web_search")
def search(query: str) -> str:
    """Search the web for information."""
    return results
```

### 3. Pydantic Schema Support

Automatic schema generation from Pydantic models:

```python
class WeatherInput(BaseModel):
    location: str
    units: Literal["celsius", "fahrenheit"]

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str):
    pass
```

### 4. Stream Writer

Stream updates during execution:

```python
async def execute(self, **kwargs, stream: StreamWriter = None):
    if stream:
        await stream.write("Processing...")
    # Tool logic
    if stream:
        await stream.write("Done!")
```

### 5. Tool Categories

Organize tools by category:

```python
class BaseTool(ABC):
    category: str = "general"  # "api", "database", "file", etc.
```

### 6. Tool Dependencies

Define tool dependencies:

```python
class BaseTool(ABC):
    dependencies: List[str] = []  # Other tools this depends on
```

## Best Practices

### 1. Clear Descriptions

Write detailed tool descriptions so LLMs know when to use them:

```python
super().__init__(
    name="web_search",
    description="Search the web for current information. Use this when you need up-to-date data, news, or information not in your training data."
)
```

### 2. Proper Schemas

Define accurate parameter schemas in `get_spec()`:

```python
"parameters": {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query string"  # Clear description!
        }
    },
    "required": ["query"]
}
```

### 3. Error Handling

Handle errors gracefully in `execute()`:

```python
async def execute(self, **kwargs):
    try:
        # Tool logic
        return result
    except Exception as e:
        return {"error": str(e), "success": False}
```

### 4. Resource Cleanup

Use `shutdown()` for cleanup:

```python
def shutdown(self):
    if self._client:
        asyncio.create_task(self._client.aclose())
```

### 5. State Management

Use instance variables for tool state:

```python
def __init__(self):
    super().__init__(...)
    self.history = []  # Tool state
    self._cache = {}   # Internal cache
```

## Examples

See `src/examples/tools/` for complete examples:
- `tool_examples.py` - Basic tool examples
- `advanced_tool_examples.py` - Web search, MCP, code execution, etc.

## Native Tools: LLM-Specific Built-in Tools

### How Native Tools Work

Native tools are passed directly to the LLM provider's API. They don't use function calling - the LLM handles them internally.

**OpenAI Native Tools:**
```python
# OpenAI's web_search_preview
{"type": "web_search_preview"}

# OpenAI's code_interpreter
{"type": "code_interpreter"}
```

### Using Native Tools in Agent

**Clean approach:** Use provider-specific tool classes that extend BaseTool!

```python
from nucleusiq.providers.llms.openai.tools import OpenAITool

# Extend OpenAITool to create native tools
class MyOpenAISearchTool(OpenAITool):
    def __init__(self):
        super().__init__(
            name="web_search_preview",
            description="Search the web",
            is_native=True
        )
    
    def to_openai_spec(self):
        return {"type": "web_search_preview"}

agent = Agent(
    name="MyAgent",
    tools=[
        calculator_tool,      # BaseTool instance (function calling)
        MyOpenAISearchTool(), # OpenAI native tool (extends OpenAITool)
    ],
    llm=openai_llm,
    ...
)
```

**How it works:**
1. All tools extend `BaseTool` and implement `get_spec()`
2. BaseTool instances return function calling specs
3. LLM-specific tools (e.g., `OpenAITool`) override `get_spec()` to return native tool specs
4. Agent calls `tool.get_spec()` for all tools: `[tool.get_spec() for tool in self.tools]`
5. Agent sends to LLM: `llm.call(..., tools=all_specs)`
6. LLM handles native tools internally (no function_call response)
7. BaseTools work as normal (function calling with execute())

**Folder Structure:**
```
src/nucleusiq/
├── core/tools/
│   └── base_tool.py          # BaseTool (generic, any LLM)
└── providers/llms/openai/tools/
    ├── __init__.py
    └── openai_tool.py        # OpenAITool (base class - extend this for OpenAI native tools)
```

## Tool Type Comparison

| Feature | BaseTool | Native Tools |
|---------|----------|--------------|
| **LLM Compatibility** | ✅ Any LLM (OpenAI, Anthropic, Gemini, etc.) | ⚠️ Specific LLM only |
| **Protocol** | Function calling | Native LLM features |
| **Execution** | You control via `execute()` | LLM handles directly |
| **Schema Support** | ✅ Pydantic schemas | ❌ No schemas |
| **Custom Logic** | ✅ Any functionality | ❌ Built-in only |
| **Cross-LLM** | ✅ Portable | ❌ Provider-specific |
| **Format** | `{"type": "function", "function": {...}}` | `{"type": "web_search_preview"}` |

## When to Use Each

### Use BaseTool when:
- ✅ You need custom functionality
- ✅ You want cross-LLM compatibility
- ✅ You need to control tool execution
- ✅ You want to use Pydantic schemas
- ✅ You're building reusable tools

### Use Native Tools when:
- ⚠️ You want to use LLM's built-in features
- ⚠️ You're only targeting one LLM provider
- ⚠️ You want native LLM capabilities (search, code, etc.)
- ⚠️ You don't need custom execution logic

## Summary

**Tool Architecture:**

1. **BaseTool** - Generic function calling tools (works with any LLM)
   - ✅ LLM-Agnostic
   - ✅ Flexible - can implement any functionality
   - ✅ Simple Interface
   - ✅ State Management
   - ✅ Lifecycle Hooks
   - ✅ OpenAI-Compatible function calling format
   - ✅ Pydantic schema support
   - ✅ Function calling protocol

2. **Native Tools** - LLM-specific built-in tools (native features)
   - ⚠️ Provider-specific (e.g., OpenAI's `web_search_preview`)
   - ⚠️ Uses native LLM capabilities
   - ⚠️ No execution control (LLM handles directly)
   - ⚠️ Limited to built-in features
   - ⚠️ Passed as simple dict specs

**Key Insight:**
- **BaseTool** is for custom, cross-LLM compatible tools using function calling
- **Native Tools** are for leveraging LLM provider's built-in features (like OpenAI's `web_search_preview`, `code_interpreter`)
- **Agent combines both** - you can use BaseTool + Native Tools together
- Choose based on your needs: portability (BaseTool) vs native features (Native Tools)
