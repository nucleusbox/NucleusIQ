# OpenAI Agent Implementation - Current State

## Overview

This document summarizes the current implementation of the basic agent with OpenAI integration in NucleusIQ.

## Architecture

### 1. **Agent Class** (`src/nucleusiq/agents/agent.py`)
- **Base Class**: Extends `BaseAgent` (abstract base class)
- **Primary Interface**: Users create and use `Agent` instances, not LLM clients directly
- **Key Features**:
  - Auto-initialization on first `execute()` call
  - Three execution modes (Gearbox Strategy):
    - `direct`: Fast, simple, no tools
    - `standard`: Tool-enabled, linear execution (default)
    - `autonomous`: Full reasoning loop (planned)
  - Prompt precedence: User-provided prompts override role/objective
  - State management: INITIALIZING → EXECUTING → COMPLETED/ERROR

### 2. **BaseOpenAI** (`src/nucleusiq/providers/llms/openai/nb_openai/base.py`)
- **Base Class**: Extends `BaseLLM` (abstract base class)
- **Implementation**: OpenAI API client wrapper
- **Key Features**:
  - Supports both async and sync modes (`async_mode` parameter)
  - Automatic retry logic with exponential backoff
  - Comprehensive error handling:
    - Rate limit errors
    - Connection errors
    - API errors (500, 502, 503, etc.)
    - Authentication errors
    - Permission errors
    - Invalid request errors
  - Tool spec conversion: Converts generic `BaseTool` specs to OpenAI format
  - Token estimation using tiktoken

### 3. **Integration Flow**

```
User Code
    ↓
Agent.execute(task)
    ↓
Agent._run_standard() or _run_direct()
    ↓
Agent._build_messages() → Creates message list
    ↓
Agent.llm.call() → BaseOpenAI.call()
    ↓
OpenAI API (via openai.AsyncOpenAI or openai.OpenAI)
    ↓
Response → _LLMResponse wrapper
    ↓
Agent processes response (handles function calls, content, etc.)
    ↓
Returns result to user
```

## Current Implementation Details

### Agent Execution Modes

#### Direct Mode (`execution_mode="direct"`)
- **Use Case**: Simple chat/conversation, no tools needed
- **Flow**:
  1. Build messages from task and prompt
  2. Single LLM call (no tools)
  3. Return response content
- **Code Path**: `Agent._run_direct()`

#### Standard Mode (`execution_mode="standard"` - Default)
- **Use Case**: Tool-enabled tasks, linear execution
- **Flow**:
  1. Convert tools to OpenAI format
  2. Build messages with tool specs
  3. LLM call loop (max 10 iterations):
     - If function_call → Execute tool → Add result to messages → Repeat
     - If content → Return final answer
  4. Handle tool execution via `Executor` component
- **Code Path**: `Agent._run_standard()`

#### Autonomous Mode (`execution_mode="autonomous"`)
- **Status**: Currently falls back to standard mode
- **Planned**: Full reasoning loop with planning and self-correction

### OpenAI Integration

#### BaseOpenAI Features

1. **Dual Mode Support**:
   ```python
   llm = BaseOpenAI(
       model_name="gpt-3.5-turbo",
       async_mode=True  # or False for sync
   )
   ```

2. **Error Handling**:
   - Rate limit errors → Retry with exponential backoff
   - Connection errors → Retry with exponential backoff
   - API errors (5xx) → Retry with exponential backoff
   - Authentication errors → Fail immediately (no retry)
   - Permission errors → Fail immediately (no retry)
   - Invalid request errors → Fail immediately (no retry)

3. **Tool Spec Conversion**:
   ```python
   # Generic BaseTool spec
   {
       "name": "add",
       "description": "Add two numbers",
       "parameters": {...}  # JSON Schema
   }
   
   # Converted to OpenAI format
   {
       "type": "function",
       "function": {
           "name": "add",
           "description": "Add two numbers",
           "parameters": {...}  # With additionalProperties: False
       }
   }
   ```

4. **Response Wrapping**:
   - OpenAI response → `_LLMResponse` wrapper
   - Matches `BaseLLM` interface expectations
   - Provides consistent API across LLM providers

### Message Building

The Agent builds messages in this order:

1. **System Message**:
   - If `prompt` exists → Use `prompt.system`
   - Else → Use `"You are a {role}. Your objective is to {objective}."`

2. **User Template** (if prompt has user field):
   - Use `prompt.user` as template

3. **Plan Context** (if plan exists and has multiple steps):
   - Add plan as context

4. **User Request**:
   - Use `task.objective` as the actual user message

### Tool Execution

1. **Tool Conversion**:
   - `Agent` converts `BaseTool` instances to OpenAI tool specs
   - Uses `llm.convert_tool_specs(tools)` method

2. **Function Calling Loop**:
   ```python
   while tool_call_count < max_tool_calls:
       response = await self.llm.call(
           model=model_name,
           messages=messages,
           tools=tool_specs
       )
       
       if function_call:
           # Execute tool via Executor
           tool_result = await executor.execute(...)
           # Add result to messages
           messages.append({"role": "function", "content": tool_result})
       else:
           # Final answer
           return content
   ```

3. **Executor Component**:
   - Handles tool selection, validation, and execution
   - Manages tool arguments parsing
   - Handles errors gracefully

## Examples

### Basic Agent (No Tools)
```python
from nucleusiq.agents import Agent
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI

llm = BaseOpenAI(model_name="gpt-3.5-turbo")
agent = Agent(
    name="BasicBot",
    role="Assistant",
    objective="Help users",
    llm=llm
)

result = await agent.execute({"id": "task1", "objective": "What is 2+2?"})
```

### Agent with Tools
```python
from nucleusiq.core.tools import BaseTool

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

tool = BaseTool.from_function(add)
agent = Agent(
    name="MathBot",
    role="Calculator",
    objective="Perform calculations",
    llm=llm,
    tools=[tool]
)

result = await agent.execute({"id": "task1", "objective": "What is 15 + 27?"})
```

## Current Limitations

1. **Autonomous Mode**: Not fully implemented, falls back to standard mode
2. **Streaming**: Basic support, but not fully integrated in agent execution
3. **Memory**: Memory system exists but may need more integration
4. **Error Recovery**: Basic retry logic, could be enhanced

## What Works Well

1. ✅ **Clean Separation**: Agent-first design, LLM is just a parameter
2. ✅ **Flexible Execution**: Multiple execution modes for different use cases
3. ✅ **Tool Integration**: Seamless tool calling with OpenAI function calling
4. ✅ **Error Handling**: Comprehensive error handling with appropriate retries
5. ✅ **Auto-initialization**: Agents initialize automatically on first use
6. ✅ **Prompt Flexibility**: Supports both custom prompts and role/objective

## Next Steps / Improvements

1. **Complete Autonomous Mode**: Full reasoning loop implementation
2. **Enhanced Streaming**: Better streaming support in agent execution
3. **Memory Integration**: More robust memory usage in agent execution
4. **Better Error Messages**: More user-friendly error messages
5. **Performance Optimization**: Caching, connection pooling, etc.
6. **Testing**: More integration tests with real OpenAI API

## File Structure

```
src/nucleusiq/
├── agents/
│   ├── agent.py              # Main Agent class
│   ├── builder/
│   │   └── base_agent.py     # BaseAgent abstract class
│   ├── components/
│   │   └── executor.py       # Tool execution component
│   └── config/
│       └── agent_config.py   # Agent configuration
├── providers/
│   └── llms/
│       └── openai/
│           └── nb_openai/
│               └── base.py   # BaseOpenAI implementation
└── core/
    └── llms/
        └── base_llm.py       # BaseLLM abstract class
```

## Testing

- **Unit Tests**: ✅ Comprehensive test coverage
- **Integration Tests**: ✅ Basic integration tests exist
- **Example Code**: ✅ Multiple examples demonstrating usage
- **Real API Tests**: ⚠️ Requires OPENAI_API_KEY (optional)

## Conclusion

The basic agent with OpenAI implementation is **functional and production-ready** for:
- Simple chat/conversation tasks (direct mode)
- Tool-enabled tasks (standard mode)
- Multi-step tool execution
- Error handling and retries

The architecture is clean, extensible, and follows the "Agent-first" philosophy where users work with Agents, not LLM APIs directly.

