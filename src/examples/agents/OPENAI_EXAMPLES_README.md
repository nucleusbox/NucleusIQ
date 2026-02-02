# OpenAI Integration Examples

This directory contains comprehensive examples demonstrating NucleusIQ's integration with OpenAI.

## 📚 Available Examples

### 1. **openai_quick_start.py** ⚡
**Best for:** Getting started quickly

A simple, quick-start guide showing the most common patterns:
- ✅ Simple chat agent (DIRECT mode)
- ✅ Agent with tools (STANDARD mode)
- ✅ Structured output extraction
- ✅ Autonomous planning

**Run:**
```bash
python src/examples/agents/openai_quick_start.py
```

---

### 2. **openai_complete_integration.py** 🎯
**Best for:** Comprehensive feature demonstration

Complete integration examples showing all features:
- ✅ DIRECT mode with structured output
- ✅ STANDARD mode with custom BaseTool tools
- ✅ STANDARD mode with OpenAI native tools (web_search, code_interpreter)
- ✅ AUTONOMOUS mode with multi-step planning
- ✅ Structured output with explicit NATIVE mode
- ✅ ReAct agent pattern with iterative reasoning
- ✅ Mixed tools (BaseTool + OpenAI native tools)

**Run:**
```bash
python src/examples/agents/openai_complete_integration.py
```

---

### 3. **openai_structured_output_examples.py** 📊
**Best for:** Deep dive into structured output

Comprehensive structured output examples:
- ✅ AUTO mode (recommended approach)
- ✅ NATIVE mode with strict validation
- ✅ Nested Pydantic models
- ✅ Dataclass schemas
- ✅ TypedDict schemas
- ✅ Error handling and retry mechanisms
- ✅ Strict vs non-strict modes

**Run:**
```bash
python src/examples/agents/openai_structured_output_examples.py
```

---

### 4. **openai_agent.py** 📝
**Best for:** Basic agent setup

Basic example showing:
- ✅ Agent creation with OpenAI
- ✅ Tool integration
- ✅ Multiple task execution

**Run:**
```bash
python src/examples/agents/openai_agent.py
```

---

### 5. **openai_tool_example.py** 🔧
**Best for:** OpenAI native tools

Examples showing OpenAI native tools:
- ✅ `OpenAITool.web_search()`
- ✅ `OpenAITool.code_interpreter()`
- ✅ `OpenAITool.file_search()`
- ✅ `OpenAITool.image_generation()`
- ✅ `OpenAITool.computer_use()`

**Run:**
```bash
python src/examples/agents/openai_tool_example.py
```

---

### 6. **comprehensive_execution_modes_test.py** 🧪
**Best for:** Testing all execution modes

Comprehensive test suite for all execution modes:
- ✅ DIRECT mode testing
- ✅ STANDARD mode testing
- ✅ AUTONOMOUS mode testing
- ✅ Error handling
- ✅ Edge cases

**Run:**
```bash
python src/examples/agents/comprehensive_execution_modes_test.py
```

---

## 🚀 Quick Start

1. **Set up environment:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # Or create a .env file with: OPENAI_API_KEY=your-api-key-here
   ```

2. **Run the quick start:**
   ```bash
   python src/examples/agents/openai_quick_start.py
   ```

3. **Explore comprehensive examples:**
   ```bash
   python src/examples/agents/openai_complete_integration.py
   ```

---

## 📖 Key Concepts Demonstrated

### Execution Modes (Gearbox Strategy)

1. **DIRECT Mode (Gear 1)**
   - Fast, single LLM call
   - No tools, no planning
   - Best for: Chat, Q&A, simple tasks

2. **STANDARD Mode (Gear 2)** - Default
   - Tool-enabled, linear execution
   - Multiple tool calls supported
   - Best for: Tool-based tasks

3. **AUTONOMOUS Mode (Gear 3)**
   - Full planning and execution
   - Multi-step task breakdown
   - Context passing between steps
   - Best for: Complex, multi-step tasks

### Structured Output

**AUTO Mode (Recommended):**
```python
agent = Agent(
    response_format=Person  # Just pass the schema!
)
```

**NATIVE Mode (Explicit):**
```python
agent = Agent(
    response_format=OutputSchema(
        schema=Person,
        mode=OutputMode.NATIVE,
        strict=True
    )
)
```

### Tool Types

**Custom BaseTool:**
```python
def add(a: int, b: int) -> int:
    return a + b

tool = BaseTool.from_function(add)
```

**OpenAI Native Tools:**
```python
from nucleusiq.providers.llms.openai.tools.openai_tool import OpenAITool

web_search = OpenAITool.web_search()
code_interpreter = OpenAITool.code_interpreter()
```

---

## 🔑 Common Patterns

### Pattern 1: Simple Chat Agent
```python
llm = BaseOpenAI(model_name="gpt-4o-mini")
agent = Agent(
    name="ChatBot",
    role="Assistant",
    objective="Help users",
    llm=llm,
    config=AgentConfig(execution_mode=ExecutionMode.DIRECT)
)
await agent.initialize()
result = await agent.execute({"id": "1", "objective": "Hello!"})
```

### Pattern 2: Tool-Enabled Agent
```python
llm = BaseOpenAI(model_name="gpt-4o-mini")
tool = BaseTool.from_function(my_function)
agent = Agent(
    name="ToolBot",
    role="Assistant",
    objective="Use tools to help",
    llm=llm,
    tools=[tool],
    config=AgentConfig(execution_mode=ExecutionMode.STANDARD)
)
await agent.initialize()
result = await agent.execute({"id": "1", "objective": "Use tool..."})
```

### Pattern 3: Structured Output
```python
class Person(BaseModel):
    name: str
    age: int

llm = BaseOpenAI(model_name="gpt-4o-mini")
agent = Agent(
    name="Extractor",
    role="Data Extractor",
    objective="Extract data",
    llm=llm,
    response_format=Person  # AUTO mode
)
await agent.initialize()
result = await agent.execute({"id": "1", "objective": "Extract: John, 30"})
```

### Pattern 4: Autonomous Planning
```python
llm = BaseOpenAI(model_name="gpt-4o-mini")
agent = Agent(
    name="Planner",
    role="Planning Assistant",
    objective="Plan and execute",
    llm=llm,
    tools=[...],
    config=AgentConfig(execution_mode=ExecutionMode.AUTONOMOUS)
)
await agent.initialize()
result = await agent.execute({"id": "1", "objective": "Complex multi-step task"})
```

---

## 📋 Requirements

- Python 3.8+
- `OPENAI_API_KEY` environment variable
- NucleusIQ framework installed
- Dependencies: `openai`, `pydantic`, `python-dotenv`

---

## 🎯 Next Steps

1. **Start with:** `openai_quick_start.py`
2. **Explore features:** `openai_complete_integration.py`
3. **Deep dive:** `openai_structured_output_examples.py`
4. **Test thoroughly:** `comprehensive_execution_modes_test.py`

---

## 💡 Tips

- Use **AUTO mode** for structured output (simplest)
- Use **DIRECT mode** for simple chat/Q&A
- Use **STANDARD mode** for tool-based tasks
- Use **AUTONOMOUS mode** for complex multi-step tasks
- Mix **BaseTool** and **OpenAI native tools** as needed
- Enable **strict=True** for strict schema validation
- Use **retry_on_error=True** for better reliability

---

## 📚 Related Documentation

- [Agent Architecture Plan](../../../docs/AGENT_ARCHITECTURE_PLAN_V2.md)
- [Structured Output Guide](../../../docs/structured_output.md)
- [Tool System Guide](../../../docs/tools.md)
- [Implementation Summary](../../../IMPLEMENTATION_SUMMARY.md)

