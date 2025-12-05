# Development Notes - NucleusIQ

This document consolidates all development planning, analysis, implementation notes, and verification documents.

---

## Table of Contents

1. [Task, Prompt, and Plan Relationship](#task-prompt-and-plan-relationship)
2. [Agent Architecture](#agent-architecture)
3. [Implementation Status](#implementation-status)
4. [Tool System](#tool-system)
5. [ReAct Agent](#react-agent)
6. [MCP Implementation](#mcp-implementation)
7. [Test Suite](#test-suite)
8. [Development Checklist](#development-checklist)

---

## Task, Prompt, and Plan Relationship

### Overview

In NucleusIQ, three key concepts work together:
- **Task** - What the user wants done (specific request)
- **Prompt** - How the agent should behave (instructions)
- **Plan** - How to break down the task (optional decomposition)

### Implementation

**Task Class** (`src/nucleusiq/agents/task.py`):
- Represents user requests with proper Pydantic validation
- Fields: `id`, `objective`, `context`, `metadata`
- Backward compatible with dict via `Task.from_dict()`

**Plan Classes** (`src/nucleusiq/agents/plan.py`):
- `PlanStep` - Single step in execution plan
- `Plan` - Complete execution plan with multiple steps
- Backward compatible with list of dicts via `Plan.from_list()`

**Agent Identity vs Task**:
- **Agent Identity** (WHO the agent is - set at creation):
  - `role`: Agent's role (e.g., "Calculator")
  - `objective`: Agent's general purpose (e.g., "Perform calculations")
  - `narrative`: Agent's description/personality
- **Task** (WHAT the user wants - passed to execute()):
  - `task.objective`: Specific user request (e.g., "What is 5 + 3?")

### Execution Flow

```
1. User calls: agent.execute(task)
   ↓
2. If config.use_planning = True:
   - Call agent.plan(task) → Create Plan
   ↓
3. If plan has multiple steps:
   - Execute plan steps sequentially
   ↓
4. Otherwise:
   - Execute directly
   ↓
5. Build messages:
   - prompt.system → Agent role
   - prompt.user → Instruction template (optional)
   - plan → Execution plan (if multiple steps)
   - task.objective → User's request
   ↓
6. Call LLM and return result
```

### Key Features

- ✅ Task and Plan have proper class structures (like Prompt)
- ✅ Backward compatible (dict/list still work)
- ✅ Planning is optional (enabled via `config.use_planning=True`)
- ✅ Clear separation between Agent identity and Task

**See:** `docs/TASK_PROMPT_PLAN.md` for detailed documentation.

---

## Agent Architecture

### Current Structure

**BaseAgent** (`src/nucleusiq/agents/builder/base_agent.py`):
- Abstract base class for all agents
- Defines agent identity (role, objective, narrative)
- Manages state, metrics, configuration
- Provides retry and timeout mechanisms

**Agent** (`src/nucleusiq/agents/agent.py`):
- Concrete implementation
- Supports planning (optional)
- Handles tool execution
- Integrates with LLM providers

**ReActAgent** (`src/nucleusiq/agents/react_agent.py`):
- Extends Agent
- Implements ReAct pattern (Reasoning + Acting)
- Thought-Action-Observation loops

### Future Enhancements

**Guardrails** (Planned):
- Pre-execution hooks
- Post-execution hooks
- Validation and safety checks
- **See:** `docs/AGENT_ARCHITECTURE_PLAN.md` for detailed plan

**Multiple Agent Types** (Planned):
- Strategy pattern for agent types
- ReWoo, Reflection, CoT, etc.
- Configuration-based, not separate classes
- **See:** `docs/AGENT_ARCHITECTURE_PLAN.md` for detailed plan

---

## Implementation Status

### Completed ✅

1. **Task and Plan Classes**
   - Created `Task` class with Pydantic validation
   - Created `Plan` and `PlanStep` classes
   - Integrated into Agent execution flow
   - Backward compatible with dict/list

2. **Planning Integration**
   - `plan()` method integrated into `execute()`
   - Optional planning via `config.use_planning`
   - Multi-step plan execution
   - Plan context in message building

3. **Agent Structure Fixes**
   - Removed private method usage from examples
   - Clarified Agent identity vs Task
   - Consistent structure (Task/Plan like Prompt)
   - All tests passing (28/28)

4. **Tool System**
   - `BaseTool` for LLM-agnostic tools
   - `OpenAITool` factory for OpenAI-specific tools
   - Pydantic schema support
   - Tool conversion handled by LLM providers

### In Progress 🚧

1. **Guardrails** - Not yet implemented
2. **Multiple Agent Types** - ReActAgent exists, others planned

---

## Tool System

### BaseTool

**Purpose:** LLM-agnostic tool interface for function calling

**Features:**
- Async execution
- Schema generation from function signatures
- Pydantic model support for arguments
- Generic tool specification format

**Usage:**
```python
from nucleusiq.core.tools import BaseTool

# From function
tool = BaseTool.from_function(
    fn=add_numbers,
    name="add",
    description="Add two numbers"
)

# With Pydantic schema
from pydantic import BaseModel

class AddInput(BaseModel):
    a: int
    b: int

tool = BaseTool.from_function(
    fn=add_numbers,
    name="add",
    description="Add two numbers",
    args_schema=AddInput
)
```

### OpenAITool

**Purpose:** Factory for OpenAI-specific native tools

**Available Tools:**
- `web_search()` - Web search preview
- `code_interpreter()` - Code execution
- `file_search()` - File search with vector stores
- `image_generation()` - Image generation
- `mcp()` - MCP servers and connectors
- `connector()` - OpenAI connectors (convenience method)
- `computer_use()` - Computer control

**Usage:**
```python
from nucleusiq.providers.llms.openai.tools import OpenAITool

# Web search
web_search = OpenAITool.web_search()

# MCP server
mcp = OpenAITool.mcp(
    server_label="dmcp",
    server_description="D&D MCP server",
    server_url="https://dmcp-server.deno.dev/sse"
)

# Connector
calendar = OpenAITool.connector(
    connector_id="connector_googlecalendar",
    server_label="google_calendar",
    server_description="Google Calendar connector",
    authorization="oauth_token"
)
```

**See:** `docs/TOOL_DESIGN.md` for detailed documentation.

---

## ReAct Agent

### Overview

ReAct (Reasoning + Acting) agent that alternates between:
- **Thought**: Reasoning about what to do next
- **Action**: Taking an action (tool call)
- **Observation**: Processing tool results
- **Loop**: Repeat until final answer

### Implementation

**File:** `src/nucleusiq/agents/react_agent.py`

**Key Features:**
- Multiple iterations (Thought-Action-Observation loop)
- Explicit Thought parsing from LLM responses
- Full history tracking
- Max iterations control

**Usage:**
```python
from nucleusiq.agents import ReActAgent

agent = ReActAgent(
    name="ReActAgent",
    role="Assistant",
    objective="Answer questions using reasoning",
    narrative="A ReAct agent",
    llm=llm,
    tools=[calculator, web_search],
    max_iterations=10
)

task = {"id": "task1", "objective": "What is 15 + 27 and what's the weather?"}
result = await agent.execute(task)
```

**See:** `src/examples/agents/react_agent_example.py` for examples.

---

## MCP Implementation

### OpenAI Native MCP

**Status:** ✅ Fully Implemented

**Features:**
- Remote MCP servers (`server_url`)
- OpenAI connectors (`connector_id`)
- `require_approval` parameter (string or dict)
- `allowed_tools` parameter (filter tools)
- `authorization` parameter (OAuth token)

**Usage:**
```python
# Remote MCP server
mcp = OpenAITool.mcp(
    server_label="dmcp",
    server_description="D&D MCP server",
    server_url="https://dmcp-server.deno.dev/sse",
    require_approval="never"
)

# OpenAI connector
calendar = OpenAITool.connector(
    connector_id="connector_googlecalendar",
    server_label="google_calendar",
    server_description="Google Calendar",
    authorization="oauth_token"
)
```

### Custom MCP Tool (BaseTool)

**Status:** ✅ Verified Working

**Question:** Can you use a custom MCP tool (BaseTool) with OpenAI LLM?

**Answer:** ✅ YES! It works via function calling.

**How:**
1. Create `CustomMCPTool` extending `BaseTool`
2. Implement MCP client protocol in `execute()`
3. OpenAI LLM uses function calling to invoke it
4. Tool connects to MCP server and returns results

**See:** `src/examples/tools/custom_mcp_tool_example.py` for example.

---

## Test Suite

### Test Files

1. **`tests/test_openai_tools.py`** - OpenAI tool factory methods (22 tests)
2. **`tests/test_tool_conversion.py`** - Tool conversion logic (8 tests)
3. **`tests/test_custom_mcp_tool.py`** - Custom MCP tool (8 tests)
4. **`tests/test_agent_with_tools.py`** - Agent tool integration (5 tests)
5. **`tests/test_agent.py`** - Agent class (28 tests)

### Test Coverage

- ✅ All OpenAI tool types
- ✅ Tool conversion (BaseTool → OpenAI format)
- ✅ Custom MCP tools
- ✅ Agent integration with tools
- ✅ Task/Plan classes
- ✅ Planning integration

**Total:** 71+ tests, all passing

**See:** `tests/README.md` for detailed test documentation.

---

## Development Checklist

### Phase 1: Task, Prompt, Plan Relationship ✅

- [x] Create Task class
- [x] Create Plan classes
- [x] Integrate plan into execute()
- [x] Update examples
- [x] Update tests
- [x] Documentation

### Phase 2: Guardrails (Not Started)

- [ ] Design guardrail system
- [ ] Implement BaseGuardrail
- [ ] Add pre-execution hooks
- [ ] Add post-execution hooks
- [ ] Create built-in guardrails
- [ ] Tests and documentation

### Phase 3: Multiple Agent Types (Partial)

- [x] ReActAgent implementation
- [ ] Strategy pattern for agent types
- [ ] ReWoo agent
- [ ] Reflection agent
- [ ] CoT agent
- [ ] Configuration-based agent types

---

## Key Decisions

### Task/Plan Structure

**Decision:** Create proper classes for Task and Plan (like Prompt)

**Rationale:**
- Consistency with Prompt structure
- Type safety with Pydantic
- Better IDE support
- Clear separation of concerns

**Implementation:**
- `Task` class with `from_dict()` for backward compatibility
- `Plan` class with `from_list()` for backward compatibility
- Agent methods accept `Union[Task, Dict]` and `Union[Plan, List]`

### Agent Identity vs Task

**Decision:** Keep Agent identity fields separate from Task

**Rationale:**
- Agent identity = WHO the agent is (constant)
- Task = WHAT the user wants (per execution)
- Clear separation of concerns

**Implementation:**
- Agent has `role`, `objective`, `narrative` (identity)
- Task has `objective` (user request)
- Documented clearly in BaseAgent docstring

### Planning Integration

**Decision:** Make planning optional via config flag

**Rationale:**
- Not all tasks need planning
- Backward compatible (default: no planning)
- Flexible for different use cases

**Implementation:**
- `config.use_planning = False` (default)
- If `True`, `plan()` is called automatically
- Plan is integrated into message building

### Tool System Design

**Decision:** LLM-agnostic BaseTool + LLM-specific factories

**Rationale:**
- BaseTool works with any LLM
- LLM-specific tools (OpenAITool) for native features
- LLM providers handle conversion

**Implementation:**
- `BaseTool` returns generic spec
- `BaseLLM.convert_tool_specs()` converts to LLM format
- `OpenAITool` factory for OpenAI native tools

---

## Agent Identity Fields (Role/Objective/Narrative)

### Summary

**Key Finding:** Prompt is optional, so `role` and `objective` serve as fallback when `prompt=None`.

**Implementation:** Precedence-based strategy where:
- If `prompt` is provided → `prompt.system`/`prompt.user` take precedence (overrides `role`/`objective`)
- If `prompt` is None → `role`/`objective` are used to build system message
- `role`/`objective` are always used for planning context (even when prompt exists)
- `narrative` is optional (never used in execution)

**Warning Messages:** When `prompt` overrides `role`/`objective`, a warning is logged to inform the user.

**Status:** ✅ Implemented

**See:** `docs/AGENT_PRECEDENCE_IMPLEMENTATION.md` for implementation details (archived for reference).

---

## Prompt System/User Fields Analysis

### Summary

**Finding:** Not all prompts have `system` and `user` fields that are always set:
- **5 out of 7 prompts** require `system` and `user` (ZeroShot, FewShot, ChainOfThought, RAG)
- **2 out of 7 prompts** have optional `system`/`user` (AutoChainOfThought, PromptComposer)
- **1 out of 7 prompts** doesn't use `system`/`user` at all (MetaPrompt)

**Impact:** Precedence strategy must check if `prompt.system` exists and is non-empty before using it.

**Status:** ✅ Analyzed and accounted for in implementation

**See:** `docs/PROMPT_SYSTEM_USER_ANALYSIS.md` for detailed analysis (archived for reference).

---

## Notes

- All implementation summaries, analysis documents, and verification notes have been consolidated here
- For detailed documentation, see:
  - `docs/TASK_PROMPT_PLAN.md` - Task/Prompt/Plan relationship
  - `docs/TOOL_DESIGN.md` - Tool system design
  - **`docs/AGENT_ARCHITECTURE_PLAN_V2.md`** - Comprehensive unified framework design (Gearbox Strategy + Agent Types + Guardrails) - **MAIN ARCHITECTURE DOCUMENT**
  - `TODAY_CHECKLIST.md` - Current tasks and checklist
  - `TODO.md` - Comprehensive TODO list
  - `docs/strategy/` - Strategy and planning documents
  - `docs/reference/` - Historical reference documents (consolidated into V2)
- For test reports, see `TEST_REPORT.md` (if exists)
- For contributing guidelines, see `CONTRIBUTING.md`
- For roadmap, see `ROADMAP.md`

---

*Last Updated: After Agent Identity Fields precedence implementation*
