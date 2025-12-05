
## Overview

The Agent class orchestrates task execution by coordinating multiple components: **Task**, **Prompt**, **Plan**, **LLM**, and **Tools**. This document explains the complete orchestration flow.

---

## Orchestration Components

### 1. **Task** - What to Do
- User's request with `id` and `objective`
- Passed to `agent.execute(task)`
- Contains the specific goal to accomplish

### 2. **Prompt** - How to Behave
- Agent's instructions and role definition
- Optional but recommended
- Contains `system` (role) and `user` (template) messages

### 3. **Plan** - How to Break Down (Optional)
- Task decomposition into steps
- Created by `agent.plan(task)` if `use_planning=True`
- Can be single-step (default) or multi-step

### 4. **LLM** - Language Model
- Processes messages and generates responses
- Handles function calling for tools
- Converts tool specs to its native format

### 5. **Tools** - Capabilities
- `BaseTool` instances (custom tools with `execute()`)
- Native tools (OpenAI tools like web_search, code_interpreter)
- Tools are converted to LLM-specific format

---

## Complete Orchestration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Orchestration Flow                      │
└─────────────────────────────────────────────────────────────────┘

User Request
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Initialize Agent (if not already done)                │
│  - Initialize memory (if provided)                              │
│  - Initialize prompt (if provided)                              │
│  - Initialize tools (all tools)                                 │
│  - Set state: INITIALIZING                                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Execute Task                                           │
│  agent.execute(task)                                            │
│  - Store current task                                           │
│  - Set state: EXECUTING                                         │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Planning (Optional)                                    │
│  IF config.use_planning = True:                                 │
│    - Set state: PLANNING                                        │
│    - Call agent.plan(task)                                      │
│      • Get context (task, agent role/objective, memory)         │
│      • IF use_llm_for_planning:                                 │
│          - Build planning prompt                                │
│          - Call LLM to generate plan                            │
│          - Parse plan response → Plan with steps                │
│      • ELSE:                                                     │
│          - Create basic one-step plan                           │
│    - Plan returned: Plan(steps=[...], task=task)                │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Route Execution                                        │
│  IF plan exists AND len(plan) > 1:                              │
│    → Execute multi-step plan (_execute_plan)                    │
│  ELSE:                                                           │
│    → Execute directly (_execute_direct)                         │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5A: Direct Execution (_execute_direct)                    │
│  ────────────────────────────────────────────────────────────   │
│  1. Convert tools to LLM format:                                │
│     - Call llm.convert_tool_specs(tools)                        │
│     - BaseTool → OpenAI function format                         │
│     - Native tools → Pass through                               │
│                                                                  │
│  2. Build messages (_build_messages):                           │
│     Messages = [                                                 │
│       System: prompt.system OR "You are {role}..."              │
│       User: prompt.user (if exists)                             │
│       User: task.objective (actual request)                     │
│     ]                                                            │
│                                                                  │
│  3. First LLM call:                                             │
│     - Send messages + tool_specs                                │
│     - LLM processes and may request tool                        │
│                                                                  │
│  4. Handle LLM response:                                        │
│     IF function_call requested:                                 │
│       - Extract tool name and arguments                         │
│       - Find tool in self.tools                                 │
│       - IF native tool: Error (handled by LLM)                  │
│       - IF BaseTool:                                            │
│           • Set state: WAITING_FOR_TOOLS                        │
│           • Call tool.execute(**args)                           │
│           • Set state: EXECUTING                                │
│       - Append to messages:                                     │
│         [assistant function_call, function result]               │
│       - Second LLM call with updated messages                   │
│       - Extract final answer                                    │
│     ELSE:                                                        │
│       - Return LLM content directly                             │
│                                                                  │
│  5. Set state: COMPLETED                                        │
│  6. Return result                                               │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5B: Multi-Step Plan Execution (_execute_plan)             │
│  ────────────────────────────────────────────────────────────   │
│  FOR each step in plan.steps:                                   │
│    1. Extract step info:                                        │
│       - step_num, action, step_task, args, details              │
│                                                                  │
│    2. Execute step based on action:                             │
│       IF action == "execute":                                   │
│         - Call _execute_direct(step_task)                       │
│       ELIF action == tool_name:                                 │
│         - Call _execute_tool(action, args)                      │
│       ELSE:                                                      │
│         - Log warning, skip step                                │
│                                                                  │
│    3. Store result:                                             │
│       - Add to results list                                     │
│       - Add to context for next steps                           │
│                                                                  │
│  4. Return final result (last step result)                      │
│  5. Set state: COMPLETED                                        │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Process Result (Optional)                              │
│  - Store in memory (if available)                               │
│  - Process through prompt.process_result() (if exists)          │
│  - Update metrics                                               │
└─────────────────────────────────────────────────────────────────┘
    ↓
Final Result Returned to User
```

---

## Detailed Orchestration Steps

### Step 1: Initialization

```python
await agent.initialize()
```

**What happens:**
1. Initialize memory system (if provided)
2. Initialize prompt (format and validate)
3. Initialize all tools (call `tool.initialize()` for each)
4. Set state to `INITIALIZING`
5. Agent is ready for task execution

---

### Step 2: Task Execution Entry Point

```python
result = await agent.execute(task)
```

**What happens:**
1. Convert task to `Task` object if it's a dict (backward compatibility)
2. Store task as `_current_task`
3. Set state to `EXECUTING`
4. Proceed to planning or direct execution

---

### Step 3: Planning (Conditional)

```python
if self.config.use_planning:
    plan = await self.plan(task)
```

**Planning Options:**

#### Option A: Basic Planning (Default)
```python
# Returns simple one-step plan
plan = Plan(
    steps=[PlanStep(step=1, action="execute", task=task)],
    task=task
)
```

#### Option B: LLM-Based Planning
```python
# If use_llm_for_planning = True
context = await self._get_context(task)  # Get context
plan = await self._create_llm_plan(task, context)  # LLM generates plan
```

**Context includes:**
- Task information
- Agent role and objective
- Memory context (if available)
- Timestamp

**LLM Planning Process:**
1. Build planning prompt with context
2. Call LLM with planning prompt
3. Parse LLM response into `PlanStep` objects
4. Return `Plan` with steps

---

### Step 4: Execution Routing

```python
if plan and len(plan) > 1:
    return await self._execute_plan(task, plan)  # Multi-step
else:
    return await self._execute_direct(task)  # Single-step or no plan
```

**Decision Logic:**
- **Multi-step plan** → `_execute_plan()` (sequential step execution)
- **Single-step or no plan** → `_execute_direct()` (direct execution)

---

### Step 5A: Direct Execution

This is the core execution path for most tasks.

#### 5A.1: Tool Conversion

```python
tool_specs = self.llm.convert_tool_specs(self.tools)
```

**What happens:**
- `BaseTool` instances → Converted to LLM's function calling format
- Native tools (e.g., `OpenAITool.web_search()`) → Passed through as-is
- Returns list of tool specifications in LLM's native format

#### 5A.2: Message Building

```python
messages = self._build_messages(task)
```

**Message Structure:**
```python
messages = [
    # 1. System message (Agent's role)
    {
        "role": "system",
        "content": prompt.system  # OR "You are a {role}. Your objective is {objective}."
    },
    
    # 2. User template (if prompt.user exists)
    {
        "role": "user",
        "content": prompt.user  # Optional instruction template
    },
    
    # 3. Plan context (if plan has multiple steps)
    {
        "role": "user",
        "content": "Execution Plan:\nStep 1: ...\nStep 2: ...\n\nNow execute..."
    },
    
    # 4. User's actual request
    {
        "role": "user",
        "content": task.objective  # "What is 5 + 3?"
    }
]
```

**Prompt Precedence:**
- If `prompt` exists → Use `prompt.system` and `prompt.user`
- If `prompt` is None → Use `role` and `objective` to build system message

#### 5A.3: First LLM Call

```python
resp1 = await self.llm.call(
    model=model_name,
    messages=messages,
    tools=tool_specs  # Tool specifications
)
```

**LLM Response:**
- May return text content directly
- May request a tool via `function_call`

#### 5A.4: Tool Execution (If Requested)

```python
if function_call:
    # Extract tool name and arguments
    tool_name = function_call["name"]
    tool_args = json.loads(function_call["arguments"])
    
    # Find tool
    tool = next(t for t in self.tools if t.name == tool_name)
    
    # Execute tool
    self.state = AgentState.WAITING_FOR_TOOLS
    tool_result = await tool.execute(**tool_args)
    self.state = AgentState.EXECUTING
    
    # Append to messages
    messages.extend([
        {"role": "assistant", "function_call": function_call},
        {"role": "function", "name": tool_name, "content": json.dumps(tool_result)}
    ])
    
    # Second LLM call
    resp2 = await self.llm.call(messages=messages)
    final_answer = resp2.choices[0].message.content
```

**Tool Types:**
- **BaseTool**: Has `execute()` method, called by agent
- **Native Tools**: Handled directly by LLM (e.g., OpenAI web_search)

#### 5A.5: Return Result

```python
self.state = AgentState.COMPLETED
return final_answer  # or content from first LLM call
```

---

### Step 5B: Multi-Step Plan Execution

```python
await self._execute_plan(task, plan)
```

**Process:**
1. Initialize context and results
2. For each step in `plan.steps`:
   - Extract step information (step_num, action, task, args)
   - Execute based on action:
     - `action == "execute"` → Call `_execute_direct(step_task)`
     - `action == tool_name` → Call `_execute_tool(action, args)`
   - Store result and update context
3. Return final result (last step result)
4. Set state to `COMPLETED`

**Context Building:**
- Each step's result is stored in context
- Context is available for subsequent steps
- Format: `context["step_1"] = result`, `context["step_2"] = result`, etc.

---

## State Management

The agent maintains state throughout orchestration:

```python
AgentState.INITIALIZING    # During initialization
AgentState.PLANNING        # During plan creation
AgentState.EXECUTING       # During task execution
AgentState.WAITING_FOR_TOOLS  # While tool is executing
AgentState.COMPLETED       # Task completed successfully
AgentState.ERROR           # Error occurred
```

---

## Tool Orchestration

### Tool Types

1. **BaseTool** (Custom Tools)
   - Has `execute()` method
   - Agent calls `tool.execute(**args)`
   - Example: Calculator, API calls, database queries

2. **Native Tools** (LLM-Specific)
   - No `execute()` method
   - Handled directly by LLM
   - Example: `OpenAITool.web_search()`, `OpenAITool.code_interpreter()`

### Tool Conversion Flow

```
Agent Tools (List[BaseTool | OpenAITool])
    ↓
llm.convert_tool_specs(tools)
    ↓
For each tool:
    - BaseTool → tool.get_spec() → Convert to LLM format
    - OpenAITool → tool.get_spec() → Pass through (already in LLM format)
    ↓
LLM Native Tool Specifications
    ↓
Sent to LLM in first call
```

---

## Error Handling

### During Execution

```python
try:
    # Execution logic
except Exception as e:
    self._logger.error(f"Error: {str(e)}")
    self.state = AgentState.ERROR
    # Store error in memory (if available)
    # Update metrics
    # Return fallback (echo objective)
```

### During Planning

```python
try:
    plan = await self._create_llm_plan(task, context)
except Exception as e:
    self._logger.error(f"LLM planning failed: {str(e)}")
    # Fallback to basic plan
    plan = await self._create_basic_plan(task)
```

---

## Example: Complete Orchestration

```python
# 1. Create agent
agent = Agent(
    name="CalculatorBot",
    role="Calculator",
    objective="Perform calculations",
    llm=llm,
    tools=[calculator_tool],
    config=AgentConfig(use_planning=False)  # Direct execution
)

# 2. Initialize
await agent.initialize()
# State: INITIALIZING

# 3. Create task
task = Task(id="task1", objective="What is 15 + 27?")

# 4. Execute
result = await agent.execute(task)
# State: EXECUTING → COMPLETED

# Orchestration flow:
# 1. No planning (use_planning=False)
# 2. Route to _execute_direct()
# 3. Convert tools: calculator_tool → OpenAI function format
# 4. Build messages: [system, task.objective]
# 5. First LLM call with tools
# 6. LLM requests calculator tool
# 7. Execute calculator.execute(a=15, b=27) → 42
# 8. Second LLM call with tool result
# 9. LLM returns: "15 + 27 = 42"
# 10. State: COMPLETED
# 11. Return result
```

---

## Key Orchestration Principles

1. **Separation of Concerns**
   - Task = What to do
   - Prompt = How to behave
   - Plan = How to break down
   - Tools = What capabilities are available

2. **Flexible Execution**
   - Direct execution for simple tasks
   - Planned execution for complex tasks
   - Tool execution when needed

3. **State Management**
   - Clear state transitions
   - Error handling at each step
   - Metrics tracking

4. **Tool Abstraction**
   - LLM-agnostic tool interface (`BaseTool`)
   - LLM-specific tool conversion
   - Support for native tools

5. **Backward Compatibility**
   - Supports both `Task` objects and dicts
   - Supports both `Plan` objects and lists
   - Graceful fallbacks

---

## Summary

The Agent orchestrates execution by:

1. **Initializing** all components (memory, prompt, tools)
2. **Planning** (optional) to break down complex tasks
3. **Routing** to direct or planned execution
4. **Building** messages from prompt and task
5. **Converting** tools to LLM format
6. **Calling** LLM with messages and tools
7. **Executing** tools when requested
8. **Processing** results and updating state
9. **Returning** final result to user

The orchestration is flexible, supports multiple execution modes, and handles errors gracefully while maintaining clear state transitions throughout the process.

