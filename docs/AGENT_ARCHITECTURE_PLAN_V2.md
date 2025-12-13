# Agent Architecture Plan V2 - Comprehensive Unified Framework Design

**Last Updated:** After Gearbox Strategy Implementation (Phase 1, Week 1)  
**Status:** Phase 1 (Gearbox Strategy) - Week 1 Complete ✅

---

## Table of Contents

1. [Overview](#overview)
2. [The Unified Architecture: Gearbox + Agent Types](#the-unified-architecture-gearbox--agent-types)
3. [Current Implementation Status](#current-implementation-status)
4. [The Gearbox Strategy: Execution Modes](#the-gearbox-strategy-execution-modes)
5. [Multiple Agent Types: Reasoning Patterns](#multiple-agent-types-reasoning-patterns)
6. [Guardrails System](#guardrails-system)
7. [Task, Prompt, and Plan Relationship](#task-prompt-and-plan-relationship)
8. [Gap Analysis Summary](#gap-analysis-summary)
9. [Competitive Analysis Summary](#competitive-analysis-summary)
10. [Orchestration Flow & Issues](#orchestration-flow--issues)
11. [Implementation Plan](#implementation-plan)
12. [Usage Examples](#usage-examples)
13. [Success Criteria](#success-criteria)
14. [Files to Create](#files-to-create)
15. [Timeline Summary](#timeline-summary)

---

## Overview

This document outlines the comprehensive architecture for NucleusIQ, integrating the **Gearbox Strategy** (progressive complexity) with **Multiple Agent Types** (reasoning patterns) to create a unified, flexible framework.

**Core Philosophy**: A single Agent class that adapts both its **execution mode** (complexity level) and **agent type** (reasoning pattern) based on configuration.

**Key Innovation**: The "Gearbox" metaphor - three execution modes (gears) that scale complexity from simple chat to autonomous reasoning, allowing users to start simple and scale to complex without changing frameworks.

---

## The Unified Architecture: Gearbox + Agent Types

### Two-Dimensional Configuration

The Agent framework operates on two dimensions:

1. **Execution Mode (Gearbox)**: Complexity level
   - `ExecutionMode.DIRECT` (Gear 1): Fast, simple, no tools
   - `ExecutionMode.STANDARD` (Gear 2): Tool-enabled, linear execution (default)
   - `ExecutionMode.AUTONOMOUS` (Gear 3): Full reasoning loop with planning and self-correction

2. **Agent Type**: Reasoning pattern (11 types planned)
   - `simple`, `react`, `rewoo`, `reflection`, `cot`, `utility`, `goal`, `model`, `reflex`, `simple_reflex`, `reactive`, `proactive`

### Configuration Matrix

```
┌─────────────────────────────────────────────────────────────┐
│              Agent Configuration Matrix                      │
└─────────────────────────────────────────────────────────────┘

Execution Mode (Gearbox)    │  Agent Type (Reasoning Pattern)
────────────────────────────┼─────────────────────────────────
Direct (Gear 1)             │  • Simple
                            │  • CoT (Chain of Thought)
                            │  • Simple Reflex
                            │
Standard (Gear 2)           │  • Simple
                            │  • Reflex
                            │  • Reactive
                            │  • CoT
                            │
Autonomous (Gear 3)         │  • ReAct
                            │  • ReWoo
                            │  • Reflection
                            │  • Goal-based
                            │  • Utility-based
                            │  • Model-based
                            │  • Proactive
```

**Key Insight**: Not all agent types work in all modes. The framework validates and routes appropriately.

---

## Current Implementation Status

### ✅ Completed (Phase 1, Week 1)

1. **ExecutionMode Enum** ✅
   - Created `ExecutionMode` enum in `src/nucleusiq/agents/config/agent_config.py`
   - Three modes: `DIRECT`, `STANDARD`, `AUTONOMOUS`
   - Exported in `__init__.py`

2. **AgentConfig Updates** ✅
   - Added `execution_mode: ExecutionMode` field (default: `STANDARD`)
   - Added `enable_memory: bool` field (default: `True`)
   - Added `require_quality_check: bool` field (default: `False`)
   - Added `max_iterations: int` field (default: `10`)
   - Marked `use_planning` as deprecated

3. **Executor Component** ✅
   - Created `src/nucleusiq/agents/components/executor.py`
   - Implements tool execution with validation
   - Supports BaseTool (not native tools)
   - Handles context passing for plan steps
   - Error handling and logging

4. **Execution Modes Implementation** ✅
   - `_run_direct()`: Fast mode without tools
   - `_run_standard()`: Tool-enabled, linear execution (uses Executor)
   - Mode routing in `execute()` method
   - Backward compatible (defaults to STANDARD)

5. **Component Initialization** ✅
   - Executor initialized in `Agent.initialize()`
   - Memory initialization respects `enable_memory` flag
   - Lazy loading of components

### 🚧 In Progress

- **Autonomous Mode**: Not yet implemented (falls back to standard)
- **Planner Component**: Not yet created (Week 2)
- **Critic Component**: Not yet created (Week 2)
- **Agent Types**: Not yet implemented (Phase 2)
- **Guardrails**: Not yet implemented (Phase 3)

### 📋 Planned

- Tests for execution modes
- Tests for Executor component
- Autonomous mode implementation
- All agent types
- Guardrails system

---

## The Gearbox Strategy: Execution Modes

### Overview

The Gearbox Strategy provides three execution modes that scale complexity based on need:

- **Gear 1 (Direct)**: Fast, cheap, no tools
- **Gear 2 (Standard)**: Tool-enabled, linear execution (default)
- **Gear 3 (Autonomous)**: Full reasoning loop with planning and self-correction

### Architecture Components

The Agent acts as a facade for specialized sub-systems:

| Component | Role | Active In | Status |
|-----------|------|-----------|--------|
| **AgentConfig** | The "Settings" - Defines mode, retries, safety rules | All Modes | ✅ Implemented |
| **Executor** | The "Hands" - Tool selection, validation, execution | Standard, Autonomous | ✅ Implemented |
| **Planner** | The "Strategist" - Breaks tasks into dependencies | Autonomous Only | 🚧 Planned (Week 2) |
| **Critic** | The "Quality Control" - Reviews output against goals | Autonomous Only | 🚧 Planned (Week 2) |
| **Memory** | The "Context" - Stores history, partial results, state | Standard, Autonomous (optional) | ✅ Exists (needs integration) |

### Execution Mode Details

#### Mode A: Direct (Fast Chat) ✅ Implemented

**Purpose**: Simple, fast interactions without tools

**Logic**: `Input → LLM → Output`

**Use Cases**: Chatbots, creative writing, simple explanations

**Characteristics**:
- Near-zero overhead
- No tool execution
- No planning
- No memory (optional)
- Single LLM call

**Compatible Agent Types**:
- `simple`: Basic direct execution
- `cot`: Chain of Thought reasoning
- `simple_reflex`: Simple reflex responses

**Configuration**:
```python
from nucleusiq.agents.config import ExecutionMode

config = AgentConfig(
    execution_mode=ExecutionMode.DIRECT,
    agent_type=AgentType.SIMPLE  # or COT, SIMPLE_REFLEX
)
```

**Implementation**:
```python
async def _run_direct(self, task: Union[Task, Dict[str, Any]]) -> Any:
    """Gear 1: Direct mode - Fast, simple, no tools."""
    # Build messages (no tools, no plan)
    messages = self._build_messages(task_dict, plan=None)
    
    # Single LLM call (no tools)
    response = await self.llm.call(
        model=getattr(self.llm, "model_name", "default"),
        messages=messages,
        tools=None,  # Direct mode: no tools
    )
    
    # Extract and return content
    return self._extract_content(response)
```

---

#### Mode B: Standard (Tool User) ✅ Implemented

**Purpose**: Tool-enabled tasks with linear execution

**Logic**: `Input → Decision → Tool Execution → Result`

**Use Cases**: "Check the weather", "Query database", "Search information"

**Characteristics**:
- Tool execution enabled
- Linear flow (no loops)
- Fire-and-forget (tries once, returns error if fails)
- Optional memory
- Multiple tool calls supported

**Compatible Agent Types**:
- `simple`: Basic tool execution
- `reflex`: Reflex-based tool selection
- `reactive`: Reactive tool usage
- `cot`: Chain of Thought with tools

**Configuration**:
```python
config = AgentConfig(
    execution_mode=ExecutionMode.STANDARD,  # Default
    agent_type=AgentType.SIMPLE  # or REFLEX, REACTIVE, COT
)
```

**Implementation**:
```python
async def _run_standard(self, task: Union[Task, Dict[str, Any]]) -> Any:
    """Gear 2: Standard mode - Tool-enabled, linear execution."""
    # Convert tools to LLM format
    tool_specs = self.llm.convert_tool_specs(self.tools) if self.tools else []
    
    # LLM call loop (may request multiple tools)
    max_tool_calls = 10
    tool_call_count = 0
    
    while tool_call_count < max_tool_calls:
        response = await self.llm.call(messages, tools=tool_specs)
        
        if function_call:
            # Use Executor to execute tool
            tool_result = await self.executor.execute(fn_call)
            # Continue loop to get final answer
        else:
            # Return content
            return self._extract_content(response)
```

---

#### Mode C: Autonomous (Reasoning Engine) 🚧 Planned (Week 2)

**Purpose**: Complex tasks requiring planning, reasoning, and self-correction

**Logic**: `Input → Plan → Loop [Action → Observe → Critique] → Result`

**Use Cases**: "Write a Python script", "Research and write report", "Debug code"

**Characteristics**:
- Full planning system
- Self-correction loop
- Quality checking (Critic)
- Memory for context
- Retry with refinement
- Only returns when Critic satisfied

**Compatible Agent Types**:
- `react`: ReAct pattern (Thought → Action → Observation)
- `rewoo`: ReWoo pattern (Plan → Execute without observation loop)
- `reflection`: Reflection pattern (Execute → Reflect → Refine)
- `goal`: Goal-based (Track progress towards goal)
- `utility`: Utility-based (Maximize utility function)
- `model`: Model-based (Use internal model)
- `proactive`: Proactive (Anticipate and act)

**Configuration**:
```python
config = AgentConfig(
    execution_mode=ExecutionMode.AUTONOMOUS,
    agent_type=AgentType.REACT,  # or REWOO, REFLECTION, GOAL, etc.
    max_retries=3,
    require_quality_check=True
)
```

**Planned Implementation**:
```python
async def _run_autonomous(self, task: Task) -> Any:
    """Gear 3: Autonomous mode - Full reasoning loop."""
    # 1. Create plan
    plan = await self.planner.create_plan(task, self.tools)
    
    # 2. Execute plan with self-correction
    context = {"original_task": task.to_dict()}
    results = []
    
    for step_num, step in enumerate(plan.steps, 1):
        success = False
        attempts = 0
        
        while not success and attempts < self.config.max_retries:
            # Execute step
            step_result = await self.executor.execute_step(step, context)
            
            # Critique (Quality Control)
            critique = await self.critic.review(step, step_result, task)
            
            if critique.passed:
                success = True
                results.append(step_result)
                context[f"step_{step_num}"] = step_result
            else:
                # Self-correction: refine step and retry
                step = await self.planner.refine_step(step, critique.feedback)
                attempts += 1
    
    # 3. Generate final result
    return await self._generate_final_result(task, plan, results)
```

---

## Multiple Agent Types: Reasoning Patterns

### Overview

Agent Types define **how** the agent reasons and acts, while Execution Modes define **how complex** the execution is. They work together to provide flexible agent behavior.

### Agent Type Enum (Planned)

```python
from enum import Enum

class AgentType(str, Enum):
    """Agent reasoning patterns."""
    # Basic
    SIMPLE = "simple"           # Basic agent (default)
    
    # Reasoning Patterns
    REACT = "react"             # ReAct (Reasoning + Acting)
    REWOO = "rewoo"             # ReWoo (Reasoning without Observation)
    REFLECTION = "reflection"   # Reflection (self-critique)
    COT = "cot"                 # Chain of Thought
    
    # Decision-Making
    UTILITY = "utility"         # Utility-based
    GOAL = "goal"               # Goal-based
    MODEL = "model"             # Model-based
    
    # Behavioral
    REFLEX = "reflex"           # Reflex
    SIMPLE_REFLEX = "simple_reflex"  # Simple Reflex
    REACTIVE = "reactive"       # Reactive
    PROACTIVE = "proactive"     # Proactive
```

### Agent Type Implementations (Planned - Phase 2)

#### 1. Simple ✅ (Current Default)

**Pattern**: Basic execution, no special reasoning

**Compatible Modes**: Direct, Standard, Autonomous

**Implementation**: Routes to current mode's logic

---

#### 2. ReAct (Reasoning + Acting) 🚧 Partial

**Pattern**: `Thought → Action → Observation → Loop`

**Compatible Modes**: Autonomous (primary), Standard (simplified)

**Status**: `ReActAgent` class exists, needs integration into strategy pattern

**Implementation**:
```python
async def _execute_react(self, task: Task) -> Any:
    """ReAct: Thought-Action-Observation loop."""
    context = {}
    iteration = 0
    
    while iteration < self.config.max_iterations:
        # Thought: Reason about what to do
        thought = await self._think(context, task)
        
        # Action: Decide on action (tool or final answer)
        action = await self._decide_action(thought, context)
        
        if action.type == "final_answer":
            return action.content
        
        # Observation: Execute action and observe result
        observation = await self.executor.execute(action)
        context[f"step_{iteration}"] = {
            "thought": thought,
            "action": action,
            "observation": observation
        }
        
        iteration += 1
    
    return "Max iterations reached"
```

---

#### 3-11. Other Agent Types 🚧 Planned

- **ReWoo**: Plan all steps first, then execute without observation loop
- **Reflection**: Execute → Reflect → Refine → Execute again
- **CoT**: Explicit reasoning steps in prompt
- **Goal-Based**: Track progress towards goal, adapt plan
- **Utility-Based**: Maximize utility function
- **Model-Based**: Use internal model of environment
- **Reflex**: Condition → Action mapping
- **Reactive**: Respond to environment changes
- **Proactive**: Anticipate and act

**See**: Full implementation details in Phase 2 section below.

---

## Guardrails System

### Overview

Guardrails provide hooks for validation, safety checks, and custom processing before and after agent execution.

### Design (Planned - Phase 3)

**BaseGuardrail** (`src/nucleusiq/agents/guardrails/base_guardrail.py`):

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
from nucleusiq.agents.task import Task

class BaseGuardrail(ABC):
    """Base class for all guardrails."""
    
    @abstractmethod
    async def before_execute(self, task: Task, agent: 'Agent') -> Optional[Task]:
        """Called before task execution. Returns modified task or None to cancel."""
        pass
    
    @abstractmethod
    async def after_execute(self, task: Task, result: Any, agent: 'Agent') -> Any:
        """Called after task execution. Returns modified result."""
        pass
```

### Built-in Guardrails (Planned)

1. **ValidationGuardrail** - Validates task format and content
2. **SafetyGuardrail** - Checks for unsafe content/actions
3. **RateLimitGuardrail** - Enforces rate limiting
4. **CostGuardrail** - Monitors and limits API costs
5. **OutputGuardrail** - Validates and sanitizes output

### Integration (Planned)

Guardrails work across all execution modes and agent types:

```python
class Agent(BaseAgent):
    guardrails: List[BaseGuardrail] = Field(default_factory=list)
    
    async def execute(self, task: Union[Task, Dict[str, Any]]) -> Any:
        # Pre-execution guardrails
        for guardrail in self.guardrails:
            task = await guardrail.before_execute(task, self)
            if task is None:
                raise ValueError("Execution cancelled by guardrail")
        
        # Execute based on mode and type
        result = await self._execute_with_strategy(task)
        
        # Post-execution guardrails
        for guardrail in self.guardrails:
            result = await guardrail.after_execute(task, result, self)
        
        return result
```

---

## Task, Prompt, and Plan Relationship ✅ COMPLETED

### Status: ✅ Implemented

**Implementation:**
- Task class created (`src/nucleusiq/agents/task.py`)
- Plan classes created (`src/nucleusiq/agents/plan.py`)
- Plan integrated into `execute()` flow
- Optional planning via `config.use_planning` (deprecated, use `execution_mode=ExecutionMode.AUTONOMOUS`)
- Backward compatible with dict/list

### Definitions

- **Task**: What the user wants done (specific request)
- **Prompt**: How the agent should behave (instructions)
- **Plan**: How to break down the task (optional decomposition)

### Execution Flow

```
1. User calls: agent.execute(task)
   ↓
2. If config.execution_mode == ExecutionMode.AUTONOMOUS:
   - Call agent.plan(task) → Create Plan
   ↓
3. Route to execution mode:
   - Direct → _run_direct()
   - Standard → _run_standard()
   - Autonomous → _run_autonomous() (uses plan)
   ↓
4. Build messages:
   - prompt.system → Agent role (or role/objective if no prompt)
   - prompt.user → Instruction template (optional)
   - plan → Execution plan (if multiple steps)
   - task.objective → User's request
   ↓
5. Call LLM and return result
```

**See:** `docs/TASK_PROMPT_PLAN.md` for detailed documentation.

---

## Gap Analysis Summary

### The Problem

**Current Implementation (Before Gearbox)**: The `Agent` class was essentially just an **LLM function-calling wrapper**, not a true agent.

**What it did:**
1. Build messages
2. Call LLM
3. If tool requested → execute tool
4. Call LLM again
5. Return result

**This was NOT an agent** - it was just orchestrated LLM calls with tools.

### What a Real Agent Should Be

**Core Agent Capabilities (Missing → Now Being Addressed):**

1. **Autonomous Decision-Making** 🚧 → Addressed by Autonomous mode
2. **Goal-Oriented Behavior** 🚧 → Addressed by Goal-based agent type
3. **State Management & Memory** ⚠️ → Memory exists but needs active usage
4. **Self-Reflection & Self-Correction** 🚧 → Addressed by Critic component
5. **Multi-Turn Reasoning** 🚧 → Addressed by ReAct and other agent types
6. **Planning & Execution** ⚠️ → Planning exists but needs fixing (Week 2)
7. **Context Awareness** ⚠️ → Context built but needs active usage
8. **Error Recovery** 🚧 → Addressed by self-correction loop

### Comparison: Current vs Target

| Feature | Before Gearbox | After Gearbox (Target) |
|---------|----------------|------------------------|
| **Decision Making** | LLM decides | Agent decides based on state/goal |
| **Goal Tracking** | None | Track progress, know when achieved |
| **State Management** | Basic | Full state with beliefs, intentions |
| **Reasoning** | Single LLM call | Multi-step reasoning loop |
| **Planning** | Broken | Working planning with adaptation |
| **Memory** | Exists but unused | Actively used for decisions |
| **Reflection** | None | Evaluate and learn from actions |
| **Context** | Built but unused | Actively used in decisions |
| **Autonomy** | None | Autonomous decision-making |
| **Learning** | None | Learn from experience |

**See:** `docs/reference/AGENT_GAP_ANALYSIS.md` for detailed analysis.

---

## Competitive Analysis Summary

### Framework Comparison

| Framework | Approach | Key Features | NucleusIQ Gap |
|-----------|----------|--------------|---------------|
| **CrewAI** | Team-based agents | Multi-agent collaboration, shared context | ❌ No multi-agent support |
| **AutoGen** | Multi-agent conversation | Agent-to-agent communication, human-in-loop | ❌ No agent communication |
| **LangChain** | Agent chains | Tool ecosystem, agent executor | ⚠️ Limited tool ecosystem |
| **LangGraph** | Stateful graphs | Graph-based workflows, state machines | ❌ No graph architecture |

### Competitive Advantages to Build

1. **Simplicity** ✅ (Current Strength)
   - Simpler API than competitors
   - Less boilerplate
   - Easier to learn

2. **Agent-First** ✅ (Current Strength)
   - Agent is primary interface
   - LLM is implementation detail
   - More intuitive

3. **Built-in Features** ⚠️ (Needs Work)
   - Planning exists but broken → Being fixed (Week 2)
   - Memory exists but unused → Needs active usage
   - Need to make them work

4. **Native Tool Support** ✅ (Current Strength)
   - OpenAI native tools
   - Custom tools
   - Best of both worlds

5. **Progressive Complexity** ✅ (Gearbox Strategy)
   - Start simple, scale to complex
   - Single Agent class
   - No framework switching

### Competitive Gaps to Fill

1. **Multi-Agent Support** ❌ → Future phase
2. **State Management** ⚠️ → Being addressed
3. **Autonomous Decision-Making** 🚧 → Being addressed (Autonomous mode)
4. **Workflow Orchestration** ❌ → Future phase
5. **Memory Usage** ⚠️ → Needs active usage

**See:** `docs/reference/COMPETITIVE_ANALYSIS.md` for detailed comparison.

---

## Orchestration Flow & Issues

### Current Orchestration Flow

```
User Request
    ↓
Initialize Agent (if not already done)
    ↓
Execute Task
    ↓
Route to Execution Mode:
    - Direct → _run_direct()
    - Standard → _run_standard()
    - Autonomous → _run_autonomous() (planned)
    ↓
Build Messages (prompt + task + plan)
    ↓
Convert Tools to LLM Format
    ↓
Call LLM (with tools if Standard/Autonomous)
    ↓
Handle Function Calls (if any)
    ↓
Return Result
```

### Known Issues (Being Addressed)

1. **Planning System** ⚠️
   - LLM planning not being triggered → Will be fixed in Week 2
   - Plan context not being passed → Will be fixed in Week 2
   - Type mismatches in `_format_plan()` → Will be fixed in Week 2

2. **Memory Usage** ⚠️
   - Memory exists but not actively used → Needs integration
   - Context not retrieved before execution → Needs implementation

3. **Context Passing** ⚠️
   - Context built but not used between steps → Will be fixed in Week 2

**See:** `docs/reference/AGENT_ORCHESTRATION_ISSUES.md` for detailed issues list.

---

## Implementation Plan

### Phase 1: Gearbox Strategy (Execution Modes) - 2-3 weeks

#### Week 1: Core Infrastructure ✅ COMPLETED

- [x] Add `ExecutionMode` enum to `AgentConfig`
- [x] Create `Executor` component
- [x] Implement `_run_direct()` method
- [x] Refactor `_execute_direct()` to `_run_standard()` (uses Executor)
- [x] Add mode routing to `execute()` method
- [ ] Tests for each mode
- [ ] Tests for Executor component
- [ ] Tests for mode routing

#### Week 2: Autonomous Mode 🚧 IN PROGRESS

- [ ] Create `Planner` component (fix existing planning)
- [ ] Create `Critic` component
- [ ] Implement `_run_autonomous()` method
- [ ] Add self-correction loop
- [ ] Tests for autonomous mode

#### Week 3: Integration & Polish

- [ ] Component initialization based on mode
- [ ] Memory integration (active usage)
- [ ] Error handling
- [ ] Documentation
- [ ] Examples

---

### Phase 2: Agent Types - 3-4 weeks

#### Week 1: Core Types

- [ ] Create `AgentType` enum
- [ ] Add `agent_type` to `AgentConfig`
- [ ] Implement strategy dispatcher
- [ ] Implement `_execute_simple()`
- [ ] Implement `_execute_react()` (refactor existing ReActAgent)
- [ ] Tests

#### Week 2: Reasoning Patterns

- [ ] Implement `_execute_rewoo()`
- [ ] Implement `_execute_reflection()`
- [ ] Implement `_execute_cot()`
- [ ] Tests

#### Week 3: Decision-Making Types

- [ ] Implement `_execute_goal()`
- [ ] Implement `_execute_utility()`
- [ ] Implement `_execute_model()`
- [ ] Tests

#### Week 4: Behavioral Types

- [ ] Implement `_execute_reflex()`
- [ ] Implement `_execute_reactive()`
- [ ] Implement `_execute_proactive()`
- [ ] Mode compatibility validation
- [ ] Tests
- [ ] Documentation

---

### Phase 3: Guardrails - 1-2 weeks

#### Week 1: Base Infrastructure

- [ ] Create `BaseGuardrail` abstract class
- [ ] Add `guardrails` field to `Agent`
- [ ] Integrate guardrails into `execute()` flow
- [ ] Tests

#### Week 2: Built-in Guardrails

- [ ] `ValidationGuardrail`
- [ ] `SafetyGuardrail`
- [ ] `RateLimitGuardrail`
- [ ] `CostGuardrail`
- [ ] `OutputGuardrail`
- [ ] Tests
- [ ] Documentation

---

### Phase 4: Integration & Testing - 1 week

- [ ] End-to-end tests
- [ ] Performance testing
- [ ] Documentation
- [ ] Examples for all combinations
- [ ] Migration guide

---

## Usage Examples

### Example 1: Simple Chat (Direct + Simple)

```python
from nucleusiq.agents import Agent
from nucleusiq.agents.config import ExecutionMode
from nucleusiq.core.llms.mock_llm import MockLLM

llm = MockLLM()
agent = Agent(
    name="ChatBot",
    role="Assistant",
    objective="Answer questions",
    llm=llm,
    config=AgentConfig(
        execution_mode=ExecutionMode.DIRECT,
        agent_type=AgentType.SIMPLE
    )
)

result = await agent.execute("Tell me a joke")
```

### Example 2: Tool User (Standard + Simple)

```python
from nucleusiq.core.tools import BaseTool

# Create tool
def add(a: int, b: int) -> int:
    return a + b

calculator = BaseTool.from_function(add, name="add", description="Add two numbers")

agent = Agent(
    name="WeatherBot",
    role="Weather Assistant",
    objective="Provide weather information",
    llm=llm,
    tools=[calculator],
    config=AgentConfig(
        execution_mode=ExecutionMode.STANDARD,
        agent_type=AgentType.SIMPLE
    )
)

result = await agent.execute("What's 15 + 27?")
```

### Example 3: ReAct Agent (Autonomous + ReAct) 🚧 Planned

```python
agent = Agent(
    name="ResearchBot",
    role="Researcher",
    objective="Research topics",
    llm=llm,
    tools=[web_search, file_writer],
    config=AgentConfig(
        execution_mode=ExecutionMode.AUTONOMOUS,
        agent_type=AgentType.REACT,
        max_iterations=10
    )
)

result = await agent.execute("Research Python web scraping and write a guide")
```

### Example 4: Goal-Based Agent (Autonomous + Goal) 🚧 Planned

```python
agent = Agent(
    name="TaskBot",
    role="Task Executor",
    objective="Complete complex tasks",
    llm=llm,
    tools=[calendar, email, file_writer],
    config=AgentConfig(
        execution_mode=ExecutionMode.AUTONOMOUS,
        agent_type=AgentType.GOAL,
        require_quality_check=True
    )
)

result = await agent.execute(
    "Find next meeting, send email to team, update calendar"
)
```

---

## Success Criteria

### Gearbox Strategy

- [x] Three execution modes defined (ExecutionMode enum)
- [x] Executor component implemented
- [x] Direct mode implemented
- [x] Standard mode implemented
- [ ] Autonomous mode implemented
- [ ] Planner component implemented
- [ ] Critic component implemented
- [ ] Mode routing working correctly
- [ ] Performance: Direct mode has near-zero overhead
- [ ] Tests for each mode

### Agent Types

- [ ] All 11 agent types implemented
- [ ] Mode compatibility validation
- [ ] Strategy dispatcher working
- [ ] Tests for each type
- [ ] Examples for each type

### Guardrails

- [ ] BaseGuardrail interface implemented
- [ ] 5+ built-in guardrails
- [ ] Integration with Agent
- [ ] Tests and documentation

### Integration

- [ ] All combinations work seamlessly
- [ ] Backward compatible
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Examples comprehensive

---

## Files to Create

### Components ✅/🚧

- [x] `src/nucleusiq/agents/components/__init__.py`
- [x] `src/nucleusiq/agents/components/executor.py`
- [ ] `src/nucleusiq/agents/components/planner.py`
- [ ] `src/nucleusiq/agents/components/critic.py`

### Agent Types 🚧

- [ ] `src/nucleusiq/agents/config/agent_type.py`
- [ ] `src/nucleusiq/agents/strategies/__init__.py`
- [ ] `src/nucleusiq/agents/strategies/simple_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/react_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/rewoo_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/reflection_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/cot_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/goal_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/utility_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/model_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/reflex_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/reactive_strategy.py`
- [ ] `src/nucleusiq/agents/strategies/proactive_strategy.py`

### Guardrails 🚧

- [ ] `src/nucleusiq/agents/guardrails/__init__.py`
- [ ] `src/nucleusiq/agents/guardrails/base_guardrail.py`
- [ ] `src/nucleusiq/agents/guardrails/validation_guardrail.py`
- [ ] `src/nucleusiq/agents/guardrails/safety_guardrail.py`
- [ ] `src/nucleusiq/agents/guardrails/rate_limit_guardrail.py`
- [ ] `src/nucleusiq/agents/guardrails/cost_guardrail.py`
- [ ] `src/nucleusiq/agents/guardrails/output_guardrail.py`

---

## Timeline Summary

| Phase | Duration | Status | Progress |
|-------|----------|--------|----------|
| **Phase 1: Gearbox Strategy** | 2-3 weeks | 🚧 In Progress | Week 1 ✅ Complete |
| **Phase 2: Agent Types** | 3-4 weeks | 📋 Planned | Not started |
| **Phase 3: Guardrails** | 1-2 weeks | 📋 Planned | Not started |
| **Phase 4: Integration** | 1 week | 📋 Planned | Not started |
| **Total** | **7-10 weeks** | | **~15% Complete** |

---

## Key Decisions & Rationale

### 1. ExecutionMode as Enum

**Decision**: Use `ExecutionMode` enum instead of string literals

**Rationale**:
- Type safety
- IDE autocomplete
- Only 3 modes (perfect for enum)
- Clear and explicit

**Implementation**:
```python
class ExecutionMode(str, Enum):
    DIRECT = "direct"
    STANDARD = "standard"
    AUTONOMOUS = "autonomous"
```

### 2. Default Mode: STANDARD

**Decision**: Default `execution_mode` is `ExecutionMode.STANDARD`

**Rationale**:
- Maintains backward compatibility
- Most common use case (tool-enabled)
- Users can opt into simpler (DIRECT) or more complex (AUTONOMOUS)

### 3. Executor Component

**Decision**: Extract tool execution into separate `Executor` component

**Rationale**:
- Separation of concerns
- Easier to test
- Reusable across modes
- Cleaner Agent class

### 4. Progressive Complexity

**Decision**: Three execution modes that scale complexity

**Rationale**:
- Matches competitive strategy
- Low barrier to entry
- High ceiling for experts
- Single API for all use cases

---

## References

### Active Documentation

- `docs/TASK_PROMPT_PLAN.md` - Task/Prompt/Plan relationship
- `docs/TOOL_DESIGN.md` - Tool system design
- `docs/DEVELOPMENT_NOTES.md` - Active development notes
- `TODO.md` - Comprehensive TODO list

### Reference Documentation (Archived)

- `docs/reference/AGENT_GAP_ANALYSIS.md` - Gap between current and target
- `docs/reference/AGENT_ORCHESTRATION.md` - Orchestration flow explanation
- `docs/reference/AGENT_ORCHESTRATION_ISSUES.md` - Known issues
- `docs/reference/COMPETITIVE_ANALYSIS.md` - Framework comparison
- `docs/reference/CURRENT_VS_TARGET_USAGE.md` - Usage comparison
- `docs/reference/USER_EXPERIENCE_EXAMPLES.md` - User experience examples
- `docs/reference/HOW_USERS_WILL_USE_FRAMEWORK.md` - Usage guide

### Strategy Documentation

- `docs/strategy/STRATEGY_SYNTHESIS.md` - Overall strategy
- `docs/strategy/MONETIZATION_STRATEGY.md` - Monetization plan
- `docs/strategy/RELEASE_PLAN.md` - Release planning

---

*Last Updated: After Phase 1, Week 1 Implementation*  
*Status: Phase 1 (Gearbox Strategy) - Week 1 Complete ✅, Week 2 In Progress 🚧*
