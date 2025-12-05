# How Users Will Use the Framework - Simple Guide

## Current State: What Works Now ✅

### Example 1: Simple Agent (Works!)

```python
from nucleusiq.agents import Agent
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.core.tools import BaseTool

# Create tool
def add(a: int, b: int) -> int:
    return a + b

tool = BaseTool.from_function(add, description="Add two numbers")

# Create agent
llm = MockLLM()
agent = Agent(
    name="CalculatorBot",
    role="Calculator",
    objective="Perform calculations",
    llm=llm,
    tools=[tool]
)

await agent.initialize()

# Use it
task = {"id": "task1", "objective": "What is 15 + 27?"}
result = await agent.execute(task)
print(result)  # "15 + 27 = 42"
```

**What happens:**
1. Agent builds messages
2. Calls LLM
3. LLM requests tool
4. Tool executes
5. LLM returns answer

**This works!** But it's just orchestrated LLM calls.

---

## What's Missing: Real Agent Behavior

### Example 2: Multi-Step Task (Doesn't Work Well)

**User wants:**
```python
task = {
    "id": "task1",
    "objective": "Find weather in Paris, calculate average temperature, and send email"
}
result = await agent.execute(task)
```

**What should happen:**
1. Agent creates plan: [search weather, extract temps, calculate average, send email]
2. Agent executes step 1 → gets weather data
3. Agent executes step 2 → extracts temperatures
4. Agent executes step 3 → calculates average (uses data from step 2)
5. Agent executes step 4 → sends email (uses average from step 3)
6. Returns final result

**What actually happens:**
- Agent calls LLM once
- LLM might request one tool
- Returns incomplete result
- No planning, no multi-step execution, no context between steps

---

### Example 3: Conversation with Memory (Doesn't Work)

**User wants:**
```python
# First conversation
task1 = {"id": "task1", "objective": "My name is John"}
await agent.execute(task1)

# Second conversation (later)
task2 = {"id": "task2", "objective": "What's my name?"}
result = await agent.execute(task2)
# Should return: "Your name is John"
```

**What should happen:**
- Agent remembers "User's name is John"
- Uses memory in second conversation
- Returns correct answer

**What actually happens:**
- Agent doesn't remember
- Each execution is isolated
- Returns: "I don't know" or generic answer

---

### Example 4: Goal-Oriented Task (Not Supported)

**User wants:**
```python
goal = Goal(
    objective="Research topic, write summary, send to team"
)

result = await agent.achieve_goal(goal)
# Agent should work until goal is achieved
```

**What should happen:**
- Agent creates plan
- Tracks progress: 0% → 33% → 66% → 100%
- Executes each step
- Knows when goal is achieved
- Returns when done

**What actually happens:**
- No `Goal` class
- No `achieve_goal()` method
- No progress tracking
- No achievement detection

---

## The Problem: Current vs Target

### Current Implementation

```
User → Agent.execute(task) → LLM Call → Tool (maybe) → LLM Call → Result
```

**This is just:**
- Message building
- LLM API calls
- Tool execution
- Response handling

**This is NOT:**
- Autonomous decision-making
- Goal tracking
- Multi-step reasoning
- Planning that works
- Memory usage
- Context awareness

---

### Target Implementation

```
User → Agent.execute(task)
         ↓
    [Agent Loop]
         ↓
    Evaluate State → Decide Action → Execute Action → Process Result
         ↑                                                      ↓
         └──────────────────────────────────────────────────────┘
    
    Continue until goal achieved or max iterations
```

**This should be:**
- Agent evaluates current state
- Agent decides what to do next
- Agent executes action
- Agent processes result
- Agent updates state
- Agent checks if goal achieved
- Repeat until done

---

## What Users Will See

### Simple Users (Current - Works!)

```python
# This continues to work
agent = Agent(
    name="MyAgent",
    role="Assistant",
    objective="Help users",
    llm=llm,
    tools=[tool]
)

task = {"id": "task1", "objective": "What is 5 + 3?"}
result = await agent.execute(task)
```

**No changes needed** - Simple usage continues to work.

---

### Advanced Users (Target - Needs Building)

```python
# New capabilities (optional)
agent = Agent(
    name="AdvancedAgent",
    role="Assistant",
    objective="Help users",
    llm=llm,
    tools=[tool],
    memory=memory,  # NEW: Memory system
    config=AgentConfig(
        use_planning=True,  # NEW: Enable planning
        reasoning_mode="chain_of_thought",  # NEW: Reasoning mode
        max_iterations=10  # NEW: Max iterations
    )
)

# Option 1: Simple task (works as before)
task = Task(id="task1", objective="What is 5 + 3?")
result = await agent.execute(task)

# Option 2: Goal-oriented (NEW)
goal = Goal(objective="Research topic and write summary")
result = await agent.achieve_goal(goal)  # NEW method
```

**New features are optional** - Users can opt in.

---

## Key Changes Needed

### 1. Keep Current Usage Working ✅
- Don't break existing code
- Simple agent usage continues to work
- Backward compatible

### 2. Add New Capabilities (Optional) 🆕

**A. Goal Tracking**
```python
goal = Goal(objective="...")
result = await agent.achieve_goal(goal)  # NEW
```

**B. Reasoning Mode**
```python
config=AgentConfig(reasoning_mode="chain_of_thought")  # NEW
```

**C. Working Planning**
```python
config=AgentConfig(use_planning=True)  # EXISTS but broken - needs fix
```

**D. Memory Usage**
```python
agent = Agent(..., memory=memory)  # EXISTS but not used - needs fix
```

---

## Summary: What Changes for Users

### Minimal Changes ✅

**Current code still works:**
```python
agent = Agent(..., tools=[tool])
result = await agent.execute(task)  # Works as before
```

**New features are optional:**
```python
# Users can opt into new features
agent = Agent(
    ...,
    config=AgentConfig(use_planning=True),  # Optional
    memory=memory  # Optional
)
```

### What We Need to Build

1. **Core Agent Loop** - Autonomous decision-making
2. **Goal Tracking** - Track progress, know when achieved
3. **State Management** - Full state with beliefs, intentions
4. **Reasoning Loop** - Multi-step reasoning
5. **Working Planning** - Fix existing planning
6. **Memory Usage** - Actually use memory in decisions
7. **Context Management** - Share context between steps

---

## User Experience: Before vs After

### Before (Current)

```python
# Simple agent - works
agent = Agent(..., tools=[tool])
result = await agent.execute(task)
# Result: Single LLM call → tool → LLM call → answer
```

### After (Target)

```python
# Simple agent - still works (backward compatible)
agent = Agent(..., tools=[tool])
result = await agent.execute(task)
# Result: Same as before

# Advanced agent - new capabilities (optional)
agent = Agent(
    ...,
    config=AgentConfig(use_planning=True, reasoning_mode="chain_of_thought"),
    memory=memory
)
goal = Goal(objective="...")
result = await agent.achieve_goal(goal)
# Result: Agent plans → executes → tracks progress → adapts → achieves goal
```

---

## Bottom Line

**Current:** Framework is an LLM function-calling wrapper
- ✅ Works for simple tasks
- ❌ Not a real agent
- ❌ No autonomous behavior
- ❌ No goal tracking
- ❌ No multi-step reasoning

**Target:** Framework becomes a real agent
- ✅ Simple usage still works (backward compatible)
- ✅ New capabilities are optional
- ✅ Autonomous decision-making
- ✅ Goal tracking
- ✅ Multi-step reasoning
- ✅ Working planning
- ✅ Memory usage

**Key Point:** Current users don't need to change anything. New features are optional additions.

