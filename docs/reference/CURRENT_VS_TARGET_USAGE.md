# Current vs Target Usage - Side by Side Comparison

## Simple Use Case: Calculator Agent

### Current (Works Now) ✅

```python
from nucleusiq.agents import Agent
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.core.tools import BaseTool

# Create tool
class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(name="add", description="Add two numbers")
    
    async def execute(self, a: int, b: int) -> int:
        return a + b
    
    def get_spec(self):
        return {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }

# Create agent
llm = MockLLM()
calculator = CalculatorTool()
agent = Agent(
    name="CalculatorBot",
    role="Calculator",
    objective="Perform calculations",
    llm=llm,
    tools=[calculator]
)

await agent.initialize()

# Execute
task = {"id": "task1", "objective": "What is 15 + 27?"}
result = await agent.execute(task)
print(result)  # "15 + 27 = 42"
```

**What happens:**
1. Build messages: `["You are a Calculator", "What is 15 + 27?"]`
2. Call LLM with tool specs
3. LLM requests `add(15, 27)`
4. Tool executes → 42
5. Call LLM again with result
6. LLM returns answer

**This works!** But it's just orchestrated LLM calls.

---

### Target (What We Want) 🎯

```python
# Same setup, but agent is more autonomous

agent = Agent(
    name="CalculatorBot",
    role="Calculator",
    objective="Perform calculations",
    llm=llm,
    tools=[calculator],
    config=AgentConfig(
        reasoning_mode="chain_of_thought",  # NEW
        show_reasoning=True  # NEW
    )
)

task = Task(id="task1", objective="What is 15 + 27?")
result = await agent.execute(task)

# Agent should:
# 1. Think: "I need to add 15 and 27"
# 2. Reason: "15 + 27 = 42"
# 3. Use calculator tool to verify
# 4. Return: "15 + 27 = 42. I calculated this by adding 15 and 27 together."
```

**Difference:** Agent reasons through the problem, not just calls LLM.

---

## Complex Use Case: Research Agent

### Current (Limited) ⚠️

```python
from nucleusiq.providers.llms.openai.tools import OpenAITool

# Create agent with web search
llm = BaseOpenAI(api_key="...")
agent = Agent(
    name="ResearchAgent",
    role="Researcher",
    objective="Research topics",
    llm=llm,
    tools=[OpenAITool.web_search()]  # Native tool
)

task = {
    "id": "task1",
    "objective": "What's the weather in Paris and what's the average temperature?"
}

result = await agent.execute(task)
```

**What happens:**
1. LLM might request web_search tool
2. Tool executes (handled by OpenAI)
3. LLM returns answer

**Problem:** 
- No planning
- No multi-step reasoning
- No context between steps
- Just one LLM call

---

### Target (What We Want) 🎯

```python
agent = Agent(
    name="ResearchAgent",
    role="Researcher",
    objective="Research topics thoroughly",
    llm=llm,
    tools=[OpenAITool.web_search()],
    config=AgentConfig(
        use_planning=True,  # Enable planning
        use_llm_for_planning=True,  # Use LLM to create plan
        reasoning_mode="chain_of_thought"
    )
)

task = Task(
    id="task1",
    objective="What's the weather in Paris and what's the average temperature?"
)

result = await agent.execute(task)

# Agent should:
# 1. Create plan:
#    - Step 1: Search for weather in Paris
#    - Step 2: Extract temperature data
#    - Step 3: Calculate average
#
# 2. Execute Step 1:
#    - Use web_search("weather in Paris")
#    - Get results
#    - Store in context
#
# 3. Execute Step 2:
#    - Extract temperatures from results
#    - Store in context
#
# 4. Execute Step 3:
#    - Calculate average from extracted temps
#    - Return final answer
#
# 5. Return: "The weather in Paris is... The average temperature is X°C."
```

**Difference:** Agent plans, executes step-by-step, uses context between steps.

---

## Multi-Turn Conversation

### Current (Doesn't Work Well) ❌

```python
# First conversation
task1 = {"id": "task1", "objective": "My name is John"}
result1 = await agent.execute(task1)
# Agent doesn't remember

# Second conversation
task2 = {"id": "task2", "objective": "What's my name?"}
result2 = await agent.execute(task2)
# Agent doesn't know - no memory between calls
```

**Problem:** No memory between calls, each execution is isolated.

---

### Target (What We Want) 🎯

```python
from nucleusiq.core.memory import VectorMemory

memory = VectorMemory()
agent = Agent(
    name="Assistant",
    role="Personal Assistant",
    objective="Help users",
    llm=llm,
    memory=memory  # Add memory
)

# First conversation
task1 = Task(id="task1", objective="My name is John")
result1 = await agent.execute(task1)
# Agent stores: "User's name is John"

# Second conversation
task2 = Task(id="task2", objective="What's my name?")
result2 = await agent.execute(task2)
# Agent retrieves from memory: "Your name is John"

# Third conversation
task3 = Task(id="task3", objective="Recommend a Python library")
result3 = await agent.execute(task3)
# Agent uses memory: "John likes Python (from context), so recommend Python libraries"
```

**Difference:** Agent remembers previous conversations and uses them.

---

## Goal-Oriented Task

### Current (Not Supported) ❌

```python
# User wants agent to work towards a goal
# But there's no goal tracking

task = {
    "id": "task1",
    "objective": "Find meeting time, send email, update calendar"
}

result = await agent.execute(task)
# Agent might do one thing, but doesn't track progress
# Doesn't know when goal is achieved
```

**Problem:** No goal tracking, no progress monitoring, no achievement detection.

---

### Target (What We Want) 🎯

```python
from nucleusiq.agents.goal import Goal

agent = Agent(
    name="TaskAgent",
    role="Task Executor",
    objective="Complete complex tasks",
    llm=llm,
    tools=[calendar_tool, email_tool],
    config=AgentConfig(use_planning=True)
)

# Create goal
goal = Goal(
    objective="Find next meeting time, send email to team, update calendar",
    success_criteria=["meeting_time_found", "email_sent", "calendar_updated"]
)

# Agent works towards goal
result = await agent.achieve_goal(goal)

# Agent:
# 1. Creates plan with 3 steps
# 2. Tracks progress: 0% → 33% → 66% → 100%
# 3. Executes each step
# 4. Checks success criteria after each step
# 5. Adapts if step fails
# 6. Returns when all criteria met
```

**Difference:** Agent tracks progress, knows when goal achieved, adapts on failure.

---

## What Changes for Users

### Minimal Changes (Backward Compatible)

**Current code still works:**
```python
# This continues to work
agent = Agent(..., tools=[tool])
result = await agent.execute(task)
```

**New features are optional:**
```python
# Users can opt into new features
agent = Agent(
    ...,
    config=AgentConfig(
        use_planning=True,  # Optional
        reasoning_mode="chain_of_thought"  # Optional
    ),
    memory=memory  # Optional
)
```

### New Capabilities (Optional)

1. **Goal Tracking** (NEW)
```python
goal = Goal(objective="...")
result = await agent.achieve_goal(goal)  # NEW method
```

2. **Reasoning Mode** (NEW)
```python
config=AgentConfig(reasoning_mode="chain_of_thought")  # NEW
```

3. **Memory** (EXISTS, needs to be used)
```python
agent = Agent(..., memory=memory)  # Already exists, but needs to work
```

4. **Planning** (EXISTS, needs to work)
```python
config=AgentConfig(use_planning=True)  # Already exists, but broken
```

---

## Summary

### What Works Now ✅
- Simple agent with tools
- LLM function calling
- Tool execution
- Basic message building

### What Doesn't Work ❌
- Goal tracking
- Multi-step reasoning
- Planning (broken)
- Memory usage (not used)
- Context between steps
- Autonomous behavior

### What Users Will See

**Simple users (current):**
```python
agent = Agent(..., tools=[tool])
result = await agent.execute(task)  # Works as before
```

**Advanced users (target):**
```python
agent = Agent(
    ...,
    config=AgentConfig(use_planning=True, reasoning_mode="chain_of_thought"),
    memory=memory
)
goal = Goal(objective="...")
result = await agent.achieve_goal(goal)  # New capabilities
```

**Key Point:** Current usage continues to work, new features are optional additions.

