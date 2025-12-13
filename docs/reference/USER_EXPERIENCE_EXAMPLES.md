# User Experience Examples - Current vs Target

## Overview

This document shows **how users will use the framework** with concrete examples, comparing what works now vs what we need to build.

---

## Current Usage (What Works Now)

### Example 1: Simple Agent with Tool

**What it does:** Agent calls LLM, LLM requests tool, tool executes, LLM returns answer.

```python
from nucleusiq.agents import Agent
from nucleusiq.core.llms.mock_llm import MockLLM
from nucleusiq.core.tools import BaseTool

# Create a calculator tool
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

# Initialize
await agent.initialize()

# Execute task
task = {"id": "task1", "objective": "What is 15 + 27?"}
result = await agent.execute(task)
# Result: "15 + 27 = 42"
```

**What happens:**
1. Agent builds messages: `[system: "You are a Calculator", user: "What is 15 + 27?"]`
2. Calls LLM with tool specs
3. LLM requests `add(a=15, b=27)`
4. Tool executes → returns 42
5. Agent calls LLM again with tool result
6. LLM returns: "15 + 27 = 42"

**This works, but it's just:**
- Message building
- LLM call
- Tool execution
- LLM call
- Return result

**Not really an "agent" - just orchestrated LLM calls.**

---

## What's Missing - Real Agent Behavior

### Example 2: Goal-Oriented Agent (NOT WORKING YET)

**What we want:** Agent works towards a goal, tracks progress, knows when done.

```python
# THIS DOESN'T WORK YET - This is what we need

agent = Agent(
    name="ResearchAgent",
    role="Researcher",
    objective="Research topics thoroughly",
    llm=llm,
    tools=[web_search, calculator]
)

# Set a goal
goal = Goal(
    objective="Find the weather in Paris and calculate the average temperature for the week"
)

# Agent works autonomously towards goal
result = await agent.achieve_goal(goal)

# What should happen:
# 1. Agent creates plan: [search weather, extract temps, calculate average]
# 2. Agent executes plan step by step
# 3. Agent tracks progress: 0% → 33% → 66% → 100%
# 4. Agent knows when goal is achieved
# 5. Returns final result
```

**Current problem:** No `achieve_goal()` method, no goal tracking, no progress monitoring.

---

### Example 3: Multi-Step Reasoning Agent (NOT WORKING YET)

**What we want:** Agent reasons through problems step-by-step, maintains reasoning chain.

```python
# THIS DOESN'T WORK YET - This is what we need

agent = Agent(
    name="ProblemSolver",
    role="Problem Solver",
    objective="Solve complex problems",
    llm=llm,
    tools=[calculator, web_search]
)

task = {"id": "task1", "objective": "A store has 50 apples. They sell 20. Then they get 30 more. How many do they have now?"}

result = await agent.execute(task)

# What should happen:
# 1. Agent thinks: "I need to solve this step by step"
# 2. Agent reasons:
#    - Thought 1: "Start with 50 apples"
#    - Thought 2: "Sell 20, so 50 - 20 = 30"
#    - Thought 3: "Get 30 more, so 30 + 30 = 60"
#    - Thought 4: "Final answer is 60"
# 3. Agent uses calculator tool for calculations
# 4. Agent maintains reasoning chain throughout
# 5. Returns: "The store has 60 apples. Here's my reasoning: ..."
```

**Current problem:** Agent just calls LLM once, no reasoning loop, no thought chain.

---

### Example 4: Adaptive Planning Agent (NOT WORKING YET)

**What we want:** Agent creates plan, adapts it based on results, handles failures.

```python
# THIS DOESN'T WORK YET - This is what we need

agent = Agent(
    name="TaskAgent",
    role="Task Executor",
    objective="Complete complex tasks",
    llm=llm,
    tools=[web_search, email_sender, calendar],
    config=AgentConfig(use_planning=True, use_llm_for_planning=True)
)

task = {
    "id": "task1",
    "objective": "Find the next meeting time, send email to team, and update calendar"
}

result = await agent.execute(task)

# What should happen:
# 1. Agent creates plan:
#    Step 1: Find next meeting time (use calendar tool)
#    Step 2: Send email to team (use email tool)
#    Step 3: Update calendar (use calendar tool)
#
# 2. Agent executes Step 1:
#    - Calls calendar tool
#    - Gets result: "Next meeting: Dec 25, 2 PM"
#    - Updates context
#
# 3. Agent executes Step 2:
#    - Uses context from Step 1
#    - Calls email tool with meeting time
#    - Gets result: "Email sent"
#
# 4. Agent executes Step 3:
#    - Uses context from Steps 1 and 2
#    - Calls calendar tool to update
#    - Gets result: "Calendar updated"
#
# 5. If any step fails:
#    - Agent adapts plan
#    - Retries with different approach
#    - Or reports failure
#
# 6. Returns final result with all steps completed
```

**Current problem:** 
- Planning exists but doesn't work (see issues doc)
- Context not passed between steps
- No plan adaptation
- No failure handling

---

### Example 5: Memory-Aware Agent (NOT WORKING YET)

**What we want:** Agent remembers previous conversations, uses memory for decisions.

```python
# THIS DOESN'T WORK YET - This is what we need

agent = Agent(
    name="Assistant",
    role="Personal Assistant",
    objective="Help users",
    llm=llm,
    memory=memory_system  # Memory system
)

# First conversation
task1 = {"id": "task1", "objective": "My name is John and I like Python"}
result1 = await agent.execute(task1)
# Agent stores: "User is John, likes Python"

# Second conversation (later)
task2 = {"id": "task2", "objective": "What programming language do I like?"}
result2 = await agent.execute(task2)
# Agent should remember: "You like Python" (from memory)

# Third conversation
task3 = {"id": "task3", "objective": "Recommend a Python library for me"}
result3 = await agent.execute(task3)
# Agent uses memory: "John likes Python, so recommend Python libraries"
```

**Current problem:** Memory exists but not used in execution, no context retrieval.

---

## How Users Should Use the Framework (Target)

### Simple Use Case: Basic Agent

```python
from nucleusiq.agents import Agent
from nucleusiq.providers.llms.openai.nb_openai import BaseOpenAI
from nucleusiq.core.tools import BaseTool

# 1. Create tools
calculator = CalculatorTool()

# 2. Create LLM
llm = BaseOpenAI(api_key="...", model="gpt-4")

# 3. Create agent
agent = Agent(
    name="MyAgent",
    role="Assistant",
    objective="Help users",
    llm=llm,
    tools=[calculator]
)

# 4. Initialize
await agent.initialize()

# 5. Execute task
task = Task(id="task1", objective="What is 5 + 3?")
result = await agent.execute(task)
```

**This should work** - Simple agent with tools.

---

### Advanced Use Case: Goal-Oriented Agent

```python
# Create agent with planning
agent = Agent(
    name="ResearchAgent",
    role="Researcher",
    objective="Research topics thoroughly",
    llm=llm,
    tools=[web_search, calculator],
    config=AgentConfig(
        use_planning=True,
        use_llm_for_planning=True,
        max_iterations=10
    )
)

# Set a goal
goal = Goal(
    objective="Find the weather in Paris and calculate the average temperature"
)

# Agent works autonomously
result = await agent.achieve_goal(goal)

# Agent:
# - Creates plan automatically
# - Executes plan step by step
# - Tracks progress
# - Adapts if needed
# - Returns when goal achieved
```

**This doesn't work yet** - Need `Goal` class and `achieve_goal()` method.

---

### Advanced Use Case: Reasoning Agent

```python
# Create reasoning agent
agent = Agent(
    name="ProblemSolver",
    role="Problem Solver",
    objective="Solve complex problems",
    llm=llm,
    tools=[calculator],
    config=AgentConfig(
        reasoning_mode="chain_of_thought",
        max_reasoning_steps=10
    )
)

task = Task(
    id="task1",
    objective="A store has 50 apples. They sell 20. Then they get 30 more. How many?"
)

result = await agent.execute(task)

# Agent should:
# - Reason step by step
# - Show reasoning chain
# - Use tools when needed
# - Return answer with reasoning
```

**This doesn't work yet** - Need reasoning loop, thought chain management.

---

### Advanced Use Case: Memory-Aware Agent

```python
from nucleusiq.core.memory import VectorMemory

# Create memory system
memory = VectorMemory(embedding_model="...")

# Create agent with memory
agent = Agent(
    name="Assistant",
    role="Personal Assistant",
    objective="Help users",
    llm=llm,
    memory=memory
)

# First conversation
task1 = Task(id="task1", objective="My name is John")
await agent.execute(task1)
# Agent stores in memory

# Second conversation
task2 = Task(id="task2", objective="What's my name?")
result = await agent.execute(task2)
# Agent retrieves from memory: "Your name is John"
```

**This partially works** - Memory exists but not actively used in execution.

---

## What Needs to Change

### 1. Core Agent Loop (NEW)

**Current:** Single LLM call → tool → LLM call → done

**Needed:** Autonomous loop that continues until goal achieved

```python
# NEW: Core agent loop
async def execute(self, task):
    goal = self._create_goal_from_task(task)
    
    while not goal.is_achieved():
        # 1. Evaluate current state
        state = self._evaluate_state(goal)
        
        # 2. Decide next action
        action = await self._decide_action(state, goal)
        
        # 3. Execute action
        result = await self._execute_action(action)
        
        # 4. Process observation
        observation = self._process_observation(result)
        
        # 5. Update state
        self._update_state(action, observation)
        
        # 6. Check if goal achieved
        if self._check_goal_achieved(goal, state):
            break
    
    return self._get_final_result(goal)
```

---

### 2. Goal Tracking (NEW)

**Current:** No goal tracking

**Needed:** Track progress towards goal

```python
# NEW: Goal class
class Goal:
    objective: str
    current_state: Dict
    progress: float  # 0.0 to 1.0
    achieved: bool
    steps_taken: List[Action]
    
    def check_progress(self):
        # Evaluate if closer to goal
        # Update progress
        # Check if achieved
```

---

### 3. Decision Engine (NEW)

**Current:** LLM decides via function calls

**Needed:** Agent decides based on state and goal

```python
# NEW: Decision engine
async def _decide_action(self, state, goal):
    # Evaluate current state
    # Consider available actions
    # Choose best action based on goal
    # Return action
```

---

### 4. State Management (NEW)

**Current:** Basic state (EXECUTING, COMPLETED, etc.)

**Needed:** Full state with beliefs, intentions, history

```python
# NEW: Agent state
class AgentState:
    current_goal: Goal
    action_history: List[Action]
    observation_history: List[Observation]
    beliefs: Dict[str, Any]  # What agent believes
    intentions: List[Intention]  # What agent intends to do
```

---

### 5. Reasoning Loop (NEW)

**Current:** Single LLM call

**Needed:** Multi-step reasoning

```python
# NEW: Reasoning loop
async def reason(self, problem):
    reasoning_chain = []
    while not solved:
        thought = await self.think(problem, reasoning_chain)
        reasoning_chain.append(thought)
        action = await self.decide_action(thought)
        result = await self.execute_action(action)
        if self.is_solved(result):
            break
```

---

### 6. Working Planning (FIX)

**Current:** Planning exists but broken

**Needed:** Planning that actually works

```python
# FIX: Make planning work
async def plan(self, task):
    if self.config.use_llm_for_planning:
        # Actually call LLM to create plan
        plan = await self._create_llm_plan(task)
    else:
        plan = await self._create_basic_plan(task)
    return plan

async def _execute_plan(self, task, plan):
    context = {}
    for step in plan.steps:
        # Use context from previous steps
        result = await self._execute_step(step, context)
        context.update(result)
    return final_result
```

---

## Summary: What Users Will Experience

### Current (What Works)
```python
# Simple agent with tools
agent = Agent(..., tools=[tool])
result = await agent.execute(task)
# Works: LLM call → tool → LLM call → result
```

### Target (What We Need)
```python
# Goal-oriented agent
agent = Agent(..., config=AgentConfig(use_planning=True))
goal = Goal(objective="...")
result = await agent.achieve_goal(goal)
# Works: Agent creates plan → executes → tracks progress → adapts → achieves goal

# Reasoning agent
agent = Agent(..., config=AgentConfig(reasoning_mode="chain_of_thought"))
result = await agent.execute(task)
# Works: Agent reasons step-by-step → uses tools → maintains reasoning chain → returns answer

# Memory-aware agent
agent = Agent(..., memory=memory)
result = await agent.execute(task)
# Works: Agent retrieves relevant memories → uses in decisions → stores new memories
```

---

## Key Changes Needed

1. **Add Goal class** - Track goals and progress
2. **Add Decision Engine** - Agent decides actions
3. **Add State Manager** - Track full agent state
4. **Add Reasoning Loop** - Multi-step reasoning
5. **Fix Planning** - Make it actually work
6. **Use Memory** - Actively use memory in decisions
7. **Add Reflection** - Evaluate and learn from actions

---

## User Experience Comparison

| Feature | Current | Target |
|---------|---------|--------|
| **Simple task** | ✅ Works | ✅ Works (same) |
| **Goal tracking** | ❌ No | ✅ Yes |
| **Multi-step reasoning** | ❌ No | ✅ Yes |
| **Planning** | ❌ Broken | ✅ Works |
| **Memory usage** | ❌ Not used | ✅ Actively used |
| **Autonomous behavior** | ❌ No | ✅ Yes |
| **Adaptation** | ❌ No | ✅ Yes |

---

## Next Steps

1. **Keep current simple usage working** - Don't break what works
2. **Add new capabilities incrementally** - Build on top
3. **Make it optional** - Users can use simple or advanced features
4. **Provide clear examples** - Show how to use each feature

The framework should support both:
- **Simple usage:** Just LLM + tools (current, works)
- **Advanced usage:** Full agent capabilities (target, needs building)

