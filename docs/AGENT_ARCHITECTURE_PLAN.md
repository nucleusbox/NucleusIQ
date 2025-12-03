# Agent Architecture Plan - Comprehensive Refactoring

## Overview

This document outlines the comprehensive plan to establish correct relationships between Task, Prompt, and Plan, add Guardrails, and support multiple agent types through configuration rather than separate classes.

---

## 1. Task, Prompt, and Plan Relationship

### Current Issues

1. **Task**: Used everywhere, clear purpose ✅
2. **Prompt**: Used for message formatting, but relationship with Task is unclear
3. **Plan**: Exists but not used by execute() - orphaned functionality ❌

### Proposed Relationship

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Execution Flow                      │
└─────────────────────────────────────────────────────────────┘

User Input
    ↓
┌─────────────┐
│    Task     │  {"id": "task1", "objective": "What is 5 + 3?"}
└──────┬──────┘
       │
       │ agent.execute(task)
       ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Plan (Optional)                                    │
│  - plan(task) → Plan                                        │
│  - Breaks down complex tasks into steps                     │
│  - Can use LLM or simple decomposition                      │
└─────────────────────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Build Messages (Uses Prompt)                       │
│  - prompt.system → Agent's role/instructions                │
│  - prompt.user → Template/instruction (optional)            │
│  - task.objective → User's actual request                   │
│  - plan steps → If plan exists, add plan context            │
└─────────────────────────────────────────────────────────────┘
       │
       ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Execute (Uses Plan if available)                   │
│  - If plan has multiple steps: Execute plan steps           │
│  - Otherwise: Execute directly                              │
│  - Uses prompt for message formatting                       │
│  - Uses task for actual user request                        │
└─────────────────────────────────────────────────────────────┘
       │
       ↓
┌─────────────┐
│   Result    │  Final answer or execution result
└─────────────┘
```

### Implementation Status

✅ **COMPLETED:**
- Task class created (`src/nucleusiq/agents/task.py`)
- Plan classes created (`src/nucleusiq/agents/plan.py`)
- Plan integrated into `execute()` flow
- Optional planning via `config.use_planning`
- Backward compatible with dict/list

---

## 2. Guardrails System

### Overview

Guardrails provide hooks for validation, safety checks, and custom processing before and after agent execution.

### Design

**BaseGuardrail** (`src/nucleusiq/agents/guardrails/base_guardrail.py`):

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from nucleusiq.agents.task import Task

class BaseGuardrail(ABC):
    """Base class for all guardrails."""
    
    @abstractmethod
    async def before_execute(self, task: Task, agent: 'Agent') -> Optional[Task]:
        """
        Called before task execution.
        
        Args:
            task: Task to be executed
            agent: Agent instance
            
        Returns:
            Modified task (or None to cancel execution)
        """
        pass
    
    @abstractmethod
    async def after_execute(self, task: Task, result: Any, agent: 'Agent') -> Any:
        """
        Called after task execution.
        
        Args:
            task: Executed task
            result: Execution result
            agent: Agent instance
            
        Returns:
            Modified result
        """
        pass
```

### Built-in Guardrails

1. **ValidationGuardrail** - Validates task format and content
2. **SafetyGuardrail** - Checks for unsafe content/actions
3. **RateLimitGuardrail** - Enforces rate limiting
4. **CostGuardrail** - Monitors and limits API costs
5. **OutputGuardrail** - Validates and sanitizes output

### Integration

**Agent Class Updates:**

```python
class Agent(BaseAgent):
    guardrails: List[BaseGuardrail] = Field(default_factory=list)
    
    async def execute(self, task: Union[Task, Dict[str, Any]]) -> Any:
        # Convert to Task if needed
        if isinstance(task, dict):
            task = Task.from_dict(task)
        
        # Pre-execution guardrails
        for guardrail in self.guardrails:
            task = await guardrail.before_execute(task, self)
            if task is None:
                raise ValueError("Execution cancelled by guardrail")
        
        # Execute task
        result = await self._execute_direct(task)
        
        # Post-execution guardrails
        for guardrail in self.guardrails:
            result = await guardrail.after_execute(task, result, self)
        
        return result
```

### Implementation Plan

**Phase 1: Base Infrastructure (6-8 hours)**
- [ ] Create `BaseGuardrail` abstract class
- [ ] Add `guardrails` field to `Agent`
- [ ] Integrate guardrails into `execute()` flow
- [ ] Add tests

**Phase 2: Built-in Guardrails (12-17 hours)**
- [ ] `ValidationGuardrail` - Task validation
- [ ] `SafetyGuardrail` - Content safety checks
- [ ] `RateLimitGuardrail` - Rate limiting
- [ ] `CostGuardrail` - Cost monitoring
- [ ] `OutputGuardrail` - Output validation
- [ ] Tests for each guardrail

**Phase 3: Documentation (2-3 hours)**
- [ ] Guardrail usage guide
- [ ] Examples
- [ ] API documentation

**Total Estimated Time: 12-17 hours (1.5-2 days)**

---

## 3. Multiple Agent Types

### Overview

Support multiple agent types (ReAct, ReWoo, Reflection, CoT, etc.) through configuration rather than separate classes.

### Strategy Pattern

**AgentType Enum** (`src/nucleusiq/agents/config/agent_type.py`):

```python
from enum import Enum

class AgentType(str, Enum):
    """Agent execution types."""
    SIMPLE = "simple"           # Basic agent (current default)
    REACT = "react"             # ReAct (Reasoning + Acting)
    REWOO = "rewoo"             # ReWoo (Reasoning without Observation)
    REFLECTION = "reflection"   # Reflection (self-critique)
    COT = "cot"                 # Chain of Thought
    UTILITY = "utility"         # Utility-based
    GOAL = "goal"               # Goal-based
    MODEL = "model"             # Model-based
    REFLEX = "reflex"           # Reflex
    SIMPLE_REFLEX = "simple_reflex"  # Simple Reflex
    REACTIVE = "reactive"       # Reactive
    PROACTIVE = "proactive"     # Proactive
```

**AgentConfig Updates:**

```python
class AgentConfig(BaseModel):
    # ... existing fields ...
    agent_type: AgentType = Field(
        default=AgentType.SIMPLE,
        description="Agent execution type"
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum iterations for iterative agents"
    )
```

**Agent Execution Strategy:**

```python
class Agent(BaseAgent):
    def _get_execution_strategy(self):
        """Get execution strategy based on agent_type."""
        strategies = {
            AgentType.SIMPLE: self._execute_simple,
            AgentType.REACT: self._execute_react,
            AgentType.REWOO: self._execute_rewoo,
            AgentType.REFLECTION: self._execute_reflection,
            AgentType.COT: self._execute_cot,
            # ... etc
        }
        return strategies.get(self.config.agent_type, self._execute_simple)
    
    async def execute(self, task: Union[Task, Dict[str, Any]]) -> Any:
        strategy = self._get_execution_strategy()
        return await strategy(task)
```

### Agent Type Implementations

#### 1. ReAct (Reasoning + Acting) ✅

**Status:** Already implemented as `ReActAgent` class

**Pattern:**
- Thought → Action → Observation → Loop

**Implementation:**
- Can be moved to strategy pattern
- Or keep as separate class for now

#### 2. ReWoo (Reasoning without Observation)

**Pattern:**
- Plan → Execute → No observation loop
- Pre-planned execution

**Implementation:**
```python
async def _execute_rewoo(self, task: Union[Task, Dict[str, Any]]) -> Any:
    """ReWoo: Plan all steps first, then execute."""
    # Create complete plan
    plan = await self.plan(task)
    
    # Execute all steps without observation
    results = []
    for step in plan.steps:
        result = await self._execute_step(step)
        results.append(result)
    
    return results[-1]  # Return final result
```

#### 3. Reflection

**Pattern:**
- Execute → Reflect → Refine → Execute again

**Implementation:**
```python
async def _execute_reflection(self, task: Union[Task, Dict[str, Any]]) -> Any:
    """Reflection: Execute, critique, refine, re-execute."""
    # Initial execution
    result = await self._execute_simple(task)
    
    # Reflection: Critique the result
    critique = await self._reflect_on_result(task, result)
    
    # Refine and re-execute if needed
    if critique.needs_refinement:
        refined_task = self._refine_task(task, critique)
        result = await self._execute_simple(refined_task)
    
    return result
```

#### 4. Chain of Thought (CoT)

**Pattern:**
- Explicit reasoning steps in prompt
- Step-by-step thinking

**Implementation:**
```python
async def _execute_cot(self, task: Union[Task, Dict[str, Any]]) -> Any:
    """Chain of Thought: Explicit reasoning steps."""
    # Use CoT prompt
    cot_prompt = PromptFactory.create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
    cot_prompt.configure(
        system="Think step by step...",
        user=task.objective
    )
    
    # Execute with CoT prompt
    return await self._execute_with_prompt(task, cot_prompt)
```

### Implementation Plan

**Phase 1: Strategy Infrastructure (4-6 hours)**
- [ ] Create `AgentType` enum
- [ ] Add `agent_type` to `AgentConfig`
- [ ] Create strategy dispatcher in `Agent`
- [ ] Add tests

**Phase 2: Implement Agent Types (20-26 hours)**
- [x] ReAct (already exists)
- [ ] ReWoo
- [ ] Reflection
- [ ] CoT
- [ ] Utility-based
- [ ] Goal-based
- [ ] Others as needed

**Phase 3: Documentation (2-3 hours)**
- [ ] Agent type guide
- [ ] Examples for each type
- [ ] When to use which type

**Total Estimated Time: 20-26 hours (2.5-3.5 days)**

---

## Implementation Timeline

### Phase 1: Task, Prompt, Plan ✅ COMPLETED
- **Time:** 6-9 hours
- **Status:** ✅ Done

### Phase 2: Guardrails
- **Time:** 12-17 hours (1.5-2 days)
- **Status:** Not started

### Phase 3: Multiple Agent Types
- **Time:** 20-26 hours (2.5-3.5 days)
- **Status:** Partial (ReAct exists)

### Total Estimated Time: 38-52 hours (5-7 days)

---

## Success Criteria

### Task, Prompt, Plan ✅
- ✅ Plan integrated into execute()
- ✅ Clear relationship documented
- ✅ Backward compatible
- ✅ All tests passing

### Guardrails
- [ ] BaseGuardrail interface implemented
- [ ] 5+ built-in guardrails
- [ ] Integration with Agent
- [ ] Tests and documentation

### Multiple Agent Types
- [ ] Strategy pattern implemented
- [ ] 5+ agent types supported
- [ ] Configuration-based selection
- [ ] Tests and documentation

---

## Files to Create

### Guardrails
- `src/nucleusiq/agents/guardrails/__init__.py`
- `src/nucleusiq/agents/guardrails/base_guardrail.py`
- `src/nucleusiq/agents/guardrails/validation_guardrail.py`
- `src/nucleusiq/agents/guardrails/safety_guardrail.py`
- `src/nucleusiq/agents/guardrails/rate_limit_guardrail.py`
- `src/nucleusiq/agents/guardrails/cost_guardrail.py`
- `src/nucleusiq/agents/guardrails/output_guardrail.py`

### Agent Types
- `src/nucleusiq/agents/config/agent_type.py`
- `src/nucleusiq/agents/strategies/__init__.py`
- `src/nucleusiq/agents/strategies/react_strategy.py`
- `src/nucleusiq/agents/strategies/rewoo_strategy.py`
- `src/nucleusiq/agents/strategies/reflection_strategy.py`
- `src/nucleusiq/agents/strategies/cot_strategy.py`

---

*Last Updated: After Task/Plan implementation*

