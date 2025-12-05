# Agent Orchestration - Critical Issues Found

## Overview

After reviewing the actual implementation, several critical issues were identified that prevent the orchestration from working as described. This document lists all issues and their fixes.

---

## Critical Issues

### Issue 1: `plan()` Method Never Uses LLM Planning ❌

**Location:** `src/nucleusiq/agents/agent.py:87-113`

**Problem:**
```python
async def plan(self, task: Union[Task, Dict[str, Any]]) -> Plan:
    # ... always returns basic plan
    step = PlanStep(step=1, action="execute", task=task)
    return Plan(steps=[step], task=task)
```

The `plan()` method **always** returns a basic one-step plan. It never calls `_create_llm_plan()` even though:
- `_create_llm_plan()` exists and is implemented
- The code references `use_llm_for_planning` config
- But `use_llm_for_planning` doesn't exist in `AgentConfig`

**Impact:** LLM-based planning can never be triggered.

**Fix Required:**
1. Add `use_llm_for_planning` to `AgentConfig`
2. Update `plan()` to check this flag and call `_create_llm_plan()` when enabled

---

### Issue 2: Plan Context Not Passed to `_execute_direct()` ❌

**Location:** `src/nucleusiq/agents/agent.py:580`

**Problem:**
```python
async def _execute_direct(self, task: Union[Task, Dict[str, Any]]) -> Any:
    # ...
    messages = self._build_messages(task_dict)  # ❌ No plan parameter!
```

`_build_messages()` accepts a `plan` parameter, but `_execute_direct()` never passes it. This means:
- Plan context is never included in messages
- Even if a plan exists, it won't be sent to the LLM

**Impact:** Plans are created but never used in execution.

**Fix Required:**
- Update `_execute_direct()` to accept and pass `plan` parameter
- Or remove plan parameter from `_build_messages()` if not needed

---

### Issue 3: `_format_plan()` Expects Dict But Gets Plan Object ❌

**Location:** `src/nucleusiq/agents/agent.py:309-327`

**Problem:**
```python
def _format_plan(self, plan: Union[Plan, List[Dict[str, Any]]]) -> str:
    plan_lines = []
    for step in plan:  # ❌ If Plan object, this iterates over Plan, not plan.steps
        step_num = step.get("step", 0)  # ❌ PlanStep doesn't have .get()
        action = step.get("action", "")
```

When `plan` is a `Plan` object:
- Iterating over `plan` doesn't work (Plan is not iterable directly)
- Should iterate over `plan.steps`
- `PlanStep` objects don't have `.get()` method (they're Pydantic models with attributes)

**Impact:** Plan formatting will fail when using Plan objects.

**Fix Required:**
- Handle both Plan object and list of dicts properly
- Access attributes correctly for PlanStep objects

---

### Issue 4: Recursive `_execute_direct()` Calls in Plan Execution ❌

**Location:** `src/nucleusiq/agents/agent.py:376-378`

**Problem:**
```python
async def _execute_plan(self, task, plan):
    for step in plan.steps:
        if action == "execute":
            result = await self._execute_direct(step_task)  # ❌ Full LLM call!
```

When executing a plan step with `action == "execute"`:
- It calls `_execute_direct()` which does a **full LLM call**
- This means each step triggers a complete LLM interaction
- This might be intended, but it's inefficient and the context from previous steps isn't used

**Impact:** 
- Each plan step does a full LLM call (expensive)
- Previous step results aren't passed to next steps
- Context building in `_execute_plan()` is unused

**Fix Required:**
- Either pass context to `_execute_direct()` 
- Or create a simpler step execution that doesn't do full LLM calls
- Or use context from previous steps in subsequent LLM calls

---

### Issue 5: Context from Previous Steps Not Used ❌

**Location:** `src/nucleusiq/agents/agent.py:352-395`

**Problem:**
```python
async def _execute_plan(self, task, plan):
    context = {}
    results = []
    
    for step in plan.steps:
        # ... execute step ...
        results.append(result)
        context[f"step_{step_num}"] = result  # ❌ Context built but never used!
        context[f"step_{step_num}_action"] = action
```

The `context` dictionary is built with results from each step, but:
- It's never passed to `_execute_direct()` 
- It's never passed to `_execute_tool()`
- It's never used in subsequent LLM calls

**Impact:** Multi-step plans don't share context between steps.

**Fix Required:**
- Pass context to step execution methods
- Include context in messages for subsequent steps

---

### Issue 6: `_execute_plan()` Doesn't Handle Plan in Messages ❌

**Location:** `src/nucleusiq/agents/agent.py:376-378`

**Problem:**
When `_execute_plan()` calls `_execute_direct()` for a step:
- `_execute_direct()` calls `_build_messages()` without a plan
- Even if we pass plan, it would be the full plan, not step-specific
- Plan context should be included in messages for each step

**Impact:** Steps don't know they're part of a larger plan.

---

### Issue 7: Tool Execution in Plan Doesn't Use Context ❌

**Location:** `src/nucleusiq/agents/agent.py:379-382`

**Problem:**
```python
elif action in [t.name for t in self.tools if hasattr(t, 'name')]:
    tool_args = step.args or {}
    result = await self._execute_tool(action, tool_args)  # ❌ No context!
```

When executing a tool in a plan step:
- Only uses `step.args` from the plan
- Doesn't include context from previous steps
- Tools can't access results from earlier steps

**Impact:** Tools in multi-step plans can't use previous step results.

---

## Summary of Issues

| Issue | Severity | Impact | Fix Complexity |
|-------|----------|--------|----------------|
| 1. LLM planning never triggered | 🔴 Critical | Planning always basic | Low |
| 2. Plan not passed to messages | 🔴 Critical | Plans ignored in execution | Low |
| 3. `_format_plan()` type mismatch | 🟡 High | Plan formatting fails | Low |
| 4. Recursive LLM calls in plan | 🟡 High | Inefficient, expensive | Medium |
| 5. Context not used | 🟡 High | Steps isolated | Medium |
| 6. Plan context in messages | 🟡 High | Steps don't know plan | Medium |
| 7. Tool context in plan | 🟡 High | Tools isolated | Medium |

---

## Recommended Fixes

### Fix 1: Enable LLM Planning

```python
# In AgentConfig
use_llm_for_planning: bool = Field(
    default=False,
    description="Use LLM to generate plans (requires use_planning=True)"
)

# In plan() method
async def plan(self, task: Union[Task, Dict[str, Any]]) -> Plan:
    if isinstance(task, dict):
        task = Task.from_dict(task)
    
    # Check if LLM planning is enabled
    if self.config.use_planning and self.config.use_llm_for_planning and self.llm:
        context = await self._get_context(task)
        return await self._create_llm_plan(task, context)
    else:
        return await self._create_basic_plan(task)
```

### Fix 2: Pass Plan to Messages

```python
async def _execute_direct(self, task, plan=None):
    # ...
    messages = self._build_messages(task_dict, plan=plan)
```

### Fix 3: Fix `_format_plan()`

```python
def _format_plan(self, plan: Union[Plan, List[Dict[str, Any]]]) -> str:
    plan_lines = []
    
    # Handle Plan object
    if isinstance(plan, Plan):
        steps = plan.steps
    else:
        steps = plan
    
    for step in steps:
        # Handle PlanStep object or dict
        if isinstance(step, PlanStep):
            step_num = step.step
            action = step.action
            details = step.details or ""
        else:
            step_num = step.get("step", 0)
            action = step.get("action", "")
            details = step.get("details", "")
        
        plan_lines.append(f"Step {step_num}: {action}")
        if details:
            plan_lines.append(f"  {details.strip()}")
    
    return "\n".join(plan_lines)
```

### Fix 4: Use Context in Plan Execution

```python
async def _execute_plan(self, task, plan):
    context = {}
    results = []
    
    for step in plan.steps:
        # ... execute step ...
        
        # Pass context to step execution
        if action == "execute":
            result = await self._execute_direct(step_task, plan=None, context=context)
        elif action in [t.name for t in self.tools]:
            # Include context in tool args
            tool_args = {**(step.args or {}), **context}
            result = await self._execute_tool(action, tool_args)
        
        # Update context for next steps
        context[f"step_{step_num}"] = result
        results.append(result)
```

---

## Testing Required

After fixes, test:
1. ✅ LLM planning is triggered when `use_llm_for_planning=True`
2. ✅ Plan context is included in messages
3. ✅ Plan formatting works with Plan objects
4. ✅ Context is passed between plan steps
5. ✅ Tools can access previous step results
6. ✅ Multi-step plans execute correctly

---

## Conclusion

The orchestration **theory is sound**, but the **implementation has critical gaps**:

1. **Planning never uses LLM** - Always basic
2. **Plans are created but ignored** - Not passed to execution
3. **Context is built but unused** - Steps are isolated
4. **Type mismatches** - Plan objects not handled correctly

These issues prevent the orchestration from working as designed. The fixes are straightforward but necessary for the system to function correctly.

