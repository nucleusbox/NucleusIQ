# NucleusIQ TODO List - Comprehensive Development Plan

*Last Updated: After Architecture Planning*

---

## Overview

This TODO list consolidates all planned work from architecture documents, development notes, and planning materials. Items are organized by priority and phase.

---

## 🎯 Phase 1: Gearbox Strategy (Execution Modes) - 2-3 weeks

### Week 1: Core Infrastructure

#### 1.1 AgentConfig Updates
- [ ] Add `execution_mode` field to `AgentConfig` class
  - Type: `Literal["direct", "standard", "autonomous"]`
  - Default: `"standard"` (maintains backward compatibility)
  - Location: `src/nucleusiq/agents/config/agent_config.py`
- [ ] Add `enable_memory` field to `AgentConfig`
  - Type: `bool`
  - Default: `True`
  - Description: Enable memory for standard/autonomous modes
- [ ] Add `require_quality_check` field to `AgentConfig`
  - Type: `bool`
  - Default: `False`
  - Description: Require quality check (autonomous mode)
- [ ] Update `use_planning` field documentation (mark as deprecated, use execution_mode)
- [ ] Add validation for execution_mode values
- [ ] Tests for AgentConfig with new fields

#### 1.2 Executor Component
- [ ] Create `src/nucleusiq/agents/components/__init__.py`
- [ ] Create `src/nucleusiq/agents/components/executor.py`
- [ ] Implement `Executor` class:
  - [ ] `__init__(llm, tools)` - Initialize with LLM and tools
  - [ ] `execute(fn_call)` - Execute single tool call
  - [ ] `execute_step(step, context)` - Execute plan step with context
  - [ ] Tool validation and error handling
  - [ ] Support for native tools (OpenAITool) vs BaseTool
- [ ] Extract tool execution logic from `Agent._execute_direct()`
- [ ] Tests for Executor component
  - [ ] Test tool execution
  - [ ] Test error handling
  - [ ] Test context passing
  - [ ] Test native vs custom tools

#### 1.3 Direct Mode Implementation
- [ ] Implement `Agent._run_direct(task)` method
  - [ ] Build messages (include prompt.system/user)
  - [ ] Call LLM (no tools)
  - [ ] Extract and return content
  - [ ] Handle errors gracefully
- [ ] Add mode routing in `Agent.execute()`
  - [ ] Route to `_run_direct()` when `execution_mode="direct"`
- [ ] Tests for Direct mode
  - [ ] Simple chat without tools
  - [ ] With prompt.system/user
  - [ ] Error handling
  - [ ] Performance (should be fast)

#### 1.4 Standard Mode Refactoring
- [ ] Refactor `Agent._execute_direct()` → `Agent._run_standard()`
  - [ ] Use Executor component for tool execution
  - [ ] Keep current behavior (backward compatible)
  - [ ] Support multiple tool calls in loop
  - [ ] Fire-and-forget behavior (no retries)
- [ ] Update mode routing in `Agent.execute()`
  - [ ] Route to `_run_standard()` when `execution_mode="standard"`
- [ ] Tests for Standard mode
  - [ ] Single tool call
  - [ ] Multiple tool calls
  - [ ] Error handling (fire-and-forget)
  - [ ] Memory integration (if enabled)

#### 1.5 Mode Routing
- [ ] Update `Agent.execute()` to route based on `execution_mode`
  - [ ] Direct mode → `_run_direct()`
  - [ ] Standard mode → `_run_standard()`
  - [ ] Autonomous mode → `_run_autonomous()` (Week 2)
- [ ] Add logging for mode selection
- [ ] Tests for mode routing
  - [ ] All three modes route correctly
  - [ ] Default mode (standard) works

---

### Week 2: Autonomous Mode

#### 2.1 Planner Component
- [ ] Create `src/nucleusiq/agents/components/planner.py`
- [ ] Implement `Planner` class:
  - [ ] `__init__(llm)` - Initialize with LLM
  - [ ] `create_plan(task, tools)` - Create execution plan
  - [ ] `refine_step(step, feedback)` - Refine step based on critique
  - [ ] `adapt_plan(plan, context, goal)` - Adapt plan based on results
  - [ ] Fix existing planning logic (currently broken)
- [ ] Integrate with existing `Agent.plan()` method
- [ ] Support for LLM-based planning
- [ ] Support for simple decomposition planning
- [ ] Tests for Planner component
  - [ ] Plan creation
  - [ ] Step refinement
  - [ ] Plan adaptation
  - [ ] Error handling

#### 2.2 Critic Component
- [ ] Create `src/nucleusiq/agents/components/critic.py`
- [ ] Create `Critique` Pydantic model:
  - [ ] `passed: bool` - Whether critique passed
  - [ ] `feedback: str` - Feedback message
  - [ ] `score: Optional[float]` - Quality score
- [ ] Implement `Critic` class:
  - [ ] `__init__(llm)` - Initialize with LLM
  - [ ] `review(step, result, original_task)` - Review step result
  - [ ] LLM-based quality checking
  - [ ] Goal alignment checking
  - [ ] Error detection
- [ ] Tests for Critic component
  - [ ] Quality checking
  - [ ] Goal alignment
  - [ ] Error detection
  - [ ] Feedback generation

#### 2.3 Autonomous Mode Implementation
- [ ] Implement `Agent._run_autonomous(task)` method:
  - [ ] Create plan using Planner
  - [ ] Execute plan steps with self-correction loop
  - [ ] Use Critic for quality checking
  - [ ] Retry with refinement on failure
  - [ ] Track context between steps
  - [ ] Return final result when all steps complete
- [ ] Self-correction loop:
  - [ ] Execute step
  - [ ] Critique result
  - [ ] If failed: refine step and retry (up to max_retries)
  - [ ] If passed: continue to next step
- [ ] Context management between steps
- [ ] Tests for Autonomous mode
  - [ ] Multi-step plan execution
  - [ ] Self-correction on failure
  - [ ] Quality checking
  - [ ] Context passing
  - [ ] Max retries handling

#### 2.4 Component Initialization
- [ ] Update `Agent.__init__()` to initialize components based on mode:
  - [ ] Always initialize Executor
  - [ ] Initialize Planner and Critic only if `execution_mode="autonomous"`
  - [ ] Initialize Memory if `enable_memory=True`
- [ ] Lazy loading (only load heavy components when needed)
- [ ] Tests for component initialization
  - [ ] Direct mode: Only Executor
  - [ ] Standard mode: Executor + Memory (if enabled)
  - [ ] Autonomous mode: All components

---

### Week 3: Integration & Polish

#### 3.1 Memory Integration
- [ ] Integrate Memory into Standard mode
  - [ ] Store conversation history
  - [ ] Store tool results
  - [ ] Retrieve relevant context
- [ ] Integrate Memory into Autonomous mode
  - [ ] Store plan steps
  - [ ] Store intermediate results
  - [ ] Retrieve context for planning
- [ ] Make Memory optional (config.enable_memory)
- [ ] Tests for Memory integration
  - [ ] Storage and retrieval
  - [ ] Context usage in execution
  - [ ] Optional memory (disabled)

#### 3.2 Error Handling
- [ ] Comprehensive error handling for all modes
- [ ] Error recovery strategies
- [ ] User-friendly error messages
- [ ] Logging for debugging
- [ ] Tests for error handling
  - [ ] LLM errors
  - [ ] Tool errors
  - [ ] Planning errors
  - [ ] Memory errors

#### 3.3 Documentation
- [ ] Update Agent class docstring
- [ ] Document execution modes
- [ ] Document component architecture
- [ ] Usage examples for each mode
- [ ] Migration guide (from old to new)

#### 3.4 Examples
- [ ] Create example: Direct mode (simple chat)
- [ ] Create example: Standard mode (tool usage)
- [ ] Create example: Autonomous mode (complex task)
- [ ] Update existing examples to use new modes

---

## 🎯 Phase 2: Agent Types (Reasoning Patterns) - 3-4 weeks

### Week 1: Core Types

#### 2.1 AgentType Enum
- [ ] Create `src/nucleusiq/agents/config/agent_type.py`
- [ ] Define `AgentType` enum with all 11 types:
  - [ ] SIMPLE
  - [ ] REACT
  - [ ] REWOO
  - [ ] REFLECTION
  - [ ] COT
  - [ ] UTILITY
  - [ ] GOAL
  - [ ] MODEL
  - [ ] REFLEX
  - [ ] SIMPLE_REFLEX
  - [ ] REACTIVE
  - [ ] PROACTIVE
- [ ] Add `agent_type` field to `AgentConfig`
  - [ ] Default: `AgentType.SIMPLE`
  - [ ] Description and validation
- [ ] Tests for AgentType enum

#### 2.2 Strategy Dispatcher
- [ ] Create strategy dispatcher in `Agent` class
- [ ] Implement `_execute_with_strategy(task)` method
- [ ] Create strategy mapping dictionary
- [ ] Route to appropriate strategy based on `agent_type`
- [ ] Tests for strategy routing

#### 2.3 Simple Strategy
- [ ] Implement `Agent._execute_simple(task)` method
  - [ ] Route to appropriate mode (direct/standard/autonomous)
  - [ ] Basic execution without special reasoning
- [ ] Tests for Simple strategy
  - [ ] Works in all modes
  - [ ] Backward compatible

#### 2.4 ReAct Strategy
- [ ] Refactor existing `ReActAgent` class
- [ ] Implement `Agent._execute_react(task)` method
  - [ ] Thought-Action-Observation loop
  - [ ] Works in autonomous mode (primary)
  - [ ] Fallback for standard mode
- [ ] Extract ReAct logic from `ReActAgent` class
- [ ] Tests for ReAct strategy
  - [ ] Full loop in autonomous mode
  - [ ] Simplified in standard mode
  - [ ] Multiple iterations
  - [ ] Final answer detection

---

### Week 2: Reasoning Patterns

#### 2.5 ReWoo Strategy
- [ ] Create `src/nucleusiq/agents/strategies/rewoo_strategy.py`
- [ ] Implement `Agent._execute_rewoo(task)` method
  - [ ] Create complete plan upfront
  - [ ] Execute all steps without observation loop
  - [ ] Return final result
- [ ] Validate: Requires autonomous mode
- [ ] Tests for ReWoo strategy
  - [ ] Plan creation
  - [ ] Sequential execution
  - [ ] No observation loop
  - [ ] Error handling

#### 2.6 Reflection Strategy
- [ ] Create `src/nucleusiq/agents/strategies/reflection_strategy.py`
- [ ] Implement `Agent._execute_reflection(task)` method
  - [ ] Initial execution
  - [ ] Critique result using Critic
  - [ ] Refine task based on critique
  - [ ] Re-execute if needed
  - [ ] Multiple reflection cycles
- [ ] Validate: Requires autonomous mode
- [ ] Tests for Reflection strategy
  - [ ] Initial execution
  - [ ] Critique and refinement
  - [ ] Re-execution
  - [ ] Multiple cycles

#### 2.7 CoT Strategy
- [ ] Create `src/nucleusiq/agents/strategies/cot_strategy.py`
- [ ] Implement `Agent._execute_cot(task)` method
  - [ ] Use ChainOfThoughtPrompt
  - [ ] Works in all modes (direct/standard/autonomous)
  - [ ] Explicit reasoning steps
- [ ] Integration with PromptFactory
- [ ] Tests for CoT strategy
  - [ ] Works in all modes
  - [ ] Reasoning steps visible
  - [ ] Prompt integration

---

### Week 3: Decision-Making Types

#### 2.8 Goal-Based Strategy
- [ ] Create `src/nucleusiq/agents/strategies/goal_strategy.py`
- [ ] Create `Goal` class (if not exists)
  - [ ] `from_task(task)` class method
  - [ ] Progress tracking
  - [ ] Success criteria
- [ ] Implement `Agent._execute_goal(task)` method
  - [ ] Create goal from task
  - [ ] Create plan
  - [ ] Track progress (0% → 100%)
  - [ ] Adapt plan if needed
  - [ ] Return when goal achieved
- [ ] Implement `_update_progress()` helper
- [ ] Implement `_generate_final_result()` helper
- [ ] Validate: Requires autonomous mode
- [ ] Tests for Goal-based strategy
  - [ ] Goal creation
  - [ ] Progress tracking
  - [ ] Plan adaptation
  - [ ] Goal achievement

#### 2.9 Utility-Based Strategy
- [ ] Create `src/nucleusiq/agents/strategies/utility_strategy.py`
- [ ] Implement `Agent._execute_utility(task)` method
  - [ ] Define utility function
  - [ ] Generate candidate actions
  - [ ] Evaluate each candidate
  - [ ] Execute best candidate
- [ ] Implement `_get_utility_function()` helper
- [ ] Implement `_generate_candidates()` helper
- [ ] Validate: Requires autonomous mode
- [ ] Tests for Utility-based strategy
  - [ ] Utility function
  - [ ] Candidate generation
  - [ ] Best candidate selection
  - [ ] Execution

#### 2.10 Model-Based Strategy
- [ ] Create `src/nucleusiq/agents/strategies/model_strategy.py`
- [ ] Implement `Agent._execute_model(task)` method
  - [ ] Build/update internal model
  - [ ] Simulate actions in model
  - [ ] Create plan with model
  - [ ] Execute plan
- [ ] Implement `_build_model()` helper
- [ ] Implement model representation
- [ ] Validate: Requires autonomous mode
- [ ] Tests for Model-based strategy
  - [ ] Model building
  - [ ] Simulation
  - [ ] Plan creation
  - [ ] Execution

---

### Week 4: Behavioral Types

#### 2.11 Reflex Strategy
- [ ] Create `src/nucleusiq/agents/strategies/reflex_strategy.py`
- [ ] Implement `Agent._execute_reflex(task)` method
  - [ ] Define reflex rules
  - [ ] Match condition
  - [ ] Execute action
  - [ ] Fallback to standard
- [ ] Implement `_get_reflex_rules()` helper
- [ ] Create reflex rule system
- [ ] Validate: Works in direct/standard modes
- [ ] Tests for Reflex strategy
  - [ ] Rule matching
  - [ ] Action execution
  - [ ] Fallback behavior

#### 2.12 Reactive Strategy
- [ ] Create `src/nucleusiq/agents/strategies/reactive_strategy.py`
- [ ] Implement `Agent._execute_reactive(task)` method
  - [ ] Monitor environment
  - [ ] Detect changes
  - [ ] Adapt task to changes
  - [ ] Execute adapted task
- [ ] Implement `_get_environment_state()` helper
- [ ] Implement `_adapt_task()` helper
- [ ] Validate: Works in standard/autonomous modes
- [ ] Tests for Reactive strategy
  - [ ] Environment monitoring
  - [ ] Change detection
  - [ ] Task adaptation
  - [ ] Execution

#### 2.13 Proactive Strategy
- [ ] Create `src/nucleusiq/agents/strategies/proactive_strategy.py`
- [ ] Implement `Agent._execute_proactive(task)` method
  - [ ] Anticipate future needs
  - [ ] Create proactive plan
  - [ ] Execute proactively
- [ ] Implement `_anticipate_tasks()` helper
- [ ] Implement proactive planning in Planner
- [ ] Validate: Requires autonomous mode
- [ ] Tests for Proactive strategy
  - [ ] Task anticipation
  - [ ] Proactive planning
  - [ ] Execution

#### 2.14 Mode Compatibility Validation
- [ ] Implement `Agent._validate_mode_type_compatibility()` method
- [ ] Define incompatible combinations:
  - [ ] Direct mode incompatible types
  - [ ] Standard mode incompatible types
- [ ] Raise clear error messages
- [ ] Tests for compatibility validation
  - [ ] Valid combinations pass
  - [ ] Invalid combinations raise errors
  - [ ] Clear error messages

#### 2.15 Strategy Documentation
- [ ] Document each agent type
- [ ] When to use which type
- [ ] Mode compatibility matrix
- [ ] Examples for each type
- [ ] Performance characteristics

---

## 🎯 Phase 3: Guardrails System - 1-2 weeks

### Week 1: Base Infrastructure

#### 3.1 BaseGuardrail Interface
- [ ] Create `src/nucleusiq/agents/guardrails/__init__.py`
- [ ] Create `src/nucleusiq/agents/guardrails/base_guardrail.py`
- [ ] Implement `BaseGuardrail` abstract class:
  - [ ] `before_execute(task, agent)` - Pre-execution hook
  - [ ] `after_execute(task, result, agent)` - Post-execution hook
  - [ ] Abstract methods with proper typing
- [ ] Tests for BaseGuardrail interface

#### 3.2 Agent Integration
- [ ] Add `guardrails` field to `Agent` class
  - [ ] Type: `List[BaseGuardrail]`
  - [ ] Default: `[]`
- [ ] Integrate guardrails into `Agent.execute()`:
  - [ ] Pre-execution: Call `before_execute()` for each guardrail
  - [ ] Cancel execution if guardrail returns None
  - [ ] Post-execution: Call `after_execute()` for each guardrail
  - [ ] Chain results through guardrails
- [ ] Tests for guardrail integration
  - [ ] Pre-execution hooks
  - [ ] Post-execution hooks
  - [ ] Cancellation
  - [ ] Result chaining

---

### Week 2: Built-in Guardrails

#### 3.3 ValidationGuardrail
- [ ] Create `src/nucleusiq/agents/guardrails/validation_guardrail.py`
- [ ] Implement `ValidationGuardrail` class:
  - [ ] Validate task format
  - [ ] Validate task content
  - [ ] Check required fields
  - [ ] Return modified task or None
- [ ] Tests for ValidationGuardrail
  - [ ] Valid tasks pass
  - [ ] Invalid tasks rejected
  - [ ] Task modification

#### 3.4 SafetyGuardrail
- [ ] Create `src/nucleusiq/agents/guardrails/safety_guardrail.py`
- [ ] Implement `SafetyGuardrail` class:
  - [ ] Check for unsafe content
  - [ ] Check for unsafe actions
  - [ ] Content filtering
  - [ ] Action blocking
- [ ] Tests for SafetyGuardrail
  - [ ] Safe content passes
  - [ ] Unsafe content blocked
  - [ ] Action validation

#### 3.5 RateLimitGuardrail
- [ ] Create `src/nucleusiq/agents/guardrails/rate_limit_guardrail.py`
- [ ] Implement `RateLimitGuardrail` class:
  - [ ] Track request rate
  - [ ] Enforce rate limits
  - [ ] Queue management
  - [ ] Rate limit exceeded handling
- [ ] Tests for RateLimitGuardrail
  - [ ] Rate tracking
  - [ ] Limit enforcement
  - [ ] Queue behavior

#### 3.6 CostGuardrail
- [ ] Create `src/nucleusiq/agents/guardrails/cost_guardrail.py`
- [ ] Implement `CostGuardrail` class:
  - [ ] Track API costs
  - [ ] Monitor token usage
  - [ ] Enforce cost limits
  - [ ] Cost exceeded handling
- [ ] Tests for CostGuardrail
  - [ ] Cost tracking
  - [ ] Limit enforcement
  - [ ] Token counting

#### 3.7 OutputGuardrail
- [ ] Create `src/nucleusiq/agents/guardrails/output_guardrail.py`
- [ ] Implement `OutputGuardrail` class:
  - [ ] Validate output format
  - [ ] Sanitize output content
  - [ ] Check output quality
  - [ ] Return modified output
- [ ] Tests for OutputGuardrail
  - [ ] Output validation
  - [ ] Content sanitization
  - [ ] Quality checking

#### 3.8 Guardrail Documentation
- [ ] Document guardrail system
- [ ] Usage guide
- [ ] Examples for each guardrail
- [ ] Custom guardrail creation guide
- [ ] Best practices

---

## 🎯 Phase 4: Integration & Testing - 1 week

#### 4.1 End-to-End Tests
- [ ] Test all mode + type combinations
- [ ] Test guardrail integration
- [ ] Test component interactions
- [ ] Test error scenarios
- [ ] Test performance

#### 4.2 Performance Testing
- [ ] Benchmark Direct mode (should be fast)
- [ ] Benchmark Standard mode
- [ ] Benchmark Autonomous mode
- [ ] Memory usage profiling
- [ ] Component initialization overhead

#### 4.3 Documentation
- [ ] Update main README
- [ ] Architecture documentation
- [ ] API documentation
- [ ] Migration guide
- [ ] Troubleshooting guide

#### 4.4 Examples
- [ ] Examples for all mode + type combinations
- [ ] Examples with guardrails
- [ ] Real-world use cases
- [ ] Performance examples

#### 4.5 Migration Guide
- [ ] Guide for migrating from old to new API
- [ ] Backward compatibility notes
- [ ] Breaking changes (if any)
- [ ] Code migration examples

---

## 🎯 Phase 5: Fix Current Issues (High Priority)

### Planning System Fixes
- [ ] Fix LLM planning not being triggered
  - [ ] Check `use_llm_for_planning` flag
  - [ ] Ensure `_create_llm_plan()` is called
  - [ ] Fix plan creation logic
- [ ] Fix plan context not being passed
  - [ ] Pass plan to `_build_messages()`
  - [ ] Include plan in message construction
  - [ ] Test plan context usage
- [ ] Fix type mismatches in `_format_plan()`
  - [ ] Handle Plan object correctly
  - [ ] Handle list of dicts (backward compat)
  - [ ] Test both formats
- [ ] Fix context passing between plan steps
  - [ ] Store step results in context
  - [ ] Pass context to next step
  - [ ] Test context accumulation

### Memory System Fixes
- [ ] Make memory actively used in decisions
  - [ ] Retrieve relevant memories before execution
  - [ ] Include memories in message construction
  - [ ] Store new memories after execution
- [ ] Fix memory retrieval
  - [ ] Implement proper retrieval logic
  - [ ] Test retrieval accuracy
  - [ ] Test context integration

### State Management
- [ ] Add full state tracking
  - [ ] Track action history
  - [ ] Track observation history
  - [ ] Track beliefs and intentions
- [ ] State persistence
  - [ ] Save state between executions
  - [ ] Load state on initialization
  - [ ] State versioning

---

## 🎯 Phase 6: Testing & Quality Assurance

### Test Coverage
- [ ] Unit tests for all components
- [ ] Integration tests for all modes
- [ ] Integration tests for all agent types
- [ ] Guardrail tests
- [ ] Error handling tests
- [ ] Performance tests
- [ ] End-to-end tests

### Test Organization
- [ ] Organize tests by component
- [ ] Organize tests by feature
- [ ] Test fixtures and helpers
- [ ] Test data management
- [ ] CI/CD integration

### Code Quality
- [ ] Linting (flake8, pylint, mypy)
- [ ] Type hints for all code
- [ ] Docstrings for all public APIs
- [ ] Code review checklist
- [ ] Performance profiling

---

## 🎯 Phase 7: Documentation

### User Documentation
- [ ] Getting started guide
- [ ] Architecture overview
- [ ] API reference
- [ ] Examples gallery
- [ ] Best practices guide
- [ ] Troubleshooting guide

### Developer Documentation
- [ ] Contributing guide (already exists, update if needed)
- [ ] Architecture deep dive
- [ ] Component design docs
- [ ] Extension guide
- [ ] Testing guide

### Reference Documentation
- [ ] Agent types reference
- [ ] Execution modes reference
- [ ] Guardrails reference
- [ ] Configuration reference
- [ ] Migration guide

---

## 🎯 Phase 8: Examples & Tutorials

### Basic Examples
- [ ] Simple chat (Direct mode)
- [ ] Tool usage (Standard mode)
- [ ] Complex task (Autonomous mode)
- [ ] Each agent type example
- [ ] Guardrail examples

### Advanced Examples
- [ ] Multi-agent scenarios
- [ ] Custom guardrails
- [ ] Custom agent types
- [ ] Performance optimization
- [ ] Real-world use cases

### Tutorials
- [ ] Building your first agent
- [ ] Choosing the right mode
- [ ] Choosing the right agent type
- [ ] Adding guardrails
- [ ] Extending the framework

---

## 📊 Summary Statistics

### Total Tasks by Phase
- **Phase 1 (Gearbox Strategy)**: ~45 tasks
- **Phase 2 (Agent Types)**: ~60 tasks
- **Phase 3 (Guardrails)**: ~25 tasks
- **Phase 4 (Integration)**: ~15 tasks
- **Phase 5 (Fixes)**: ~10 tasks
- **Phase 6 (Testing)**: ~15 tasks
- **Phase 7 (Documentation)**: ~15 tasks
- **Phase 8 (Examples)**: ~15 tasks

**Total: ~200 tasks**

### Estimated Timeline
- **Phase 1**: 2-3 weeks
- **Phase 2**: 3-4 weeks
- **Phase 3**: 1-2 weeks
- **Phase 4**: 1 week
- **Phase 5**: 1 week (can be done in parallel)
- **Phase 6**: Ongoing
- **Phase 7**: Ongoing
- **Phase 8**: 1 week

**Total: 9-12 weeks (2.5-3 months)**

---

## 🚀 Quick Start (Immediate Next Steps)

### Priority 1: Fix Current Issues
1. Fix planning system (LLM planning, context passing)
2. Make memory actively used
3. Fix state management

### Priority 2: Gearbox Strategy
1. Add execution_mode to AgentConfig
2. Create Executor component
3. Implement Direct mode
4. Refactor Standard mode

### Priority 3: Agent Types
1. Create AgentType enum
2. Implement strategy dispatcher
3. Implement Simple strategy
4. Refactor ReAct strategy

---

## 📝 Notes

- Tasks marked with `[ ]` are pending
- Tasks marked with `[x]` are completed
- Some tasks can be done in parallel
- Testing should be done continuously, not just at the end
- Documentation should be updated as features are implemented

---

*This TODO list is a living document and will be updated as work progresses.*


