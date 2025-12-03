# src/nucleusiq/agents/agent.py
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

from nucleusiq.agents.builder.base_agent import BaseAgent
from nucleusiq.agents.config.agent_config import AgentState, AgentMetrics
from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan, PlanStep
# from nucleusiq.core.memory import BaseMemory
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.llms.base_llm import BaseLLM
# from nucleusiq.core.tools import BaseTool

class Agent(BaseAgent):
    """
    Concrete implementation of an agent in the NucleusIQ framework.
    
    This class provides the actual implementation of agent behaviors,
    building upon the foundation established in BaseAgent.
    
    Prompt Precedence:
    - If `prompt` is provided, it takes precedence over `role`/`objective`
      for LLM message construction during execution.
    - If `prompt` is None, `role` and `objective` are used to construct
      the system message: "You are a {role}. Your objective is to {objective}."
    - `role` and `objective` are always used for planning context, even when prompt exists.
    
    Example:
        # With prompt (prompt takes precedence)
        agent = Agent(
            name="CalculatorBot",
            role="Calculator",              # Used for planning context only
            objective="Perform calculations", # Used for planning context only
            prompt=PromptFactory.create_prompt().configure(
                system="You are a helpful calculator assistant.",
                user="Answer questions accurately."
            ),
            llm=llm
        )
        
        # Without prompt (role/objective used)
        agent = Agent(
            name="CalculatorBot",
            role="Calculator",              # Used to build system message
            objective="Perform calculations", # Used to build system message
            prompt=None,
            llm=llm
        )
    """

    async def initialize(self) -> None:
        """Initialize agent components and resources."""
        self._logger.info(f"Initializing agent: {self.name}")
        
        try:
            # Initialize memory if provided
            if self.memory:
                await self.memory.initialize()
                self._logger.debug("Memory system initialized")
            
            # Initialize prompt if provided
            if self.prompt:
                prompt_text = self.prompt.format_prompt()
                self._logger.debug(f"Prompt system initialized \n {prompt_text}")
            
            # Initialize tools
            for tool in self.tools:
                await tool.initialize()
            if self.tools:
                self._logger.debug("Initialised %d tools", len(self.tools))
            
            # Initialization succeeded – agent is now ready but has not run any tasks yet.
            # Keep state as INITIALIZING so execution-related states (PLANNING/EXECUTING/
            # COMPLETED) are only used during/after task processing.
            self.state = AgentState.INITIALIZING
            self._logger.info("Agent initialization completed successfully")
            
        except Exception as e:
            self.state = AgentState.ERROR
            self._logger.error(f"Agent initialization failed: {str(e)}")
            raise

    # --------------------------------------------------------------------- #
    # PLANNING (very simple by default)                                     #
    # --------------------------------------------------------------------- #
    async def plan(self, task: Union[Task, Dict[str, Any]]) -> Plan:
        """
        Create an execution plan for the given task.
        
        This method breaks down complex tasks into smaller, manageable steps.
        By default, returns a simple one-step plan that executes the task directly.
        
        Override this method or use LLM-based planning (_create_llm_plan) for
        more sophisticated multi-step planning.
        
        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys
            
        Returns:
            Plan instance with steps
        """
        # Convert dict to Task if needed (backward compatibility)
        if isinstance(task, dict):
            task = Task.from_dict(task)
        
        # Create default one-step plan
        step = PlanStep(
            step=1,
            action="execute",
            task=task
        )
        return Plan(steps=[step], task=task)
    
    async def _get_context(self, task: Union[Task, Dict[str, Any]]) -> Dict[str, Any]:
        """Retrieve relevant context for task execution."""
        # Convert to dict for context
        if isinstance(task, Task):
            task_dict = task.to_dict()
        else:
            task_dict = task
            
        context = {
            "task": task_dict,
            "agent_role": self.role,
            "agent_objective": self.objective,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add memory context if available
        if self.memory:
            memory_context = await self.memory.get_relevant_context(task_dict)
            context["memory"] = memory_context
            
        return context

    async def _create_llm_plan(self, task: Union[Task, Dict[str, Any]], context: Dict[str, Any]) -> Plan:
        """Create an execution plan using the LLM."""
        # Construct planning prompt
        plan_prompt = self._construct_planning_prompt(task, context)
        
        # Generate plan using LLM
        try:
            # Convert string prompt to message format for LLM.call()
            messages = [{"role": "user", "content": plan_prompt}]
            plan_response = await self.llm.call(
                model=getattr(self.llm, "model_name", "default"),
                messages=messages,
            )
            
            # Extract content from response (handle both dict and object)
            if not plan_response or not hasattr(plan_response, "choices") or not plan_response.choices:
                raise ValueError("LLM returned empty response for planning")
            
            response_msg = plan_response.choices[0].message
            if isinstance(response_msg, dict):
                response_content = response_msg.get("content")
            else:
                response_content = getattr(response_msg, "content", None)
            
            if not response_content:
                raise ValueError("LLM returned no content for planning")
            
            steps = self._parse_plan_response(response_content)
            # Convert to PlanStep objects
            plan_steps = [PlanStep(**step) if isinstance(step, dict) else step for step in steps]
            return Plan(steps=plan_steps, task=task)
        except Exception as e:
            self._logger.error(f"LLM planning failed: {str(e)}")
            return await self._create_basic_plan(task)

    async def _create_basic_plan(self, task: Union[Task, Dict[str, Any]]) -> Plan:
        """Create a basic execution plan without LLM."""
        # Convert dict to Task if needed
        if isinstance(task, dict):
            task = Task.from_dict(task)
        # Simple single-step plan
        step = PlanStep(step=1, action="execute", task=task)
        return Plan(steps=[step], task=task)

    def _construct_planning_prompt(self, task: Union[Task, Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Construct a prompt for plan generation."""
        if self.prompt:
            return self.prompt.format_prompt(
                task=task,
                context=context,
                tools=self.tools,
                agent_role=self.role
            )
        
        # Basic prompt construction
        return f"""
        As {self.role} with objective '{self.objective}',
        create a plan to accomplish the following task:
        {task}
        
        Available tools: {[t.name for t in self.tools]}
        
        Create a step-by-step plan to accomplish this task.
        """

    def _parse_plan_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM's planning response into structured steps.
        Returns list of dicts (will be converted to PlanStep in _create_llm_plan).
        """
        """Parse the LLM's planning response into structured steps."""
        # This is a simplified version - in practice, you'd want more sophisticated parsing
        steps = []
        current_step = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Step'):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'step': len(steps) + 1,
                    'action': '',
                    'details': ''
                }
            elif current_step:
                if not current_step['action']:
                    current_step['action'] = line
                else:
                    current_step['details'] += line + '\n'
                    
        if current_step:
            steps.append(current_step)
            
        return steps
    
    def _build_messages(self, task: Union[Task, Dict[str, Any]], plan: Union[Plan, List[Dict[str, Any]], None] = None) -> List[Dict[str, Any]]:
        """
        Build messages for LLM call.
        
        Prompt Precedence:
        - If `prompt` is provided, it takes precedence over `role`/`objective`
          for LLM message construction (prompt.system and prompt.user are used).
        - If `prompt` is None, `role` and `objective` are used to construct
          the system message: "You are a {role}. Your objective is to {objective}."
        
        Message structure:
        1. System message: From prompt.system (if prompt exists) or role/objective (if no prompt)
        2. User template: From prompt.user (if prompt exists and has user field)
        3. Plan context: Execution plan if plan exists and has multiple steps
        4. User request: Actual user request (from task["objective"])
        
        Args:
            task: Task instance or dictionary
            plan: Optional plan (Plan object or list of dicts)
            
        Returns:
            List of message dictionaries
        """
        messages: List[Dict[str, Any]] = []
        
        # Convert task to dict for message building (backward compatibility)
        task_dict = task.to_dict() if isinstance(task, Task) else task
        
        # Priority 1: Use prompt if provided (overrides role/objective)
        if self.prompt:
            # Check if prompt has system/user fields
            has_system = hasattr(self.prompt, 'system') and self.prompt.system
            has_user = hasattr(self.prompt, 'user') and self.prompt.user
            
            if has_system:
                messages.append({"role": "system", "content": self.prompt.system})
                # Warn user that role/objective are being overridden
                if self.role or self.objective:
                    self._logger.info(
                        f"Using prompt.system for execution (overriding role='{self.role}', objective='{self.objective}'). "
                        f"role/objective will still be used for planning context."
                    )
            if has_user:
                messages.append({"role": "user", "content": self.prompt.user})
        # Priority 2: Fallback to role/objective if no prompt
        else:
            if self.role:
                system_msg = f"You are a {self.role}."
                if self.objective:
                    system_msg += f" Your objective is to {self.objective}."
                messages.append({"role": "system", "content": system_msg})
                self._logger.debug(f"Using role/objective for system message: {system_msg}")
        
        # Plan context (if plan exists and has multiple steps)
        if plan:
            # Handle both Plan object and list of dicts
            if isinstance(plan, Plan):
                plan_steps = plan.steps
            else:
                plan_steps = plan
            
            if len(plan_steps) > 1:
                plan_text = self._format_plan(plan)
                messages.append({
                    "role": "user",
                    "content": f"Execution Plan:\n{plan_text}\n\nNow execute the task following this plan."
                })
        
        # User's actual request (from task)
        messages.append({"role": "user", "content": task_dict.get("objective", "")})
        
        return messages
    
    def _format_plan(self, plan: Union[Plan, List[Dict[str, Any]]]) -> str:
        """
        Format plan steps into a readable string.
        
        Args:
            plan: List of plan step dictionaries
            
        Returns:
            Formatted plan string
        """
        plan_lines = []
        for step in plan:
            step_num = step.get("step", 0)
            action = step.get("action", "")
            details = step.get("details", "")
            plan_lines.append(f"Step {step_num}: {action}")
            if details:
                plan_lines.append(f"  {details.strip()}")
        return "\n".join(plan_lines)
    
    async def _execute_plan(self, task: Union[Task, Dict[str, Any]], plan: Union[Plan, List[Dict[str, Any]]]) -> Any:
        """
        Execute task following a multi-step plan.
        
        This method executes each plan step sequentially, building context
        from previous steps and passing it to subsequent steps.
        
        Args:
            task: Original task (Task instance or dict)
            plan: Plan instance or list of plan step dictionaries
            
        Returns:
            Final result from last step or aggregated result
        """
        # Convert plan to Plan instance if needed
        if isinstance(plan, list):
            if isinstance(task, dict):
                task = Task.from_dict(task)
            plan = Plan.from_list(plan, task)
        
        self._logger.info(f"Executing plan with {len(plan)} steps")
        self.state = AgentState.EXECUTING
        
        context = {}
        results = []
        
        for step in plan.steps:
            step_num = step.step
            action = step.action
            step_task = step.task if step.task else task
            step_details = step.details or ""
            
            # Convert step_task to dict if it's a Task
            if isinstance(step_task, Task):
                step_task = step_task.to_dict()
            elif isinstance(step_task, dict):
                pass  # Already a dict
            else:
                # Fallback: use original task
                step_task = task.to_dict() if isinstance(task, Task) else task
            
            self._logger.info(f"Executing plan step {step_num}: {action}")
            if step_details:
                self._logger.debug(f"Step details: {step_details}")
            
            # Execute step
            try:
                if action == "execute":
                    # Direct execution for this step
                    result = await self._execute_direct(step_task)
                elif action in [t.name for t in self.tools if hasattr(t, 'name')]:
                    # Tool call
                    tool_args = step.args or {}
                    result = await self._execute_tool(action, tool_args)
                else:
                    # Unknown action - log and continue
                    self._logger.warning(f"Unknown action '{action}' in plan step {step_num}, skipping")
                    result = f"Skipped unknown action: {action}"
                
                results.append(result)
                context[f"step_{step_num}"] = result
                context[f"step_{step_num}_action"] = action
                
            except Exception as e:
                self._logger.error(f"Error executing plan step {step_num}: {str(e)}")
                results.append(f"Error in step {step_num}: {str(e)}")
                context[f"step_{step_num}_error"] = str(e)
        
        # Return final result (last step result)
        final_result = results[-1] if results else None
        self.state = AgentState.COMPLETED
        return final_result

    async def _process_result(self, result: Any) -> Any:
        """Process and store execution results."""
        try:
            # Store in memory if available
            if self.memory:
                await self.memory.store(
                    key=f"task_result_{datetime.now().isoformat()}",
                    value={
                        'task': self._current_task,
                        'result': result,
                        'metrics': self.metrics.dict()
                    }
                )
            
            # Process through prompt if available and method exists
            if self.prompt and hasattr(self.prompt, 'process_result'):
                result = await self.prompt.process_result(result)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Result processing failed: {str(e)}")
            raise

    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task format and requirements."""
        required_fields = ['id', 'objective']
        return all(field in task for field in required_fields)

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a specific tool with parameters."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
            
        self.state = AgentState.WAITING_FOR_TOOLS
        try:
            return await tool.execute(**params)
        finally:
            self.state = AgentState.EXECUTING

    async def _handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Handle execution errors with appropriate logging and recovery."""
        self._logger.error(f"Error during execution: {str(error)}")
        
        if self.memory:
            await self.memory.store(
                key=f"error_{datetime.now().isoformat()}",
                value={
                    'error': str(error),
                    'context': context,
                    'task': self._current_task
                }
            )
            
        self.metrics.error_count += 1
        self.state = AgentState.ERROR

    async def save_state(self) -> Dict[str, Any]:
        """Save agent's current state."""
        state = {
            'id': self.id,
            'name': self.name,
            'state': self.state,
            'metrics': self.metrics.dict(),
            'current_task': self._current_task,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.memory:
            state['memory'] = await self.memory.export_state()
            
        return state

    async def load_state(self, state: Dict[str, Any]) -> None:
        """Load agent's saved state."""
        self.state = state['state']
        self.metrics = AgentMetrics(**state['metrics'])
        self._current_task = state['current_task']
        
        if self.memory and 'memory' in state:
            await self.memory.import_state(state['memory'])
            
        self._logger.info(f"Loaded agent state from {state['timestamp']}")

    async def delegate_task(
        self,
        task: Dict[str, Any],
        target_agent: 'Agent'
    ) -> Any:
        """Delegate a task to another agent."""
        self._logger.info(f"Delegating task to agent to perfoming the task: {target_agent.name}")
        self.state = AgentState.WAITING_FOR_HUMAN
        
        try:
            return await target_agent.execute(task)
        finally:
            self.state = AgentState.EXECUTING

    # --------------------------------------------------------------------- #
    # EXECUTION WITH FUNCTION-CALLING LOOP                                  #
    # --------------------------------------------------------------------- #
    async def execute(self, task: Union[Task, Dict[str, Any]]) -> Any:
        """
        Execute a task using the agent's capabilities.
        
        Execution Flow:
        1. Optionally create a plan if use_planning is enabled
        2. If plan has multiple steps, execute plan steps sequentially
        3. Otherwise, execute task directly (current behavior)
        
        The execution uses:
        - Task: User's request (what to do) - from task.objective
        - Prompt: Agent's instructions (how to behave) - from self.prompt
        - Plan: Task decomposition (how to break down) - optional, from plan()
        
        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys
            
        Returns:
            Execution result (final answer or tool result)
        """
        # Convert dict to Task if needed (backward compatibility)
        if isinstance(task, dict):
            task = Task.from_dict(task)
        
        self._logger.debug("Starting execution for task %s", task.id)
        self._current_task = task.to_dict()  # Store as dict for compatibility
        
        # Optionally create plan if planning is enabled
        plan = None
        if self.config.use_planning:
            self.state = AgentState.PLANNING
            self._logger.debug("Planning enabled, creating execution plan...")
            plan = await self.plan(task)
            self._logger.debug(f"Plan created with {len(plan)} step(s)")
        
        # If plan has multiple steps, execute plan
        if plan and len(plan) > 1:
            self._logger.info(f"Executing multi-step plan with {len(plan)} steps")
            return await self._execute_plan(task, plan)
        
        # Otherwise, execute directly (backward compatible)
        return await self._execute_direct(task)
    
    async def _execute_direct(self, task: Union[Task, Dict[str, Any]]) -> Any:
        """
        Execute task directly without plan (current behavior).
        
        • Builds the conversation (system → template user → real user)
        • Sends tools spec on the first call
        • Executes any requested tool and feeds result back
        • Returns the model's final answer or echoes the objective.
        """
        self._logger.debug("Executing task directly (no plan)")
        self.state = AgentState.EXECUTING
        
        # Check if LLM is available
        if not self.llm:
            self._logger.warning("No LLM configured, falling back to echo mode")
            self.state = AgentState.COMPLETED
            # Handle both Task and dict
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"

        # (1) Convert tools to LLM-specific format
        # The LLM provider handles conversion from BaseTool specs to its own format
        tool_specs = []
        if self.tools and self.llm:
            tool_specs = self.llm.convert_tool_specs(self.tools)

        # (2) construct messages
        # Convert to dict for message building (backward compatibility)
        task_dict = task.to_dict() if isinstance(task, Task) else task
        messages = self._build_messages(task_dict)

        # (3) first LLM call
        try:
            resp1 = await self.llm.call(
                model=getattr(self.llm, "model_name", "default"),
                messages=messages,
                tools=tool_specs if tool_specs else None,
            )

            if not resp1 or not hasattr(resp1, "choices") or not resp1.choices:
                raise ValueError("LLM returned empty response")

            # BaseOpenAI returns a dict; MockLLM returns an object.
            first_msg = resp1.choices[0].message
            if isinstance(first_msg, dict):
                fn_call = first_msg.get("function_call")
            else:
                fn_call = getattr(first_msg, "function_call", None)

            # (4) handle function_call (only for BaseTool instances, not native tools)
            if fn_call:
                fn_name = fn_call.get("name") if isinstance(fn_call, dict) else getattr(fn_call, "name", None)
                fn_args_str = fn_call.get("arguments") if isinstance(fn_call, dict) else getattr(fn_call, "arguments", "{}")
                fn_args = json.loads(fn_args_str or "{}")

                self._logger.info("Tool requested: %s  args=%s", fn_name, fn_args)
                # Find tool instance (only BaseTool instances have execute() - native tools don't)
                tool = next((t for t in self.tools if hasattr(t, 'name') and t.name == fn_name), None)
                if tool is None:
                    raise ValueError(f"Tool '{fn_name}' not found")
                
                # Check if it's a native tool (shouldn't have function calls, but just in case)
                if hasattr(tool, 'is_native') and tool.is_native:
                    raise ValueError(f"Tool '{fn_name}' is a native tool and doesn't support execute()")

                tool_result = await tool.execute(**fn_args)

                # append assistant's function_call & the tool (function) message
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": fn_call,
                        },
                        {
                            "role": "function",
                            "name": fn_name,
                            "content": json.dumps(tool_result),
                        },
                    ]
                )

                # (5) final LLM call
                resp2 = await self.llm.call(
                    model=getattr(self.llm, 'model_name', 'default'), 
                    messages=messages
                )
                
                if not resp2 or not hasattr(resp2, 'choices') or not resp2.choices:
                    raise ValueError("LLM returned empty response on final call")
                    
                final_msg = resp2.choices[0].message
                # Handle both dict and object responses consistently
                if isinstance(final_msg, dict):
                    final = final_msg.get("content")
                else:
                    final = getattr(final_msg, "content", None)
                
                if not final:
                    final = str(final_msg)
                
                self._logger.debug("Final LLM response: %s", final)
                self.state = AgentState.COMPLETED
                return final

            # (6) no function call - return content or echo
            if isinstance(first_msg, dict):
                content = first_msg.get("content")
            else:
                content = getattr(first_msg, "content", None)

            if content:
                self.state = AgentState.COMPLETED
                return content

            self._logger.info(
                "LLM did not request a tool and returned no content; echoing objective."
            )
            self.state = AgentState.COMPLETED
            # Handle both Task and dict
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"
                
        except Exception as e:
            self._logger.error(f"Error during execution: {str(e)}")
            self.state = AgentState.ERROR
            # Fallback to echo on error
            # Handle both Task and dict
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"