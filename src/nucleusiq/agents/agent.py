# src/nucleusiq/agents/agent.py
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import inspect
from pydantic import PrivateAttr

from nucleusiq.agents.builder.base_agent import BaseAgent
from nucleusiq.agents.config.agent_config import AgentState, AgentMetrics
from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan, PlanStep, PlanResponse, PlanStepResponse
from nucleusiq.agents.components.executor import Executor
# from nucleusiq.core.memory import BaseMemory
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.core.llms.base_llm import BaseLLM
# from nucleusiq.core.tools import BaseTool

class Agent(BaseAgent):
    """
    Concrete implementation of an agent in the NucleusIQ framework.
    
    This class provides the actual implementation of agent behaviors,
    building upon the foundation established in BaseAgent.
    
    Execution Modes (Gearbox Strategy):
    - "direct": Fast, simple, no tools (Gear 1)
    - "standard": Tool-enabled, linear execution (Gear 2) - default
    - "autonomous": Full reasoning loop with planning and self-correction (Gear 3)
    
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
            llm=llm,
            config=AgentConfig(execution_mode="standard")
        )
        
        # Without prompt (role/objective used)
        agent = Agent(
            name="CalculatorBot",
            role="Calculator",              # Used to build system message
            objective="Perform calculations", # Used to build system message
            prompt=None,
            llm=llm,
            config=AgentConfig(execution_mode="direct")
        )
    """
    
    # Private attributes (initialized in initialize())
    _executor: Optional[Executor] = PrivateAttr(default=None)

    async def initialize(self) -> None:
        """Initialize agent components and resources."""
        self._logger.info(f"Initializing agent: {self.name}")
        
        try:
            # Initialize Executor component (always needed for tool execution)
            if self.llm:
                self._executor = Executor(self.llm, self.tools)
                self._logger.debug("Executor component initialized")
            else:
                self._executor = None
                self._logger.debug("Executor not initialized (no LLM)")
            
            # Initialize memory if provided and enabled
            if self.memory and self.config.enable_memory:
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
        """
        Create an execution plan using the LLM with structured output.
        
        Uses function calling (if supported) or structured JSON prompt to ensure
        deterministic, parseable plan structure.
        """
        if not self.llm:
            raise ValueError("LLM is required for LLM-based planning")
        
        # Construct planning prompt with structured output instructions
        plan_prompt = self._construct_planning_prompt(task, context)
        
        # Generate plan using LLM
        try:
            # Convert string prompt to message format for LLM.call()
            messages = [{"role": "user", "content": plan_prompt}]
            
            # Try function calling first (if LLM supports it) for structured output
            plan_function_spec = self._get_plan_function_spec()
            llm_response = None
            
            # Check if LLM supports function calling by checking if it has convert_tool_specs
            # or if we can pass tools parameter
            try:
                # Try with function calling for structured output
                llm_response = await self.llm.call(
                    model=getattr(self.llm, "model_name", "default"),
                    messages=messages,
                    tools=[plan_function_spec],  # Pass plan function as a tool
                )
                
                # Check if LLM returned a function call
                response_msg = llm_response.choices[0].message
                if isinstance(response_msg, dict):
                    function_call = response_msg.get("function_call")
                else:
                    function_call = getattr(response_msg, "function_call", None)
                
                if function_call:
                    # Extract function arguments (structured JSON)
                    if isinstance(function_call, dict):
                        fn_name = function_call.get("name")
                        fn_args_str = function_call.get("arguments", "{}")
                    else:
                        fn_name = getattr(function_call, "name", None)
                        fn_args_str = getattr(function_call, "arguments", "{}")
                    
                    if fn_name == "create_plan":
                        import json
                        try:
                            plan_data = json.loads(fn_args_str)
                            # Use Pydantic model for validation
                            plan_response_model = PlanResponse.from_dict(plan_data)
                            self._logger.debug("Successfully received structured plan via function calling")
                            # Convert PlanResponse to Plan
                            return plan_response_model.to_plan(task)
                        except (json.JSONDecodeError, ValueError) as e:
                            self._logger.warning(f"Failed to parse function call arguments: {e}. Falling back to content parsing.")
            except Exception as e:
                self._logger.debug(f"Function calling not available or failed: {e}. Using content-based parsing.")
            
            # Fallback: Parse from content (structured JSON or text)
            if not llm_response:
                llm_response = await self.llm.call(
                    model=getattr(self.llm, "model_name", "default"),
                    messages=messages,
                )
            
            # Extract content from response (handle both dict and object)
            if not llm_response or not hasattr(llm_response, "choices") or not llm_response.choices:
                raise ValueError("LLM returned empty response for planning")
            
            response_msg = llm_response.choices[0].message
            if isinstance(response_msg, dict):
                response_content = response_msg.get("content")
            else:
                response_content = getattr(response_msg, "content", None)
            
            if not response_content:
                raise ValueError("LLM returned no content for planning")
            
            # Parse structured response (returns PlanResponse model)
            plan_response_model = self._parse_plan_response(response_content)
            # Convert PlanResponse to Plan
            return plan_response_model.to_plan(task)
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

    def _get_plan_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for structured plan output from Pydantic model.
        
        This ensures deterministic, parseable plan structure.
        The schema is automatically generated from PlanResponse model.
        """
        # Generate JSON schema from Pydantic model
        schema = PlanResponse.model_json_schema()
        
        # Extract the main schema (remove $defs if present, they'll be inlined)
        # For function calling, we want a clean schema
        main_schema = {
            "type": schema.get("type", "object"),
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
        
        # Inline any $defs references for cleaner schema
        defs = schema.get("$defs", {}) or schema.get("definitions", {})
        if defs and "properties" in main_schema:
            # Inline PlanStepResponse definition into items
            if "steps" in main_schema["properties"]:
                steps_prop = main_schema["properties"]["steps"]
                if "items" in steps_prop and "$ref" in steps_prop["items"]:
                    ref_name = steps_prop["items"]["$ref"].split("/")[-1]
                    if ref_name in defs:
                        steps_prop["items"] = defs[ref_name]
                        # Ensure required fields are present
                        if "required" not in steps_prop["items"]:
                            steps_prop["items"]["required"] = ["step", "action"]
        
        return main_schema
    
    def _get_plan_function_spec(self) -> Dict[str, Any]:
        """
        Get function specification for structured plan generation via function calling.
        
        This enforces structured output through LLM function calling mechanism.
        """
        return {
            "type": "function",
            "function": {
                "name": "create_plan",
                "description": "Create a structured execution plan with step-by-step actions",
                "parameters": self._get_plan_schema()
            }
        }
    
    def _construct_planning_prompt(self, task: Union[Task, Dict[str, Any]], context: Dict[str, Any]) -> str:
        """
        Construct a prompt for plan generation with structured output instructions.
        
        The prompt explicitly requests JSON format for deterministic parsing.
        """
        # Convert task to dict for string representation
        task_dict = task.to_dict() if isinstance(task, Task) else task
        task_obj = task_dict.get("objective", str(task))
        
        # Get available tools
        tool_names = [t.name for t in self.tools] if self.tools else []
        tools_str = ", ".join(tool_names) if tool_names else "None"
        
        # Build structured prompt
        if self.prompt:
            base_prompt = self.prompt.format_prompt(
                task=task,
                context=context,
                tools=self.tools,
                agent_role=self.role
            )
        else:
            base_prompt = f"""As {self.role} with objective '{self.objective}',
create a plan to accomplish the following task:
{task_obj}"""
        
        # Add structured output instructions
        structured_instructions = f"""

IMPORTANT: You must respond with a valid JSON object following this exact structure:
{{
    "steps": [
        {{
            "step": 1,
            "action": "execute",
            "args": {{}},
            "details": "Description of this step"
        }},
        {{
            "step": 2,
            "action": "tool_name",
            "args": {{"param1": "value1"}},
            "details": "Description of this step"
        }}
    ]
}}

Requirements:
- "steps" must be an array of step objects
- Each step must have "step" (integer, 1-indexed) and "action" (string)
- "args" (object) and "details" (string) are optional
- Action can be "execute" for direct execution or a tool name from: [{tools_str}]
- Return ONLY valid JSON, no additional text before or after

Available tools: {tools_str}

Create a step-by-step plan to accomplish this task. Return the plan as a JSON object."""
        
        return base_prompt + structured_instructions

    def _parse_plan_response(self, response: str) -> PlanResponse:
        """
        Parse the LLM's planning response into structured PlanResponse model.
        
        Supports multiple formats:
        1. Function call response (preferred - structured)
        2. JSON object in response text
        3. Fallback to text parsing (backward compatibility)
        
        Returns PlanResponse model instance.
        """
        import json
        import re
        
        # Method 1: Try to extract JSON from response (handles markdown code blocks)
        json_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*"steps".*\}', response, re.DOTALL)
            if json_match and json_match.group(0):
                json_str = json_match.group(0)
            else:
                json_str = response.strip()
        
        # Try to parse as JSON and validate with Pydantic model
        try:
            plan_data = json.loads(json_str)
            if isinstance(plan_data, dict) and "steps" in plan_data:
                # Use Pydantic model for validation and parsing
                plan_response = PlanResponse.from_dict(plan_data)
                self._logger.debug(f"Successfully parsed {len(plan_response.steps)} steps from JSON using PlanResponse model")
                return plan_response
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self._logger.warning(f"Failed to parse JSON from response: {e}. Trying fallback parsing.")
        
        # Method 2: Fallback to text parsing (backward compatibility)
        self._logger.warning("Using fallback text parsing. Consider using structured JSON output for better reliability.")
        step_responses = []
        current_step = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Look for step markers
            step_match = re.match(r'Step\s+(\d+)[:.]?\s*(.*)', line, re.IGNORECASE)
            if step_match:
                if current_step:
                    step_responses.append(PlanStepResponse(**current_step))
                step_num = int(step_match.group(1))
                action = step_match.group(2).strip()
                current_step = {
                    'step': step_num,
                    'action': action or 'execute',
                    'args': {},
                    'details': ''
                }
            elif current_step:
                # Check if line contains action information
                if 'action' in line.lower() and ':' in line:
                    action_match = re.search(r'action[:\s]+(.+)', line, re.IGNORECASE)
                    if action_match and (not current_step.get('action') or current_step.get('action') == 'execute'):
                        current_step['action'] = action_match.group(1).strip()
                else:
                    # Add to details
                    if current_step['details']:
                        current_step['details'] += '\n' + line
                    else:
                        current_step['details'] = line
                    
        if current_step:
            step_responses.append(PlanStepResponse(**current_step))
        
        # If no steps found, create a default single-step plan
        if not step_responses:
            self._logger.warning("No steps parsed from response. Creating default single-step plan.")
            step_responses = [PlanStepResponse(
                step=1,
                action='execute',
                args={},
                details='Execute the task'
            )]
            
        return PlanResponse(steps=step_responses)
    
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
            plan: Plan instance or list of plan step dictionaries
            
        Returns:
            Formatted plan string
        """
        plan_lines = []
        # Convert Plan to list of dicts if needed
        if isinstance(plan, Plan):
            steps = plan.to_list()
        else:
            steps = plan
        
        for step in steps:
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
                    # Direct execution for this step (use current execution mode)
                    result = await self._run_standard(step_task)
                elif action in [t.name for t in self.tools if hasattr(t, 'name')]:
                    # Tool call - use Executor if available
                    if hasattr(self, '_executor') and self._executor:
                        fn_call = {
                            "name": action,
                            "arguments": json.dumps(step.args or {})
                        }
                        result = await self._executor.execute(fn_call)
                    else:
                        # Fallback to old method
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
            if self.prompt:
                process_result = getattr(self.prompt, 'process_result', None)
                if process_result and callable(process_result):
                    if inspect.iscoroutinefunction(process_result):
                        result = await process_result(result)
                    else:
                        result = process_result(result)
            
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
        
        Execution Flow (Gearbox Strategy):
        - Direct mode: Fast, simple, no tools
        - Standard mode: Tool-enabled, linear execution (default)
        - Autonomous mode: Full reasoning loop with planning and self-correction
        
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
        
        # Route to appropriate execution mode (Gearbox Strategy)
        from nucleusiq.agents.config.agent_config import ExecutionMode
        execution_mode = self.config.execution_mode
        
        # Get mode value (handle both enum and string for backward compatibility)
        mode_value = execution_mode.value if hasattr(execution_mode, 'value') else str(execution_mode)
        self._logger.info(f"Agent '{self.name}' executing in {mode_value.upper()} mode")
        
        if execution_mode == ExecutionMode.DIRECT:
            return await self._run_direct(task)
        elif execution_mode == ExecutionMode.STANDARD:
            return await self._run_standard(task)
        elif execution_mode == ExecutionMode.AUTONOMOUS:
            # For now, fallback to standard (autonomous mode will be implemented in Week 2)
            self._logger.warning("Autonomous mode not yet implemented, falling back to standard mode")
            return await self._run_standard(task)
        else:
            raise ValueError(f"Unknown execution mode: {execution_mode}")
    
    async def _run_direct(self, task: Union[Task, Dict[str, Any]]) -> Any:
        """
        Gear 1: Direct mode - Fast, simple, no tools.
        
        Logic: Input → LLM → Output
        
        Use Cases: Chatbots, creative writing, simple explanations
        
        Characteristics:
        - Near-zero overhead
        - No tool execution
        - No planning
        - Single LLM call
        
        Args:
            task: Task instance or dictionary
            
        Returns:
            LLM response content
        """
        self._logger.debug("Executing in DIRECT mode (fast, no tools)")
        self.state = AgentState.EXECUTING
        
        # Check if LLM is available
        if not self.llm:
            self._logger.warning("No LLM configured, falling back to echo mode")
            self.state = AgentState.COMPLETED
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"

        # Build messages (no tools, no plan)
        task_dict = task.to_dict() if isinstance(task, Task) else task
        messages = self._build_messages(task_dict, plan=None)

        # Single LLM call (no tools)
        try:
            response = await self.llm.call(
                model=getattr(self.llm, "model_name", "default"),
                messages=messages,
                tools=None,  # Direct mode: no tools
            )

            if not response or not hasattr(response, "choices") or not response.choices:
                raise ValueError("LLM returned empty response")

            # Extract content
            msg = response.choices[0].message
            if isinstance(msg, dict):
                content = msg.get("content")
            else:
                content = getattr(msg, "content", None)

            if content:
                self.state = AgentState.COMPLETED
                return content

            # Fallback
            self._logger.warning("LLM returned no content, echoing objective")
            self.state = AgentState.COMPLETED
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"
                
        except Exception as e:
            self._logger.error(f"Error during direct execution: {str(e)}")
            self.state = AgentState.ERROR
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"
    
    async def _run_standard(self, task: Union[Task, Dict[str, Any]]) -> Any:
        """
        Gear 2: Standard mode - Tool-enabled, linear execution.
        
        Logic: Input → Decision → Tool Execution → Result
        
        Use Cases: "Check the weather", "Query database", "Search information"
        
        Characteristics:
        - Tool execution enabled
        - Linear flow (no loops)
        - Fire-and-forget (tries once, returns error if fails)
        - Optional memory
        - Multiple tool calls supported
        
        Args:
            task: Task instance or dictionary
            
        Returns:
            Execution result
        """
        self._logger.debug("Executing in STANDARD mode (tool-enabled, linear)")
        self.state = AgentState.EXECUTING
        
        # Check if LLM is available
        if not self.llm:
            self._logger.warning("No LLM configured, falling back to echo mode")
            self.state = AgentState.COMPLETED
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"

        # Check if Executor is initialized
        if not hasattr(self, '_executor') or self._executor is None:
            if self.llm:
                self._executor = Executor(self.llm, self.tools)
            else:
                raise RuntimeError("Cannot execute in standard mode: LLM not available")

        # (1) Convert tools to LLM-specific format
        tool_specs = []
        if self.tools and self.llm:
            tool_specs = self.llm.convert_tool_specs(self.tools)

        # (2) Build messages
        task_dict = task.to_dict() if isinstance(task, Task) else task
        messages = self._build_messages(task_dict)

        # (3) LLM call loop (may request multiple tools)
        try:
            max_tool_calls = 10  # Prevent infinite loops
            tool_call_count = 0
            
            while tool_call_count < max_tool_calls:
                response = await self.llm.call(
                    model=getattr(self.llm, "model_name", "default"),
                    messages=messages,
                    tools=tool_specs if tool_specs else None,
                )

                if not response or not hasattr(response, "choices") or not response.choices:
                    raise ValueError("LLM returned empty response")

                msg = response.choices[0].message
                if isinstance(msg, dict):
                    fn_call = msg.get("function_call")
                    content = msg.get("content")
                else:
                    fn_call = getattr(msg, "function_call", None)
                    content = getattr(msg, "content", None)

                # (4) Handle function call (fire-and-forget: try once, return error if fails)
                if fn_call:
                    tool_call_count += 1
                    fn_name = fn_call.get("name") if isinstance(fn_call, dict) else getattr(fn_call, "name", None)
                    fn_args_str = fn_call.get("arguments") if isinstance(fn_call, dict) else getattr(fn_call, "arguments", "{}")
                    
                    self._logger.info("Tool requested: %s", fn_name)
                    
                    try:
                        # Use Executor to execute tool
                        tool_result = await self._executor.execute(fn_call)
                        
                        # Append to conversation
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "function_call": fn_call,
                        })
                        messages.append({
                            "role": "function",
                            "name": fn_name,
                            "content": json.dumps(tool_result),
                        })
                        
                        # Continue loop to get final answer
                        continue
                        
                    except Exception as e:
                        # Fire-and-forget: return error immediately
                        self._logger.error(f"Tool execution failed: {e}")
                        self.state = AgentState.ERROR
                        return f"Error: Tool '{fn_name}' execution failed: {str(e)}"

                # (5) No function call - return content
                if content:
                    self.state = AgentState.COMPLETED
                    
                    # Store in memory if enabled
                    if self.memory and self.config.enable_memory:
                        try:
                            await self.memory.store(task, content)
                        except Exception as e:
                            self._logger.warning(f"Failed to store in memory: {e}")
                    
                    return content

                # No content and no function call - echo
                self._logger.info("LLM did not request a tool and returned no content; echoing objective")
                self.state = AgentState.COMPLETED
                objective = task.objective if isinstance(task, Task) else task.get('objective', '')
                return f"Echo: {objective}"
            
            # Max tool calls reached
            self._logger.warning(f"Maximum tool calls ({max_tool_calls}) reached")
            self.state = AgentState.ERROR
            return f"Error: Maximum tool calls ({max_tool_calls}) reached"
                
        except Exception as e:
            self._logger.error(f"Error during standard execution: {str(e)}")
            self.state = AgentState.ERROR
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return f"Echo: {objective}"
    
    async def _execute_direct(self, task: Union[Task, Dict[str, Any]]) -> Any:
        """
        [DEPRECATED] Execute task directly without plan.
        
        This method is kept for backward compatibility.
        It now delegates to _run_standard() which uses the Executor component.
        
        Use execution_mode="standard" instead.
        """
        self._logger.warning("_execute_direct() is deprecated, use _run_standard() or set execution_mode='standard'")
        return await self._run_standard(task)