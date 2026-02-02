# src/nucleusiq/agents/agent.py
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio
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

        # A shorter prompt used specifically for tool-calling (more reliable than a huge JSON-instructions prompt).
        task_dict = task.to_dict() if isinstance(task, Task) else task
        task_obj = task_dict.get("objective", str(task_dict))
        tool_names = [t.name for t in self.tools] if self.tools else []
        tools_str = ", ".join(tool_names) if tool_names else "None"
        # Include tool parameter hints (critical for getting usable args back in the plan).
        tool_param_lines: List[str] = []
        for t in self.tools or []:
            try:
                spec = t.get_spec() if hasattr(t, "get_spec") else None
                params = (spec or {}).get("parameters", {})
                props = (params or {}).get("properties", {}) if isinstance(params, dict) else {}
                required = (params or {}).get("required", []) if isinstance(params, dict) else []
                if isinstance(props, dict) and props:
                    tool_param_lines.append(
                        f"- {t.name}({', '.join(props.keys())}) required={required}"
                    )
                else:
                    tool_param_lines.append(f"- {t.name}(...)")
            except Exception:
                tool_param_lines.append(f"- {getattr(t, 'name', 'unknown')}(...)")

        tool_call_prompt = (
            "Create a step-by-step execution plan for the task below. "
            "You MUST call the create_plan tool with the plan.\n\n"
            f"Task: {task_obj}\n\n"
            f"Available tools:\n"
            + ("\n".join(tool_param_lines) if tool_param_lines else "- None") + "\n\n"
            "CRITICAL RULES:\n"
            "1. EVERY tool step MUST include 'args' with ALL required parameters filled in.\n"
            "2. For the first steps, extract CONCRETE VALUES from the task (e.g., numbers, strings).\n"
            "3. For later steps that need results from previous steps, use \"$step_N\" references.\n\n"
            "EXAMPLE for task 'Calculate (5 + 3) * 2':\n"
            "{\n"
            '  "steps": [\n'
            '    {"step": 1, "action": "add", "args": {"a": 5, "b": 3}, "details": "5 + 3 = 8"},\n'
            '    {"step": 2, "action": "multiply", "args": {"a": "$step_1", "b": 2}, "details": "8 * 2 = 16"}\n'
            "  ]\n"
            "}\n\n"
            "Now create the plan for the given task. Extract actual values from the task text.\n"
        )
        
        # Generate plan using LLM
        def _content_to_text(content: Any) -> Optional[str]:
            """
            Coerce OpenAI-style message content into plain text.

            Chat completions generally return `content: str | None`, but some SDKs / modes
            may represent content as a list of parts.
            """
            if content is None:
                return None
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Common shape: [{"type":"text","text":"..."}]
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                joined = "\n".join(p for p in parts if p.strip())
                return joined if joined.strip() else None
            # Last resort
            s = str(content)
            return s if s.strip() else None

        # We occasionally see transient empty assistant messages (no tool_calls, no content).
        # Retry once before falling back.
        empty_retries_remaining = 1

        while True:
            # Convert string prompt to message format for LLM.call()
            messages = [{"role": "user", "content": plan_prompt}]
            
            # Try tool calling first for structured output
            plan_function_spec = self._get_plan_function_spec()
            llm_response = None
            
            # Check if LLM supports function calling by checking if it has convert_tool_specs
            # or if we can pass tools parameter
            try:
                # Try with function calling for structured output
                llm_response = await self.llm.call(
                    model=getattr(self.llm, "model_name", "default"),
                    messages=[{"role": "user", "content": tool_call_prompt}],
                    tools=[plan_function_spec],  # Pass plan function as a tool
                    max_tokens=getattr(self.config, "planning_max_tokens", 4096),
                )
                
                # Check if LLM returned a tool call (modern format: tool_calls)
                response_msg = llm_response.choices[0].message
                if isinstance(response_msg, dict):
                    tool_calls = response_msg.get("tool_calls")
                else:
                    tool_calls = getattr(response_msg, "tool_calls", None)
                
                # Handle tool_calls format (modern OpenAI format)
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # Find the create_plan tool call
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            fn_info = tool_call.get("function", {})
                            fn_name = fn_info.get("name") if isinstance(fn_info, dict) else None
                            fn_args_str = fn_info.get("arguments", "{}") if isinstance(fn_info, dict) else "{}"
                        else:
                            fn_info = getattr(tool_call, "function", None)
                            fn_name = getattr(fn_info, "name", None) if fn_info else None
                            fn_args_str = getattr(fn_info, "arguments", "{}") if fn_info else "{}"
                        
                        if fn_name == "create_plan":
                            import json
                            try:
                                plan_data = json.loads(fn_args_str)
                                # Use Pydantic model for validation
                                plan_response_model = PlanResponse.from_dict(plan_data)
                                self._logger.debug("Successfully received structured plan via tool calling")
                                # Convert PlanResponse to Plan
                                return plan_response_model.to_plan(task)
                            except (json.JSONDecodeError, ValueError) as e:
                                self._logger.warning(f"Failed to parse tool call arguments: {e}. Falling back to content parsing.")
                                break
            except Exception as e:
                self._logger.debug(f"Tool calling not available or failed: {e}. Using content-based parsing.")
            
            # Fallback: Parse from content (structured JSON or text)
            if not llm_response:
                llm_response = await self.llm.call(
                    model=getattr(self.llm, "model_name", "default"),
                    messages=messages,
                    max_tokens=getattr(self.config, "planning_max_tokens", 4096),
                )
            
            # Extract content from response (handle both dict and object)
            if not llm_response or not hasattr(llm_response, "choices") or not llm_response.choices:
                raise ValueError("LLM returned empty response for planning")
            
            response_msg = llm_response.choices[0].message
            if isinstance(response_msg, dict):
                refusal = response_msg.get("refusal")
                response_content = _content_to_text(response_msg.get("content"))
            else:
                refusal = getattr(response_msg, "refusal", None)
                response_content = _content_to_text(getattr(response_msg, "content", None))

            # Refusals should fail planning so AUTONOMOUS can fall back.
            if refusal:
                raise ValueError(f"LLM refused to generate plan: {refusal}")
            
            if not response_content:
                if empty_retries_remaining > 0:
                    empty_retries_remaining -= 1
                    # Nudge the model to respond with either tool call or JSON content.
                    plan_prompt = plan_prompt + "\n\nIMPORTANT: Do not return an empty message. Return either a tool call to create_plan, or JSON content."
                    continue
                raise ValueError("LLM returned no content for planning")
            
            # Parse structured response (returns PlanResponse model)
            plan_response_model = self._parse_plan_response(response_content)
            # Convert PlanResponse to Plan
            return plan_response_model.to_plan(task)
        # unreachable

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
        # NOTE:
        # Some OpenAI models / endpoints behave poorly (including returning empty messages)
        # when given large/complex JSON schemas generated by Pydantic. We keep the schema
        # minimal and OpenAI-friendly to maximize reliability.
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer", "description": "Step number (1-indexed)"},
                            "action": {"type": "string", "description": "Action/tool name or 'execute'"},
                            "args": {"type": "object", "description": "Arguments for the action/tool"},
                            "details": {"type": "string", "description": "Human-readable description"},
                        },
                        # Always require args. For "execute" actions, args can be {}.
                        "required": ["step", "action", "args"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        }
    
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
        
        # Helper function to extract JSON with balanced brackets (handles nested structures)
        def extract_balanced_json(text: str, start_pos: int) -> str | None:
            """Extract JSON object using balanced bracket matching to handle nested structures."""
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text[start_pos:], start=start_pos):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            return text[start_pos:i+1]
            return None
        
        # Method 1: Try to extract JSON from markdown code blocks
        json_str = None
        code_block_match = re.search(
            r'```(?:json)?\s*(\{)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if code_block_match:
            json_str = extract_balanced_json(response, code_block_match.start(1))
        
        # Method 2: Try to find JSON object directly in text
        if not json_str:
            first_brace = response.find('{')
            if first_brace != -1:
                json_str = extract_balanced_json(response, first_brace)
        
        # Method 3: Fallback to entire response
        if not json_str:
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

        def _resolve_arg_value(val: Any) -> Any:
            if isinstance(val, str):
                s = val.strip()
                # Support "$step_1", "${step_1}", "{{step_1}}"
                for prefix, suffix in [("$", ""), ("${", "}"), ("{{", "}}")]:
                    if s.startswith(prefix) and s.endswith(suffix):
                        key = s[len(prefix): len(s) - (len(suffix) if suffix else 0)]
                        key = key.strip()
                        if key in context:
                            return context[key]
                # Also allow direct "step_1" keys
                if s in context:
                    return context[s]
            if isinstance(val, dict):
                return {k: _resolve_arg_value(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_resolve_arg_value(v) for v in val]
            return val

        def _resolve_args(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not args:
                return {}
            return {k: _resolve_arg_value(v) for k, v in args.items()}
        
        # Get timeout and retry settings from config
        step_timeout = getattr(self.config, 'step_timeout', 60)
        step_max_retries = getattr(self.config, 'step_max_retries', 2)
        
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
            
            # Execute step with timeout and retry
            step_result = None
            step_error = None
            
            for attempt in range(step_max_retries + 1):
                try:
                    step_result = await asyncio.wait_for(
                        self._execute_step(
                            step, step_num, action, step_task, step_details, 
                            context, _resolve_args, task
                        ),
                        timeout=step_timeout
                    )
                    break  # Success - exit retry loop
                except asyncio.TimeoutError:
                    step_error = f"Step {step_num} ({action}) timed out after {step_timeout}s"
                    if attempt < step_max_retries:
                        self._logger.warning(f"{step_error} (attempt {attempt + 1}/{step_max_retries + 1})")
                        await asyncio.sleep(1)  # Brief pause before retry
                    else:
                        self._logger.error(f"{step_error} - max retries exceeded")
                except Exception as e:
                    step_error = str(e)
                    self._logger.error(f"Error in step {step_num}: {step_error}")
                    break  # Don't retry non-timeout errors
            
            # Handle step result or failure
            if step_result is not None:
                results.append(step_result)
                context[f"step_{step_num}"] = step_result
                context[f"step_{step_num}_action"] = action
            elif step_error:
                # Step failed after retries
                self.state = AgentState.ERROR
                return f"Error: Step {step_num} ({action}) failed: {step_error}"
        
        # Return final result (last step result)
        final_result = results[-1] if results else None
        self.state = AgentState.COMPLETED
        return final_result

    async def _execute_step(
        self,
        step: PlanStep,
        step_num: int,
        action: str,
        step_task: Dict[str, Any],
        step_details: str,
        context: Dict[str, Any],
        _resolve_args,
        task: Union[Task, Dict[str, Any]]
    ) -> Any:
        """
        Execute a single plan step with the given parameters.
        
        This method is called by _execute_plan with timeout wrapping.
        
        Args:
            step: The PlanStep object
            step_num: Step number for logging
            action: The action/tool name to execute
            step_task: The task dictionary for this step
            step_details: Additional details about the step
            context: Context dictionary with results from previous steps
            _resolve_args: Function to resolve $step_N references in args
            task: The original task
            
        Returns:
            The result of the step execution
        """
        if action == "execute":
            # Direct execution for this step (use current execution mode)
            return await self._run_standard(step_task)
        elif action in [t.name for t in self.tools if hasattr(t, 'name')]:
            # Tool call - use Executor if available
            if hasattr(self, '_executor') and self._executor:
                resolved_args = _resolve_args(step.args)
                # If the plan didn't provide args (or provided empty args), ask the LLM to
                # produce a tool call for this specific tool given current context.
                if (not resolved_args) and self.llm:
                    tool_specs = self.llm.convert_tool_specs(self.tools) if self.tools else []
                    if tool_specs:
                        # Try to make the tool call deterministic by including explicit required arg names.
                        required_keys: List[str] = []
                        for spec in tool_specs:
                            try:
                                fn = spec.get("function", {}) if isinstance(spec, dict) else {}
                                if fn.get("name") == action:
                                    params = fn.get("parameters", {}) if isinstance(fn, dict) else {}
                                    required_keys = params.get("required", []) if isinstance(params, dict) else []
                                    break
                            except Exception:
                                continue

                        task_obj = (task.to_dict() if isinstance(task, Task) else task).get("objective", "")
                        step_prompt = (
                            "You must call the tool below with valid JSON arguments.\n\n"
                            f"Tool: {action}\n"
                            f"Required args: {required_keys}\n\n"
                            f"Overall task: {task_obj}\n"
                            f"Current context (use these concrete values): {json.dumps(context)}\n"
                            f"Step {step_num} details: {step_details}\n\n"
                            "Call the tool now."
                        )
                        step_resp = await self.llm.call(
                            model=getattr(self.llm, "model_name", "default"),
                            messages=[{"role": "user", "content": step_prompt}],
                            tools=tool_specs,
                            max_tokens=getattr(self.config, "step_inference_max_tokens", 2048),
                        )
                        step_msg = step_resp.choices[0].message if step_resp and step_resp.choices else {}
                        step_tool_calls = step_msg.get("tool_calls") if isinstance(step_msg, dict) else getattr(step_msg, "tool_calls", None)
                        if step_tool_calls and isinstance(step_tool_calls, list):
                            # Pick the first matching call
                            for tc in step_tool_calls:
                                fn = (tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)) or {}
                                fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                                if fn_name == action:
                                    args_str = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "{}")
                                    try:
                                        resolved_args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                                    except Exception:
                                        resolved_args = {}
                                    break
                        if not resolved_args:
                            raise ValueError(
                                f"Plan step {step_num} requires tool '{action}' but no args were provided "
                                "and argument inference failed."
                            )

                fn_call = {
                    "name": action,
                    "arguments": json.dumps(resolved_args)
                }
                return await self._executor.execute(fn_call)
            else:
                # Fallback to old method
                tool_args = _resolve_args(step.args)
                return await self._execute_tool(action, tool_args)
        else:
            # Unknown action - log and continue
            self._logger.warning(f"Unknown action '{action}' in plan step {step_num}, skipping")
            return f"Skipped unknown action: {action}"

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
    # STRUCTURED OUTPUT HELPERS                                             #
    # --------------------------------------------------------------------- #
    def _resolve_response_format(self):
        """
        Resolve response_format to an OutputSchema configuration.
        
        Returns:
            OutputSchema or None
        """
        if self.response_format is None:
            return None
        
        from nucleusiq.agents.structured_output import (
            resolve_output_config,
            get_provider_from_llm,
        )
        
        model_name = getattr(self.llm, "model_name", "") if self.llm else ""
        provider = get_provider_from_llm(self.llm)
        
        return resolve_output_config(
            self.response_format,
            model_name=model_name,
            provider=provider,
        )
    
    def _get_structured_output_kwargs(self, output_config) -> dict:
        """
        Build LLM call kwargs for structured output.
        
        This is the centralized method used by all execution modes (DIRECT, STANDARD, AUTONOMOUS)
        to consistently handle structured output configuration.
        
        Args:
            output_config: Resolved OutputSchema configuration
            
        Returns:
            Dict with 'response_format' key if structured output is configured, empty dict otherwise
        """
        if output_config is None:
            return {}
        
        from nucleusiq.agents.structured_output import OutputMode, OutputSchema, get_provider_from_llm
        
        if output_config._resolved_mode != OutputMode.NATIVE:
            # Mode not implemented - validate will raise helpful error
            from nucleusiq.agents.structured_output import OutputMode as OM
            OM.validate_mode(output_config._resolved_mode)
            return {}  # Won't reach here if validation fails
        
        # Check if user passed explicit OutputSchema with custom settings
        if isinstance(self.response_format, OutputSchema):
            # Use for_provider() to get properly formatted response with strict/mode settings
            provider = get_provider_from_llm(self.llm) or "openai"
            provider_format = output_config.for_provider(provider)
            # Pass both: provider format for API, schema type for parsing
            return {"response_format": (provider_format, output_config.schema)}
        else:
            # Simple schema passed - let LLM provider handle it
            return {"response_format": output_config.schema}
    
    def _wrap_structured_output_result(self, response, output_config) -> Any:
        """
        Wrap LLM response with structured output metadata.
        
        This is the centralized method used by all execution modes to consistently
        return structured output results.
        
        Args:
            response: The LLM response (could be validated instance or raw response)
            output_config: Resolved OutputSchema configuration
            
        Returns:
            Dict with 'output', 'schema', and optional metadata if structured output,
            otherwise returns the raw response
        """
        if output_config is None:
            return response
        
        from nucleusiq.agents.structured_output import OutputMode
        
        if output_config._resolved_mode == OutputMode.NATIVE:
            # Response should be a validated instance (no 'choices' attribute)
            if not hasattr(response, 'choices'):
                return {
                    "output": response,
                    "schema": output_config.schema_name,
                    "mode": "native"
                }
        
        # If we get here with a normal response, extract content
        if hasattr(response, 'choices') and response.choices:
            msg = response.choices[0].message
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            return content
        
        return response

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
            return await self._run_autonomous(task)
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

        # Resolve structured output configuration (centralized)
        output_config = self._resolve_response_format()

        # Single LLM call (no tools)
        try:
            # Build LLM call kwargs
            call_kwargs = {
                "model": getattr(self.llm, "model_name", "default"),
                "messages": messages,
                "tools": None,  # Direct mode: no tools
                "max_tokens": getattr(self.config, "llm_max_tokens", 1024),
            }
            
            # Add structured output kwargs (centralized helper)
            call_kwargs.update(self._get_structured_output_kwargs(output_config))
            
            response = await self.llm.call(**call_kwargs)
            
            # Handle structured output result (centralized helper)
            wrapped_result = self._wrap_structured_output_result(response, output_config)
            if isinstance(wrapped_result, dict) and "output" in wrapped_result:
                self.state = AgentState.COMPLETED
                return wrapped_result

            if not response or not hasattr(response, "choices") or not response.choices:
                raise ValueError("LLM returned empty response")

            # Extract content (handle both string and list-of-parts formats)
            msg = response.choices[0].message
            if isinstance(msg, dict):
                raw_content = msg.get("content")
            else:
                raw_content = getattr(msg, "content", None)
            
            # Normalize content (can be string, list of content parts, or None)
            if isinstance(raw_content, str) and raw_content.strip():
                content = raw_content
            elif isinstance(raw_content, list):
                # Content parts format: [{"type": "text", "text": "..."}]
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            text_parts.append(t)
                content = "\n".join(text_parts) if text_parts else None
            else:
                content = None

            if content:
                self.state = AgentState.COMPLETED
                return content

            # Model returned empty content - this often means the task needs tools/planning
            self._logger.warning("LLM returned no content in DIRECT mode (task may require tools)")
            self.state = AgentState.COMPLETED
            objective = task.objective if isinstance(task, Task) else task.get('objective', '')
            return (
                f"No response from LLM. The task '{objective[:100]}...' may require tools or planning. "
                "Try using STANDARD or AUTONOMOUS execution mode."
            )
                
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

        # (2.5) Resolve structured output configuration (centralized)
        output_config = self._resolve_response_format()

        # (3) LLM call loop (may request multiple tools)
        try:
            max_tool_calls = 10  # Prevent infinite loops
            tool_call_count = 0
            
            empty_retries_remaining = 1

            def _content_to_text(content: Any) -> Optional[str]:
                if content is None:
                    return None
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                    joined = "\n".join(p for p in parts if p.strip())
                    return joined if joined.strip() else None
                s = str(content)
                return s if s.strip() else None

            while tool_call_count < max_tool_calls:
                # Build LLM call kwargs with structured output support
                call_kwargs = {
                    "model": getattr(self.llm, "model_name", "default"),
                    "messages": messages,
                    "tools": tool_specs if tool_specs else None,
                    "max_tokens": getattr(self.config, "llm_max_tokens", 2048),
                }
                # Add structured output kwargs (centralized helper)
                call_kwargs.update(self._get_structured_output_kwargs(output_config))
                
                response = await self.llm.call(**call_kwargs)
                
                # Handle structured output response first
                wrapped_result = self._wrap_structured_output_result(response, output_config)
                if isinstance(wrapped_result, dict) and "output" in wrapped_result:
                    self.state = AgentState.COMPLETED
                    return wrapped_result

                if not response or not hasattr(response, "choices") or not response.choices:
                    raise ValueError("LLM returned empty response")

                msg = response.choices[0].message
                if isinstance(msg, dict):
                    tool_calls = msg.get("tool_calls")
                    refusal = msg.get("refusal")
                    content = _content_to_text(msg.get("content"))
                else:
                    tool_calls = getattr(msg, "tool_calls", None)
                    refusal = getattr(msg, "refusal", None)
                    content = _content_to_text(getattr(msg, "content", None))

                if refusal:
                    self.state = AgentState.ERROR
                    return f"Error: LLM refused request: {refusal}"

                # (4) Handle tool calls (modern OpenAI format: tool_calls array)
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # Process each tool call in the response
                    tool_results = []
                    assistant_msg = {
                        "role": "assistant",
                        "content": msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None),
                        "tool_calls": tool_calls,
                    }
                    messages.append(assistant_msg)
                    
                    for tool_call in tool_calls:
                        tool_call_count += 1
                        if tool_call_count > max_tool_calls:
                            break
                            
                        # Extract tool call info (modern format)
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id")
                            fn_info = tool_call.get("function", {})
                            fn_name = fn_info.get("name") if isinstance(fn_info, dict) else None
                            fn_args_str = fn_info.get("arguments", "{}") if isinstance(fn_info, dict) else "{}"
                        else:
                            tool_call_id = getattr(tool_call, "id", None)
                            fn_info = getattr(tool_call, "function", None)
                            fn_name = getattr(fn_info, "name", None) if fn_info else None
                            fn_args_str = getattr(fn_info, "arguments", "{}") if fn_info else "{}"
                        
                        if not fn_name:
                            self._logger.warning("Tool call missing function name, skipping")
                            continue
                        
                        self._logger.info("Tool requested: %s", fn_name)
                        
                        try:
                            # Create function_call dict for Executor (it expects name/arguments)
                            fn_call_dict = {"name": fn_name, "arguments": fn_args_str}
                            # Use Executor to execute tool
                            tool_result = await self._executor.execute(fn_call_dict)
                            
                            # Append tool result in modern format (role="tool" with tool_call_id)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(tool_result),
                            })
                            
                        except Exception as e:
                            # Fire-and-forget: return error immediately
                            self._logger.error(f"Tool execution failed: {e}")
                            self.state = AgentState.ERROR
                            return f"Error: Tool '{fn_name}' execution failed: {str(e)}"
                    
                    # Continue loop to get final answer after tool execution
                    continue

                # (5) No tool calls - return content
                if content:
                    self.state = AgentState.COMPLETED
                    
                    # Store in memory if enabled
                    if self.memory and self.config.enable_memory:
                        try:
                            await self.memory.store(task, content)
                        except Exception as e:
                            self._logger.warning(f"Failed to store in memory: {e}")
                    
                    return content

                # No content and no tool calls - this is a failure
                if empty_retries_remaining > 0:
                    empty_retries_remaining -= 1
                    # Add a nudge and retry; transient empty messages do occur.
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your last message was empty. You MUST either call a tool or provide a final answer.",
                        }
                    )
                    continue

                # LLM failed to respond properly after retry - this is an error, not success
                self._logger.error("LLM returned no tool calls and no content after retry")
                self.state = AgentState.ERROR
                objective = task.objective if isinstance(task, Task) else task.get('objective', '')
                return f"Error: LLM did not respond. Task '{objective[:80]}...' may require AUTONOMOUS mode for multi-step planning."
            
            # Max tool calls reached
            self._logger.warning(f"Maximum tool calls ({max_tool_calls}) reached")
            self.state = AgentState.ERROR
            return f"Error: Maximum tool calls ({max_tool_calls}) reached"
                
        except Exception as e:
            self._logger.error(f"Error during standard execution: {str(e)}")
            self.state = AgentState.ERROR
            return f"Error: Standard execution failed: {str(e)}"
    
    async def _run_autonomous(self, task: Union[Task, Dict[str, Any]]) -> Any:
        """
        Gear 3: Autonomous mode - Full reasoning loop with planning and self-correction.
        
        Logic: Input → Plan → Execute Plan → Self-Correct → Result
        
        Use Cases: Complex multi-step tasks, research, analysis, problem-solving
        
        Characteristics:
        - Automatic planning (calls plan() internally)
        - Multi-step execution following the plan
        - Context building across steps
        - Self-correction capabilities (future enhancement)
        - Memory enabled for context retention
        - Iterative refinement (future enhancement)
        
        Args:
            task: Task instance or dictionary
            
        Returns:
            Execution result from plan execution
        """
        self._logger.debug("Executing in AUTONOMOUS mode (planning + execution)")
        self.state = AgentState.PLANNING
        
        # Check if LLM is available (required for planning)
        if not self.llm:
            self._logger.warning("No LLM configured for autonomous mode, falling back to standard mode")
            return await self._run_standard(task)
        
        try:
            # Step 1: Generate plan automatically (user doesn't need to call plan() explicitly)
            self._logger.info("Autonomous mode: Generating execution plan...")
            
            # Get timeout from config
            planning_timeout = getattr(self.config, 'planning_timeout', 120)
            max_retries = getattr(self.config, 'max_retries', 3)
            
            context = await self._get_context(task)
            
            # Try LLM-based planning with timeout and retry
            plan = None
            last_error = None
            for attempt in range(max_retries):
                try:
                    plan = await asyncio.wait_for(
                        self._create_llm_plan(task, context),
                        timeout=planning_timeout
                    )
                    self._logger.info(f"Generated plan with {len(plan)} steps using LLM")
                    break
                except asyncio.TimeoutError:
                    last_error = f"Planning timed out after {planning_timeout}s (attempt {attempt + 1}/{max_retries})"
                    self._logger.warning(last_error)
                    if attempt < max_retries - 1:
                        self._logger.info(f"Retrying planning (attempt {attempt + 2}/{max_retries})...")
                        await asyncio.sleep(1)  # Brief pause before retry
                except Exception as e:
                    last_error = str(e)
                    self._logger.warning(f"LLM planning failed (attempt {attempt + 1}): {e}")
                    break  # Don't retry on non-timeout errors
            
            # Fallback to basic plan if LLM planning failed
            if plan is None:
                self._logger.warning(f"LLM planning failed after retries: {last_error}. Falling back to basic plan.")
                plan = await self.plan(task)
            
            # Log plan details
            if len(plan.steps) > 1:
                self._logger.info("Multi-step plan generated:")
                for step in plan.steps:
                    self._logger.debug(f"  Step {step.step}: {step.action}" + 
                                     (f" - {step.details}" if step.details else ""))
            else:
                self._logger.debug("Single-step plan (direct execution)")
            
            # Step 2: Execute the plan
            self._logger.info("Autonomous mode: Executing plan...")
            result = await self._execute_plan(task, plan)
            
            # Check if plan execution failed (result starts with "Error:")
            if isinstance(result, str) and result.strip().startswith("Error:"):
                # State was already set to ERROR by _execute_plan
                return result
            
            # Step 3: Store in memory if enabled
            if self.memory and self.config.enable_memory:
                try:
                    await self.memory.store(task, result)
                except Exception as e:
                    self._logger.warning(f"Failed to store in memory: {e}")
            
            # Step 4: Wrap result with structured output if configured (centralized)
            output_config = self._resolve_response_format()
            if output_config is not None:
                # For AUTONOMOUS mode, the result is already computed by tool execution
                # We wrap it in the structured output format for consistency
                self.state = AgentState.COMPLETED
                return {
                    "output": result,
                    "schema": output_config.schema_name if hasattr(output_config, 'schema_name') else "Result",
                    "mode": "autonomous"
                }
            
            self.state = AgentState.COMPLETED
            return result
            
        except Exception as e:
            self._logger.error(f"Error during autonomous execution: {str(e)}")
            self.state = AgentState.ERROR
            # Fallback to standard mode on error
            self._logger.warning("Falling back to standard mode due to error")
            return await self._run_standard(task)
    
    async def _execute_direct(self, task: Union[Task, Dict[str, Any]]) -> Any:
        """
        [DEPRECATED] Execute task directly without plan.
        
        This method is kept for backward compatibility.
        It now delegates to _run_standard() which uses the Executor component.
        
        Use execution_mode="standard" instead.
        """
        self._logger.warning("_execute_direct() is deprecated, use _run_standard() or set execution_mode='standard'")
        return await self._run_standard(task)