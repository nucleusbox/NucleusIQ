"""
Executor Component - Handles tool execution.

The Executor is responsible for:
- Tool selection and validation
- Argument parsing and validation
- Tool execution
- Error handling
- Result formatting
"""

import json
import logging
from typing import Any, Dict, List, Optional

from nucleusiq.core.tools.base_tool import BaseTool
from nucleusiq.agents.plan import PlanStep
from nucleusiq.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class Executor:
    """
    Executor component for handling tool execution.
    
    The Executor acts as the "hands" of the agent, responsible for:
    - Validating tool calls
    - Executing tools with proper arguments
    - Handling errors gracefully
    - Supporting both BaseTool and native tools
    """
    
    def __init__(self, llm: BaseLLM, tools: List[BaseTool]):
        """
        Initialize the Executor.
        
        Args:
            llm: LLM instance (for potential future use)
            tools: List of available tools
        """
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools if hasattr(tool, 'name')}
        logger.debug(f"Executor initialized with {len(self.tools)} tools: {list(self.tools.keys())}")
    
    async def execute(self, fn_call: Dict[str, Any]) -> Any:
        """
        Execute a single tool call.
        
        Args:
            fn_call: Function call dictionary with 'name' and 'arguments' keys
                Example: {"name": "add", "arguments": '{"a": 5, "b": 3}'}
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool not found or is a native tool
        """
        tool_name = fn_call.get("name")
        if not tool_name:
            raise ValueError("Function call missing 'name' field")
        
        # Parse arguments
        fn_args_str = fn_call.get("arguments", "{}")
        try:
            fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in function arguments: {e}")
        
        logger.debug(f"Executing tool '{tool_name}' with args: {fn_args}")
        
        # Find tool
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")
        
        tool = self.tools[tool_name]
        
        # Check if it's a native tool (handled by LLM, not executed here)
        is_native = getattr(tool, 'is_native', False)
        if is_native:
            raise ValueError(
                f"Tool '{tool_name}' is a native tool and is handled directly by the LLM. "
                f"Native tools should not be executed via Executor."
            )
        
        # Execute tool
        try:
            result = await tool.execute(**fn_args)
            logger.debug(f"Tool '{tool_name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            raise
    
    async def execute_step(self, step: PlanStep, context: Dict[str, Any]) -> Any:
        """
        Execute a plan step with context.
        
        Args:
            step: PlanStep instance with action and args
            context: Context dictionary from previous steps
        
        Returns:
            Step execution result
        """
        # Merge context into step args
        step_args = step.args or {}
        merged_args = {**context, **step_args}
        
        # Create function call from step
        fn_call = {
            "name": step.action,
            "arguments": json.dumps(merged_args)
        }
        
        logger.debug(f"Executing plan step {step.step}: {step.action} with context")
        return await self.execute(fn_call)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)


