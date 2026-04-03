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
from typing import Any, Dict, List

from nucleusiq.agents.chat_models import ToolCallRequest
from nucleusiq.agents.plan import PlanStep
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.tools.base_tool import BaseTool
from nucleusiq.tools.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
)

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
        self.tools = {tool.name: tool for tool in tools if hasattr(tool, "name")}
        logger.debug(
            f"Executor initialized with {len(self.tools)} tools: {list(self.tools.keys())}"
        )

    async def execute(self, fn_call: ToolCallRequest | Dict[str, Any]) -> Any:
        """
        Execute a single tool call.

        Args:
            fn_call: ToolCallRequest or dict with 'name' and 'arguments' keys

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found or is a native tool
        """
        if isinstance(fn_call, ToolCallRequest):
            tool_name = fn_call.name
            fn_args_str = fn_call.arguments
        else:
            tool_name = fn_call.get("name")
            if not tool_name:
                raise ToolValidationError(
                    "Function call missing 'name' field",
                    tool_name="(missing)",
                )
            fn_args_str = fn_call.get("arguments", "{}")
        try:
            fn_args = (
                json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            )
        except json.JSONDecodeError as e:
            raise ToolValidationError(
                f"Invalid JSON in function arguments: {e}",
                tool_name=tool_name or "unknown",
                original_error=e,
            ) from e

        logger.debug(f"Executing tool '{tool_name}' with args: {fn_args}")

        if tool_name not in self.tools:
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found. "
                f"Available tools: {list(self.tools.keys())}",
                tool_name=tool_name,
            )

        tool = self.tools[tool_name]

        is_native = getattr(tool, "is_native", False)
        if is_native:
            raise ToolExecutionError(
                f"Tool '{tool_name}' is a native tool and is handled directly "
                f"by the LLM. Native tools should not be executed via Executor.",
                tool_name=tool_name,
            )

        try:
            result = await tool.execute(**fn_args)
            logger.debug(f"Tool '{tool_name}' executed successfully")
            return result
        except ToolExecutionError:
            raise
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            raise ToolExecutionError(
                f"Tool '{tool_name}' failed: {e}",
                tool_name=tool_name,
                original_error=e,
                args_snapshot=fn_args,
            ) from e

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
        fn_call = {"name": step.action, "arguments": json.dumps(merged_args)}

        logger.debug(f"Executing plan step {step.step}: {step.action} with context")
        return await self.execute(fn_call)

    def get_tool(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
