# src/nucleusiq/agents/react_agent.py
"""
ReAct (Reasoning + Acting) Agent Implementation.

ReAct agents alternate between:
- Thought: Reasoning about what to do next
- Action: Taking an action (tool call)
- Observation: Processing tool results
- Loop until final answer
"""

from typing import Any, Dict, List, Optional, Union
import json
import re
from pydantic import Field, PrivateAttr
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentState


class ReActAgent(Agent):
    """
    ReAct (Reasoning + Acting) Agent.
    
    Implements the ReAct pattern with explicit Thought-Action-Observation loops.
    This extends the base Agent class to support iterative reasoning and tool usage.
    
    Example:
        ```python
        agent = ReActAgent(
            name="ReActAgent",
            role="Assistant",
            objective="Answer questions using reasoning and tools",
            narrative="A ReAct agent",
            llm=llm,
            tools=[calculator, web_search],
            max_iterations=10
        )
        
        task = {"id": "task1", "objective": "What is 15 + 27 and what's the weather?"}
        result = await agent.execute(task)
        ```
    """
    
    max_iterations: int = Field(default=10, description="Maximum number of Thought-Action-Observation cycles")
    _react_history: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    
    async def execute(self, task: Union[Dict[str, Any], Any]) -> Any:
        """
        Execute task using ReAct pattern.
        
        ReAct Loop:
        1. Thought: LLM reasons about next step
        2. Action: LLM decides to use a tool or give final answer
        3. Observation: Tool result (if action was tool call)
        4. Repeat until final answer or max_iterations reached
        
        Args:
            task: Task dictionary with 'id' and 'objective' keys, or Task object
            
        Returns:
            Final answer from the agent
        """
        self.state = AgentState.EXECUTING
        self._react_history = []
        
        if not self.llm:
            self._logger.warning("No LLM configured, falling back to echo mode")
            self.state = AgentState.COMPLETED
            # Handle both Task object and dict
            if isinstance(task, dict):
                objective = task.get('objective', '')
            else:
                objective = getattr(task, 'objective', '')
            return f"Echo: {objective}"
        
        # Build initial messages with ReAct instructions
        messages: list[dict[str, Any]] = self._build_react_messages(task)
        
        # ReAct loop
        for iteration in range(self.max_iterations):
            self._logger.info(f"ReAct iteration {iteration + 1}/{self.max_iterations}")
            
            # Call LLM for Thought and Action
            tool_specs = []
            if self.tools and self.llm:
                tool_specs = self.llm.convert_tool_specs(self.tools)
            
            response = await self.llm.call(
                model=getattr(self.llm, "model_name", "default"),
                messages=messages,
                tools=tool_specs if tool_specs else None
            )
            
            if not response or not hasattr(response, "choices") or not response.choices:
                raise ValueError("LLM returned empty response")
            
            message = response.choices[0].message
            content = self._extract_content(message)
            
            # Parse Thought and Action from response
            thought, action = self._parse_react_response(content, message)
            
            # Store in history
            self._react_history.append({
                "iteration": iteration + 1,
                "thought": thought,
                "action": action
            })
            
            self._logger.debug(f"Thought: {thought}")
            self._logger.debug(f"Action: {action}")
            
            # Check if we have a final answer
            if action["type"] == "final_answer":
                self.state = AgentState.COMPLETED
                return action["answer"]
            
            # Execute action (tool call)
            if action["type"] == "tool":
                tool_name = action["name"]
                # Remove namespace prefix if present (e.g., "functions.add" -> "add")
                if "." in tool_name:
                    tool_name = tool_name.split(".")[-1]
                tool_args = action["args"]
                
                # Find and execute tool
                tool = next((t for t in self.tools if hasattr(t, 'name') and t.name == tool_name), None)
                if not tool:
                    observation = f"Error: Tool '{tool_name}' not found"
                elif hasattr(tool, 'is_native') and tool.is_native:
                    observation = f"Error: Tool '{tool_name}' is a native tool and cannot be executed directly"
                else:
                    try:
                        tool_result = await tool.execute(**tool_args)
                        observation = tool_result
                    except Exception as e:
                        observation = f"Error: {str(e)}"
                
                # Add to messages for next iteration
                messages.append({
                    "role": "assistant",
                    "content": content
                })
                # Sanitize tool name for OpenAI (only alphanumeric, underscore, hyphen)
                sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_name)
                messages.append({
                    "role": "function",
                    "name": sanitized_name,
                    "content": json.dumps(observation) if not isinstance(observation, str) else str(observation)
                })
                
                # Update history with observation
                self._react_history[-1]["observation"] = observation
                
                self._logger.info(f"Tool '{tool_name}' executed. Observation: {observation}")
            else:
                # Unknown action type - treat as error
                observation = f"Error: Unknown action type '{action['type']}'"
                messages.append({
                    "role": "assistant",
                    "content": content
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
                
                self._react_history[-1]["observation"] = observation
        
        # Max iterations reached
        self.state = AgentState.ERROR
        return f"Max iterations ({self.max_iterations}) reached. Task incomplete."
    
    def _build_react_messages(self, task: Union[Dict[str, Any], Any]) -> List[Dict[str, Any]]:
        """Build initial messages with ReAct instructions."""
        messages = []
        
        # System message with ReAct instructions
        tool_names = [t.name for t in self.tools] if self.tools else []
        tool_list = ", ".join(tool_names) if tool_names else "None"
        
        react_system = f"""You are a ReAct agent. Follow this pattern:

Thought: [Your reasoning about what to do next]
Action: [Tool name to use, or "Final Answer" if you have the answer]
Action Input: [Input for the tool as JSON, or your final answer]

After each action, you will receive an Observation. Use it to inform your next Thought.

Available tools: {tool_list}

Always follow the Thought -> Action -> Observation pattern. When you have the final answer, use Action: Final Answer."""
        
        messages.append({"role": "system", "content": react_system})
        
        # Add prompt if provided (can override or supplement ReAct instructions)
        if self.prompt:
            if hasattr(self.prompt, 'system') and self.prompt.system:
                messages.append({"role": "system", "content": self.prompt.system})
            if hasattr(self.prompt, 'user') and self.prompt.user:
                messages.append({"role": "user", "content": self.prompt.user})
        
        # Add task objective (handle both Task object and dict)
        if isinstance(task, dict):
            task_objective = task.get('objective', '')
        else:
            # Task is a Task object (Pydantic model)
            task_objective = getattr(task, 'objective', '')
        
        messages.append({
            "role": "user",
            "content": f"Task: {task_objective}"
        })
        
        return messages
    
    def _parse_react_response(self, content: str, message: Any) -> tuple[str, Dict[str, Any]]:
        """
        Parse Thought and Action from LLM response.
        
        Returns:
            Tuple of (thought, action_dict)
        """
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', content, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        
        # Check for function call (Action via function calling)
        if isinstance(message, dict):
            fn_call = message.get("function_call")
        else:
            fn_call = getattr(message, "function_call", None)
        
        if fn_call:
            # Tool action via function calling
            fn_name = fn_call.get("name") if isinstance(fn_call, dict) else getattr(fn_call, "name", None)
            fn_args_str = fn_call.get("arguments") if isinstance(fn_call, dict) else getattr(fn_call, "arguments", "{}")
            try:
                fn_args = json.loads(fn_args_str or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            
            return thought, {
                "type": "tool",
                "name": fn_name,
                "args": fn_args
            }
        
        # Check for "Final Answer" in content
        final_answer_match = re.search(
            r'Action:\s*Final Answer\s*Action Input:\s*(.+?)$',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if final_answer_match:
            return thought, {
                "type": "final_answer",
                "answer": final_answer_match.group(1).strip()
            }
        
        # Check for action in content (text-based format)
        action_match = re.search(
            r'Action:\s*(.+?)(?:\s+Action Input:\s*(.+?))?$',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action_name = action_match.group(1).strip()
            action_input = action_match.group(2).strip() if action_match.group(2) else ""
            
            if action_name.lower() in ["final answer", "final_answer", "answer"]:
                return thought, {
                    "type": "final_answer",
                    "answer": action_input or content
                }
            else:
                # Try to parse as tool call
                try:
                    args = json.loads(action_input) if action_input else {}
                    return thought, {
                        "type": "tool",
                        "name": action_name,
                        "args": args
                    }
                except (json.JSONDecodeError, ValueError):
                    return thought, {
                        "type": "unknown",
                        "raw": content
                    }
        
        # Default: treat as final answer if no action pattern found
        return thought, {
            "type": "final_answer",
            "answer": content
        }
    
    def _extract_content(self, message: Any) -> str:
        """Extract content from message (handles both dict and object)."""
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        return content if content is not None else ""
    
    def get_react_history(self) -> List[Dict[str, Any]]:
        """
        Get the ReAct execution history.
        
        Returns:
            List of dictionaries containing Thought-Action-Observation steps
        """
        return self._react_history
    
    def get_last_thought(self) -> Optional[str]:
        """Get the last thought from the ReAct history."""
        if self._react_history:
            return self._react_history[-1].get("thought")
        return None
    
    def get_last_action(self) -> Optional[Dict[str, Any]]:
        """Get the last action from the ReAct history."""
        if self._react_history:
            return self._react_history[-1].get("action")
        return None

