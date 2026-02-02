# File: src/nucleusiq/core/llms/mock_llm.py
import re
import json
from typing import List, Dict, Any, Optional

from .base_llm import BaseLLM

class MockLLM(BaseLLM):
    """
    Mock Language Model for testing function-calling.

    On the first call (with `tools` provided), returns a fake function_call.
    On the second call, returns a final content response.
    """
    model_name: str = "mock-model"

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self._call_count = 0

    class Message:
        def __init__(
            self,
            content: Optional[str] = None,
            function_call: Optional[Dict[str, Any]] = None,
            tool_calls: Optional[List[Dict[str, Any]]] = None,
        ):
            self.content = content
            self.function_call = function_call
            # Modern format: agent checks tool_calls first
            self.tool_calls = tool_calls

    class Choice:
        def __init__(self, message: 'MockLLM.Message'):
            self.message = message

    class LLMResponse:
        def __init__(self, choices: List['MockLLM.Choice']):
            self.choices = choices

    # override method from BaseLLM
    async def call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> 'MockLLM.LLMResponse':
        self._call_count += 1
        # First call: simulate a function_call if tools are provided
        if self._call_count == 1 and tools:
            user_msg = next((m['content'] for m in reversed(messages) if m.get('role') == 'user'), '')
            # Extract integers from the user prompt
            # Handle both OpenAI format {"type": "function", "function": {...}} and generic format
            tool = tools[0]
            if "function" in tool:
                # OpenAI format
                tool_name = tool["function"]["name"]
                tool_params = tool["function"].get("parameters", {})
            else:
                # Generic format
                tool_name = tool.get("name", "")
                tool_params = tool.get("parameters", {})
            
            params = list(tool_params.get("properties", {}).keys())
            nums = re.findall(r'-?\d+', user_msg)
            args = {params[i]: int(nums[i]) for i in range(min(len(nums), len(params)))}
            fn_call = {'name': tool_name, 'arguments': json.dumps(args)}
            # Modern tool_calls format (agent checks this first)
            tool_calls = [{
                "id": "call_mock_1",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)},
            }]
            msg = self.Message(content=None, function_call=fn_call, tool_calls=tool_calls)
            return self.LLMResponse([self.Choice(msg)])

        # Subsequent call or no tools: return a normal completion
        last = messages[-1]
        if last.get('role') in ('function', 'tool'):
            body = last.get('content', '')
            reply = f"Dummy Model final answer incorporating function output is: {body}"
        else:
            reply = f"Echo: {messages[-1].get('content', '')}"
        msg = self.Message(content=reply)
        return self.LLMResponse([self.Choice(msg)])
    
    def create_completion(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Synchronous completion method for compatibility with AutoChainOfThought.
        
        This method wraps the async call() method for synchronous use.
        Note: This is a simplified version that returns a string directly.
        """
        import asyncio
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context - return mock response
            return "This is a mock reasoning chain."
        except RuntimeError:
            # No running loop, we can use asyncio.run()
            try:
                response = asyncio.run(
                    self.call(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=kwargs.get("max_tokens", 150),
                        temperature=kwargs.get("temperature", 0.5),
                        top_p=kwargs.get("top_p", 1.0),
                        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                        presence_penalty=kwargs.get("presence_penalty", 0.0),
                    )
                )
                # Extract content from response
                if response and response.choices:
                    message = response.choices[0].message
                    if isinstance(message, dict):
                        return message.get("content", "This is a mock reasoning chain.")
                    else:
                        return getattr(message, "content", "This is a mock reasoning chain.")
                return "This is a mock reasoning chain."
            except RuntimeError:
                # Fallback if asyncio.run() fails
                return "This is a mock reasoning chain."
    
    def convert_tool_specs(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        MockLLM doesn't perform any special tool conversion, just returns the specs as-is.
        """
        tool_specs = []
        for tool in tools:
            if hasattr(tool, 'get_spec'):
                tool_specs.append(tool.get_spec())
            else:
                tool_specs.append(tool)
        return tool_specs

