# File: src/nucleusiq/llms/mock_llm.py
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
    """
    Mock Language Model for testing function-calling.

    On the first call (with `tools` provided), returns a fake function_call.
    On the second call, returns a final content response.
    """
    def __init__(self):
        self._call_count = 0

    class Message:
        def __init__(self, content: Optional[str] = None, function_call: Optional[Dict[str, Any]] = None):
            self.content = content
            self.function_call = function_call

    class Choice:
        def __init__(self, message: 'MockLLM.Message'):
            self.message = message

    class LLMResponse:
        def __init__(self, choices: List['MockLLM.Choice']):
            self.choices = choices

    async def call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
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
            user_msg = next((m['content'] for m in messages if m.get('role') == 'user'), '')
            # Extract integers from the user prompt
            nums = re.findall(r'-?\d+', user_msg)
            params = list(tools[0]['parameters']['properties'].keys())
            args = {params[i]: int(nums[i]) for i in range(min(len(nums), len(params)))}
            fn_call = {'name': tools[0]['name'], 'arguments': json.dumps(args)}
            msg = self.Message(content=None, function_call=fn_call)
            return self.LLMResponse([self.Choice(msg)])

        # Subsequent call or no tools: return a normal completion
        last = messages[-1]
        if last.get('role') == 'function':
            body = last.get('content')
            reply = f"Dummy Model final answer incorporating function output is: {body}"
        else:
            reply = f"Echo: {messages[-1].get('content', '')}"
        msg = self.Message(content=reply)
        return self.LLMResponse([self.Choice(msg)])