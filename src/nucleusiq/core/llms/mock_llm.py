# File: src/nucleusiq/core/llms/mock_llm.py
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from nucleusiq.streaming.events import StreamEvent

from .base_llm import BaseLLM


class MockLLM(BaseLLM):
    """Mock Language Model for testing.

    * First ``call()`` with *tools* → returns a fake ``function_call``.
    * Second ``call()`` → returns a final content response.
    * ``call_stream()`` → yields tokens character-by-character for
      testing consumer streaming logic.
    """

    model_name: str = "mock-model"

    def __init__(
        self,
        model_name: str = "mock-model",
        *,
        stream_chunk_size: int = 1,
    ):
        self.model_name = model_name
        self.stream_chunk_size = stream_chunk_size
        self._call_count = 0

    # ------------------------------------------------------------------ #
    # Internal response types                                             #
    # ------------------------------------------------------------------ #

    class Message:
        def __init__(
            self,
            content: str | None = None,
            function_call: dict[str, Any] | None = None,
            tool_calls: list[dict[str, Any]] | None = None,
        ):
            self.content = content
            self.function_call = function_call
            self.tool_calls = tool_calls

    class Choice:
        def __init__(self, message: "MockLLM.Message"):
            self.message = message

    class LLMResponse:
        def __init__(self, choices: list["MockLLM.Choice"]):
            self.choices = choices

    # ------------------------------------------------------------------ #
    # Non-streaming call                                                  #
    # ------------------------------------------------------------------ #

    async def call(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> "MockLLM.LLMResponse":
        self._call_count += 1

        if self._call_count == 1 and tools:
            return self._tool_call_response(messages, tools)

        return self._content_response(messages)

    # ------------------------------------------------------------------ #
    # Streaming call                                                      #
    # ------------------------------------------------------------------ #

    async def call_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield mock token events, then a complete event."""
        self._call_count += 1

        if self._call_count == 1 and tools:
            resp = self._tool_call_response(messages, tools)
            tool_calls_meta = []
            msg = resp.choices[0].message
            if msg.tool_calls:
                tool_calls_meta = msg.tool_calls
            yield StreamEvent.complete_event(
                "", metadata={"tool_calls": tool_calls_meta}
            )
            return

        full_text = self._build_reply(messages)
        chunk = self.stream_chunk_size

        for i in range(0, len(full_text), chunk):
            yield StreamEvent.token_event(full_text[i : i + chunk])

        yield StreamEvent.complete_event(full_text)

    # ------------------------------------------------------------------ #
    # Sync compatibility shim                                             #
    # ------------------------------------------------------------------ #

    def create_completion(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """Synchronous completion for ``AutoChainOfThought`` compatibility."""
        import asyncio

        try:
            asyncio.get_running_loop()
            return "This is a mock reasoning chain."
        except RuntimeError:
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
                if response and response.choices:
                    message = response.choices[0].message
                    if isinstance(message, dict):
                        return message.get("content", "This is a mock reasoning chain.")
                    return getattr(
                        message, "content", "This is a mock reasoning chain."
                    )
                return "This is a mock reasoning chain."
            except RuntimeError:
                return "This is a mock reasoning chain."

    def convert_tool_specs(self, tools: list[Any]) -> list[dict[str, Any]]:
        tool_specs = []
        for tool in tools:
            if hasattr(tool, "get_spec"):
                tool_specs.append(tool.get_spec())
            else:
                tool_specs.append(tool)
        return tool_specs

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _tool_call_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> "MockLLM.LLMResponse":
        user_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        tool = tools[0]
        if "function" in tool:
            tool_name = tool["function"]["name"]
            tool_params = tool["function"].get("parameters", {})
        else:
            tool_name = tool.get("name", "")
            tool_params = tool.get("parameters", {})

        params = list(tool_params.get("properties", {}).keys())
        nums = re.findall(r"-?\d+", user_msg)
        args = {params[i]: int(nums[i]) for i in range(min(len(nums), len(params)))}

        fn_call = {"name": tool_name, "arguments": json.dumps(args)}
        tool_calls = [
            {
                "id": "call_mock_1",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)},
            }
        ]
        msg = self.Message(content=None, function_call=fn_call, tool_calls=tool_calls)
        return self.LLMResponse([self.Choice(msg)])

    def _content_response(
        self,
        messages: list[dict[str, Any]],
    ) -> "MockLLM.LLMResponse":
        reply = self._build_reply(messages)
        msg = self.Message(content=reply)
        return self.LLMResponse([self.Choice(msg)])

    @staticmethod
    def _build_reply(messages: list[dict[str, Any]]) -> str:
        last = messages[-1]
        if last.get("role") in ("function", "tool"):
            body = last.get("content", "")
            return f"Dummy Model final answer incorporating function output is: {body}"
        return f"Echo: {messages[-1].get('content', '')}"
