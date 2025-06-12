# File: src/nucleusiq/agents/agent.py
import json
import logging
from typing import Any, Dict, List
import asyncio
import logging
import json

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.core.tools.base_tool import BaseTool
from typing import Dict, Any, List

class EchoAgent(Agent):
    """
    A simple agent that echoes the task objective or invokes tools
    following OpenAI-style function-calling steps.
    """
    async def initialize(self) -> None:
        # Initialize tools (plain functions no-op)
        for tool in getattr(self, 'tools', []):
            await tool.initialize()
        self.state = AgentState.COMPLETED
    
    async def plan(self, task):
        return await super().plan(task)

    async def execute(self, task: Dict[str, Any]) -> Any:
        # 1) Collect tool specs for the LLM
        tools_spec = [t.get_spec() for t in self.tools]

        # 2) Build message list  ───────────────────────────────────────────
        messages: List[Dict[str, Any]] = []

        if self.prompt and self.prompt.system:
            messages.append({"role": "system", "content": self.prompt.system})

        # First user turn from the zero-shot template (optional)
        if self.prompt and self.prompt.user:
            messages.append({"role": "user", "content": self.prompt.user})

        # Second user turn with the *actual* task text  
        messages.append({"role": "user", "content": task.get("objective", "")})
        print(messages)
        # 3) First LLM call (may return a function_call)
        response = await self.llm.call(
            model=self.llm.model_name,
            messages=messages,
            tools=tools_spec,
        )
        choice_msg = response.choices[0].message
        print(choice_msg)
        fn_call: Dict[str, Any] | None = getattr(choice_msg, "function_call", None)
        print("fn_call: ", fn_call)
        # 4) If a function was requested, execute the tool
        if fn_call:
            name = fn_call["name"]
            arguments_str = fn_call["arguments"]
            args = json.loads(arguments_str or "{}")
            print("args: ", args)

            self._logger.info(f"Calling tool {name} with args {args}")
            tool = next((t for t in self.tools if t.name == name), None)
            if tool is None:
                raise ValueError(f"Tool '{name}' not found")

            result = await tool.execute(**args)

            # 5) Feed the result back for a final model answer
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": name, "arguments": arguments_str},
                    },
                    {
                        "role": "function",
                        "name": name,
                        "content": json.dumps(result),
                    },
                ]
            )
            follow_up = await self.llm.call(model=self.llm.model_name, messages=messages)
            return follow_up.choices[0].message.content

        # 6) Fallback to a simple echo
        self._logger.info("No function call returned; echoing objective.")
        return f"Echo: {task.get('objective', '')}"


async def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Initialize mock LLM and prompt
    llm = MockLLM()
    zero_shot = PromptFactory.create_prompt(
        technique=PromptTechnique.ZERO_SHOT
    ).configure(
        system="You are a helpful assistant.",
        user="Compute the sum of two numbers or repeat the request."
    )

    # Wrap a simple function as a tool
    def add(a: int, b: int) -> int:
        return a + b
    adder = BaseTool.from_function(add, description="Add two integers.")

    # Instantiate EchoAgent with tools
    agent = EchoAgent(
        name="EchoBot",
        role="Echo Service",
        objective="Echo or compute sums.",
        narrative="EchoBot repeats or uses tools.",
        llm=llm,
        prompt=zero_shot,
        tools=[adder],
        config=AgentConfig(verbose=True)
    )

    await agent.initialize()

    # Execute a function-call task
    task = {"id": "task1", "objective": "Add 7 and 8."}
    try:
        result = await agent.execute(task)
        print(f"Task Result: {result}")
    except Exception as e:
        print(f"Task failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())