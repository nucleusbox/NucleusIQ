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
        # 1) Prepare tool specs
        tools_spec = [tool.get_spec() for tool in getattr(self, 'tools', [])]
        print(tools_spec)

        # 2) Build messages
        messages: List[Dict[str, Any]] = []
        if self.prompt and self.prompt.system:
            messages.append({"role": "system", "content": self.prompt.system})
        user_content = self.prompt.user if self.prompt and self.prompt.user else task.get("objective", "")
        messages.append({"role": "user", "content": user_content})
        print(messages)
        # 3) First LLM call with function specs
        response = await self.llm.call(
            model=self.llm.model_name,
            messages=messages,
            tools=tools_spec,
        )
        choice = response.choices[0].message
        print(choice)

        # 4) If function_call requested, execute tool
        fn_call = getattr(choice, "function_call", None)
        print(fn_call)
        if fn_call:
            args = json.loads(fn_call.arguments)
            self._logger.info(f"Calling tool {fn_call.name} with args {args}")
            tool = next((t for t in self.tools if t.name == fn_call.name), None)
            if not tool:
                raise ValueError(f"Tool '{fn_call.name}' not found")
            result = await tool.execute(**args)

            # 5) Send function result back
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {"name": fn_call.name, "arguments": fn_call.arguments}
            })
            messages.append({
                "role": "function",
                "name": fn_call.name,
                "content": json.dumps(result)
            })
            follow_up = await self.llm.call(
                model=self.llm.model_name,
                messages=messages,
            )
            return follow_up.choices[0].message.content

        # 6) Fallback: echo
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