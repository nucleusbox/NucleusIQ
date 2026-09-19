"""Example: first-class WebSearchTool on an Agent.

Default backend is DuckDuckGo (no API key). The agent calls ``web_search``
through the normal Standard-mode tool loop — same wiring as file tools.

    python examples/agents/web_search_example.py

This run uses MockLLM so it does not hit the network. Swap in a real LLM
to search live. Restrict results with ``allowed_domains``.
"""

# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq.tools import WebSearchTool
from nucleusiq.tools.web_search import SearchHit, WebSearchBackend

logging.basicConfig(level=logging.WARNING)


class ScriptedSearchLLM(MockLLM):
    """First turn: call web_search. Second turn: answer from the tool result."""

    def _tool_call_response(self, messages, tools):  # type: ignore[no-untyped-def]
        fn_call = {
            "name": "web_search",
            "arguments": json.dumps({"query": "python typing"}),
        }
        return self.LLMResponse(
            [
                self.Choice(
                    self.Message(
                        content=None,
                        function_call=fn_call,
                        tool_calls=[
                            {
                                "id": "call_web_1",
                                "type": "function",
                                "function": fn_call,
                            }
                        ],
                    )
                )
            ]
        )


class DemoSearchBackend(WebSearchBackend):
    name = "demo"

    async def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        return [
            SearchHit(
                title="typing — Support for type hints",
                url="https://docs.python.org/3/library/typing.html",
                snippet=f"Official docs matching {query}",
            )
        ]


async def main() -> None:
    prompt = ZeroShotPrompt().configure(
        system=(
            "You are a research assistant. "
            "Use the web_search tool for current facts. "
            "Cite the URLs you used."
        ),
    )

    # Open web:
    #   tools=[WebSearchTool()]
    # Docs-only (whitelist):
    #   tools=[WebSearchTool(allowed_domains=["docs.python.org", "peps.python.org"])]
    # Paid engine:
    #   tools=[WebSearchTool(provider="brave", api_key=os.environ["BRAVE_API_KEY"])]

    agent = Agent(
        name="researcher",
        role="Researcher",
        objective="Answer questions with cited web results",
        llm=ScriptedSearchLLM(),
        prompt=prompt,
        tools=[
            WebSearchTool(
                backend=DemoSearchBackend(),
                allowed_domains=["docs.python.org"],
            )
        ],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )

    result = await agent.execute(
        Task(id="q1", objective="What is the Python typing module?")
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
