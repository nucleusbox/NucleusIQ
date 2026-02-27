"""
Example: Streaming Agent Execution

Demonstrates ``agent.execute_stream()`` — the streaming counterpart
to ``agent.execute()``.  Events arrive as they are generated, enabling
real-time token-by-token display and tool-call visibility.

Run with:
    python src/examples/agents/streaming_example.py
"""

import asyncio
import logging
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.tools.base_tool import BaseTool

logging.basicConfig(level=logging.WARNING)


# ── Tool definition ──────────────────────────────────────────────────────────


def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


adder_tool = BaseTool.from_function(add, description="Add two integers.")


# ── Helper: consume a stream and print events ────────────────────────────────


async def run_streaming(agent: Agent, objective: str) -> None:
    """Execute a task via streaming and print events in real time."""
    print(f"\n{'=' * 60}")
    print(f"Task: {objective}")
    print(f"{'=' * 60}\n")

    async for event in agent.execute_stream({"id": "demo", "objective": objective}):
        match event.type:
            case "llm_start":
                print(f"  [LLM call #{event.call_count} started]")
            case "token":
                print(event.token, end="", flush=True)
            case "llm_end":
                print(f"\n  [LLM call #{event.call_count} ended]")
            case "tool_start":
                print(f"  >> Tool: {event.tool_name}({event.tool_args})")
            case "tool_end":
                print(f"  << Result: {event.tool_result}")
            case "thinking":
                print(f"  ... {event.message}")
            case "complete":
                print(f"\nFinal: {event.content}")
            case "error":
                print(f"\nERROR: {event.message}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("NucleusIQ Streaming Example")
    print("===========================\n")

    # 1. Simple text streaming (Direct mode)
    agent_direct = Agent(
        name="DirectBot",
        role="Assistant",
        objective="Help users",
        narrative="Fast direct-mode agent",
        llm=MockLLM(stream_chunk_size=5),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )
    await agent_direct.initialize()
    await run_streaming(agent_direct, "Tell me about NucleusIQ")

    # 2. Tool-calling stream (Standard mode)
    agent_standard = Agent(
        name="CalcBot",
        role="Calculator",
        objective="Perform calculations",
        narrative="Standard-mode agent with tools",
        llm=MockLLM(stream_chunk_size=3),
        tools=[adder_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
    )
    await agent_standard.initialize()
    await run_streaming(agent_standard, "Add 42 and 58")

    # 3. Autonomous mode (includes THINKING events)
    agent_auto = Agent(
        name="AutoBot",
        role="Analyst",
        objective="Analyze data",
        narrative="Autonomous-mode agent",
        llm=MockLLM(stream_chunk_size=4),
        config=AgentConfig(
            execution_mode=ExecutionMode.AUTONOMOUS,
            max_retries=1,
        ),
    )
    await agent_auto.initialize()
    await run_streaming(agent_auto, "What is the meaning of life?")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
