"""
Plugin + Memory Example — NucleusIQ

Demonstrates the new plugin API with typed requests and immutable overrides:

1. **Decorator plugins** — Simple ``@before_model`` / ``@wrap_tool_call`` hooks
   that receive ``ModelRequest`` / ``ToolRequest`` with ``.with_()`` support.

2. **Class-based plugins** — Multi-hook ``BasePlugin`` subclass for audit trails.

3. **Built-in plugins** — ``ModelCallLimitPlugin`` prevents runaway LLM usage.

4. **Memory** — Full-history and summary memory across conversation turns.

Run with: python plugin_memory_example.py
Requires OPENAI_API_KEY environment variable.
"""

import asyncio
import logging
import os
import sys
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.memory.factory import MemoryFactory, MemoryStrategy
from nucleusiq.plugins import (
    AgentContext,
    BasePlugin,
    ModelRequest,
    ToolRequest,
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_tool_call,
)
from nucleusiq.plugins.builtin import ModelCallLimitPlugin
from nucleusiq.tools import BaseTool
from nucleusiq_openai import BaseOpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Decorator-based plugins (simple one-off hooks)
# ============================================================================


@before_agent
async def log_task_start(ctx: AgentContext) -> None:
    """Logs every task. Returns None = no change to context."""
    task_obj = getattr(ctx.task, "objective", str(ctx.task))
    logger.info(
        f"[plugin:log_task_start] Agent '{ctx.agent_name}' received: {task_obj}"
    )


@after_agent
async def log_task_end(ctx: AgentContext, result: Any) -> Any:
    """Logs the result after the agent finishes."""
    preview = str(result)[:120]
    logger.info(f"[plugin:log_task_end] Agent '{ctx.agent_name}' result: {preview}")
    return result


@before_model
def log_model_call(request: ModelRequest) -> None:
    """Counts and logs each LLM call. Returns None = observe only."""
    msg_count = len(request.messages)
    logger.info(
        f"[plugin:log_model_call] LLM call #{request.call_count} "
        f"to {request.model} with {msg_count} messages"
    )


@after_model
def log_model_response(request: ModelRequest, response: Any) -> Any:
    """Logs the raw LLM response type."""
    logger.info(
        f"[plugin:log_model_response] LLM call #{request.call_count} "
        f"returned {type(response).__name__}"
    )
    return response


@wrap_tool_call
async def log_tool_execution(request: ToolRequest, handler):
    """Wraps every tool call to log before and after execution."""
    logger.info(
        f"[plugin:log_tool] Calling tool '{request.tool_name}' "
        f"with args {request.tool_args}"
    )
    result = await handler(request)
    logger.info(f"[plugin:log_tool] Tool '{request.tool_name}' returned: {result}")
    return result


# ============================================================================
# 2. Class-based plugin (multi-hook audit trail)
# ============================================================================


class AuditTrailPlugin(BasePlugin):
    """Records a structured audit trail of the entire execution."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    @property
    def name(self) -> str:
        return "audit_trail"

    async def before_agent(self, ctx: AgentContext) -> None:
        self.events.append(
            {
                "hook": "before_agent",
                "agent": ctx.agent_name,
                "task": getattr(ctx.task, "objective", str(ctx.task)),
            }
        )

    async def after_agent(self, ctx: AgentContext, result: Any) -> Any:
        self.events.append(
            {
                "hook": "after_agent",
                "agent": ctx.agent_name,
                "result_length": len(str(result)),
            }
        )
        return result

    async def before_model(self, request: ModelRequest) -> None:
        self.events.append(
            {
                "hook": "before_model",
                "model": request.model,
                "call_count": request.call_count,
                "message_count": len(request.messages),
            }
        )

    def print_trail(self) -> None:
        logger.info("\n--- Audit Trail ---")
        for i, event in enumerate(self.events, 1):
            logger.info(f"  {i}. {event}")
        logger.info("--- End Trail ---\n")


# ============================================================================
# Example 1: Conversational Memory + Decorator Plugins
# ============================================================================


async def example_memory_with_plugins():
    """Agent remembers conversation context; plugins log every step."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Memory + Decorator Plugins")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)

    agent = Agent(
        name="MemoryBot",
        role="Friendly Assistant",
        objective="Remember user details and answer questions",
        llm=llm,
        memory=memory,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            log_task_start,
            log_task_end,
            log_model_call,
            log_model_response,
        ],
    )
    await agent.initialize()

    result1 = await agent.execute(
        {
            "id": "1",
            "objective": "Hello! My name is Brijesh and I am the creator of NucleusIQ.",
        }
    )
    logger.info(f"Turn 1 response: {result1}\n")

    result2 = await agent.execute(
        {
            "id": "2",
            "objective": "Who is the creator of NucleusIQ?",
        }
    )
    logger.info(f"Turn 2 response: {result2}\n")

    assert "Brijesh" in str(result2), f"Expected 'Brijesh' in response, got: {result2}"
    logger.info("Memory verification passed: agent remembered 'Brijesh'")


# ============================================================================
# Example 2: Tool Agent + Audit Trail Plugin + Model Call Limit
# ============================================================================


async def example_tools_with_audit_and_limit():
    """Agent uses tools; audit trail records everything; limit prevents runaway."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Tools + Audit Trail + Model Call Limit")
    logger.info("=" * 60)

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    add_tool = BaseTool.from_function(add, name="add", description="Add two numbers")
    mul_tool = BaseTool.from_function(
        multiply, name="multiply", description="Multiply two numbers"
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.SLIDING_WINDOW, window_size=10)
    audit = AuditTrailPlugin()

    agent = Agent(
        name="MathBot",
        role="Calculator",
        objective="Perform math calculations",
        llm=llm,
        memory=memory,
        tools=[add_tool, mul_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            audit,
            log_tool_execution,
            ModelCallLimitPlugin(max_calls=15),
        ],
    )
    await agent.initialize()

    result = await agent.execute(
        {
            "id": "calc-1",
            "objective": "What is 25 + 17?",
        }
    )
    logger.info(f"Calculation result: {result}\n")
    audit.print_trail()


# ============================================================================
# Example 3: Summary Memory for Long Conversations + Plugins
# ============================================================================


async def example_summary_memory_with_plugins():
    """Uses summary memory to keep conversations compact while plugins monitor."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Summary Memory + Monitoring Plugins")
    logger.info("=" * 60)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(
        MemoryStrategy.SUMMARY_WINDOW,
        window_size=4,
        llm=llm,
        llm_model="gpt-4o-mini",
    )
    audit = AuditTrailPlugin()

    agent = Agent(
        name="SummaryBot",
        role="Knowledge Assistant",
        objective="Answer questions and remember context efficiently",
        llm=llm,
        memory=memory,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            audit,
            log_task_start,
            log_model_call,
        ],
    )
    await agent.initialize()

    conversations = [
        "My favorite programming language is Python.",
        "I work at a company called NucleusIQ.",
        "We are building an AI agent framework.",
        "The framework supports plugins, memory, and tools.",
        "What do you know about me and my work so far?",
    ]

    for i, message in enumerate(conversations, 1):
        result = await agent.execute(
            {
                "id": f"turn-{i}",
                "objective": message,
            }
        )
        logger.info(f"Turn {i}: {str(result)[:150]}\n")

    logger.info(
        f"Audit trail has {len(audit.events)} events across {len(conversations)} turns"
    )
    audit.print_trail()


# ============================================================================
# Main
# ============================================================================


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set! Please set it in your .env file.")
        return

    examples = [
        ("Memory + Decorator Plugins", example_memory_with_plugins),
        ("Tools + Audit Trail + Limit", example_tools_with_audit_and_limit),
        ("Summary Memory + Monitoring", example_summary_memory_with_plugins),
    ]

    for name, fn in examples:
        try:
            await fn()
            logger.info(f"[PASS] {name}\n")
        except Exception as e:
            logger.error(f"[FAIL] {name}: {e}\n", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
