"""
Plugin API Showcase — NucleusIQ

Demonstrates all key patterns of the new plugin API:

1. Observe (return None)
2. Modify request via .with_()
3. Retry with model fallback
4. Block dangerous tools
5. Class-based multi-hook audit plugin
6. Combine everything together

Run with: python plugin_api_showcase.py
Requires OPENAI_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
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
from nucleusiq_openai import BaseOpenAI
from nucleusiq.memory.factory import MemoryFactory, MemoryStrategy
from nucleusiq.tools import BaseTool

from nucleusiq.plugins import (
    BasePlugin,
    AgentContext,
    ModelRequest,
    ToolRequest,
    before_agent,
    after_agent,
    before_model,
    after_model,
    wrap_model_call,
    wrap_tool_call,
)
from nucleusiq.plugins.builtin import ModelCallLimitPlugin, ToolRetryPlugin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Pattern 1: Observe (return None = just watching)
# ============================================================================

@before_model
def log_calls(request: ModelRequest) -> None:
    """Return None = I'm just watching, don't change anything."""
    logger.info(
        f"  [observe] LLM call #{request.call_count} "
        f"to {request.model} with {len(request.messages)} messages"
    )


# ============================================================================
# Pattern 2: Modify request via .with_()
# ============================================================================

@before_model
def use_cheap_model_after_3_calls(request: ModelRequest):
    """After 3 calls, downgrade to cheaper model to save cost."""
    if request.call_count > 3:
        logger.info(
            f"  [modify] Call #{request.call_count} > 3, "
            f"downgrading {request.model} -> gpt-4o-mini"
        )
        return request.with_(model="gpt-4o-mini")
    # return None implicitly — no change for first 3 calls


# ============================================================================
# Pattern 3: Retry with model fallback (wrap hook)
# ============================================================================

@wrap_model_call
async def retry_with_fallback(request: ModelRequest, handler):
    """Try the request. If it fails, retry with gpt-4o-mini."""
    try:
        return await handler(request)
    except Exception as e:
        logger.info(f"  [retry] Primary failed: {e}, falling back to gpt-4o-mini")
        return await handler(request.with_(model="gpt-4o-mini"))


# ============================================================================
# Pattern 4: Block dangerous tools
# ============================================================================

BLOCKED_TOOLS = {"delete_file", "drop_table", "rm_rf"}

@wrap_tool_call
async def guard_tools(request: ToolRequest, handler):
    """Block dangerous tools, let safe ones through."""
    if request.tool_name in BLOCKED_TOOLS:
        logger.info(f"  [guard] BLOCKED tool '{request.tool_name}'")
        return f"Tool '{request.tool_name}' is blocked by security policy"
    logger.info(f"  [guard] ALLOWED tool '{request.tool_name}'")
    return await handler(request)


# ============================================================================
# Pattern 5: Class-based multi-hook audit plugin
# ============================================================================

class AuditPlugin(BasePlugin):
    """No need to define name — defaults to 'AuditPlugin' automatically."""

    def __init__(self):
        self.log: list[str] = []

    async def before_agent(self, ctx: AgentContext) -> None:
        task_text = getattr(ctx.task, "objective", str(ctx.task))
        self.log.append(f"START  agent={ctx.agent_name} task={task_text[:50]}")

    async def before_model(self, request: ModelRequest) -> None:
        self.log.append(
            f"MODEL  #{request.call_count} model={request.model} "
            f"msgs={len(request.messages)}"
        )

    async def wrap_tool_call(self, request: ToolRequest, handler):
        self.log.append(f"TOOL   {request.tool_name}({request.tool_args})")
        result = await handler(request)
        self.log.append(f"RESULT {request.tool_name} -> {result}")
        return result

    async def after_agent(self, ctx: AgentContext, result: Any) -> Any:
        self.log.append(f"END    agent={ctx.agent_name} len={len(str(result))}")
        return result

    def print_trail(self):
        print("\n  === Audit Trail ===")
        for i, entry in enumerate(self.log, 1):
            print(f"    {i:2d}. {entry}")
        print("  === End Trail ===\n")


# ============================================================================
# Demo 1: Observe + Modify with conversation memory
# ============================================================================

async def demo_observe_and_modify():
    print("\n" + "=" * 70)
    print("  Demo 1: Observe + Modify (return None vs .with_())")
    print("=" * 70)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)

    agent = Agent(
        name="ChatBot",
        role="Helpful assistant",
        objective="Chat and remember context",
        llm=llm,
        memory=memory,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            log_calls,                        # Pattern 1: observe (returns None)
            use_cheap_model_after_3_calls,    # Pattern 2: modify via .with_()
        ],
    )
    await agent.initialize()

    print("\n  Turn 1: Introduce myself")
    r1 = await agent.execute({"id": "1", "objective": "Hi, I'm Brijesh. I created NucleusIQ."})
    print(f"  Response: {r1}\n")

    print("  Turn 2: Ask recall question")
    r2 = await agent.execute({"id": "2", "objective": "Who created NucleusIQ?"})
    print(f"  Response: {r2}\n")

    assert "Brijesh" in str(r2), f"Memory failed: {r2}"
    print("  PASS: Agent remembered 'Brijesh' across turns")


# ============================================================================
# Demo 2: Wrap hooks — retry, guard, audit
# ============================================================================

async def demo_wrap_hooks():
    print("\n" + "=" * 70)
    print("  Demo 2: Wrap Hooks (retry, guard, audit)")
    print("=" * 70)

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    add_tool = BaseTool.from_function(add, name="add", description="Add two numbers")
    mul_tool = BaseTool.from_function(multiply, name="multiply", description="Multiply two numbers")

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    audit = AuditPlugin()

    agent = Agent(
        name="MathBot",
        role="Calculator",
        objective="Do math",
        llm=llm,
        tools=[add_tool, mul_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            log_calls,                            # Pattern 1: observe
            retry_with_fallback,                  # Pattern 3: retry
            guard_tools,                          # Pattern 4: guard
            audit,                                # Pattern 5: class-based audit
            ModelCallLimitPlugin(max_calls=10),   # Built-in: safety limit
        ],
    )
    await agent.initialize()

    print("\n  Asking: What is (12 + 8) * 3?")
    result = await agent.execute({"id": "math-1", "objective": "What is (12 + 8) * 3?"})
    print(f"\n  Result: {result}")

    audit.print_trail()


# ============================================================================
# Demo 3: Everything combined — memory + plugins + tools
# ============================================================================

async def demo_full_stack():
    print("\n" + "=" * 70)
    print("  Demo 3: Full Stack (memory + plugins + tools)")
    print("=" * 70)

    def search(query: str) -> str:
        """Search for information."""
        data = {
            "nucleusiq": "NucleusIQ is an AI agent framework built by Brijesh.",
            "python": "Python is a versatile programming language.",
        }
        for key, val in data.items():
            if key in query.lower():
                return val
        return f"No results for: {query}"

    search_tool = BaseTool.from_function(search, name="search", description="Search for information")

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.SLIDING_WINDOW, window_size=8)
    audit = AuditPlugin()

    agent = Agent(
        name="FullBot",
        role="Research assistant",
        objective="Search and remember",
        llm=llm,
        memory=memory,
        tools=[search_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            log_calls,
            guard_tools,
            audit,
            ModelCallLimitPlugin(max_calls=20),
            ToolRetryPlugin(max_retries=2, base_delay=0.5),
        ],
    )
    await agent.initialize()

    print("\n  Turn 1: Search for NucleusIQ")
    r1 = await agent.execute({"id": "1", "objective": "Search for NucleusIQ and tell me about it."})
    print(f"  Response: {str(r1)[:200]}\n")

    print("  Turn 2: Recall from memory")
    r2 = await agent.execute({"id": "2", "objective": "Based on what you found earlier, who built NucleusIQ?"})
    print(f"  Response: {str(r2)[:200]}\n")

    audit.print_trail()


# ============================================================================
# Main
# ============================================================================

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        return

    demos = [
        ("Observe + Modify", demo_observe_and_modify),
        ("Wrap Hooks", demo_wrap_hooks),
        ("Full Stack", demo_full_stack),
    ]

    for name, fn in demos:
        try:
            await fn()
            print(f"  [PASS] {name}\n")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
