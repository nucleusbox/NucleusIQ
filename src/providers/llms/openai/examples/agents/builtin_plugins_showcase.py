"""
Built-in Plugins Showcase — NucleusIQ

Demonstrates all 8 built-in plugins working with real agents:

1. ModelCallLimitPlugin   — cap LLM calls to prevent runaway costs
2. ToolCallLimitPlugin    — cap tool calls
3. ToolRetryPlugin        — retry failed tools with exponential backoff
4. ModelFallbackPlugin    — try backup models on failure
5. PIIGuardPlugin         — redact emails, SSNs, phones before the LLM sees them
6. HumanApprovalPlugin    — programmatic approval gate for tool calls
7. ContextWindowPlugin    — trim messages when context gets too long
8. ToolGuardPlugin        — whitelist/blacklist tool access

Run with: python builtin_plugins_showcase.py
Requires OPENAI_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging

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

from nucleusiq.plugins.builtin import (
    ModelCallLimitPlugin,
    ToolCallLimitPlugin,
    ToolRetryPlugin,
    ModelFallbackPlugin,
    PIIGuardPlugin,
    HumanApprovalPlugin,
    ContextWindowPlugin,
    ToolGuardPlugin,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==================================================================== #
# Helper tools                                                          #
# ==================================================================== #

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def search_contacts(query: str) -> str:
    """Search the company contact directory."""
    contacts = {
        "alice": "Alice Smith — alice@company.com — SSN: 123-45-6789 — Phone: 555-123-4567",
        "bob": "Bob Jones — bob@company.com — IP: 192.168.1.50",
        "nucleusiq": "NucleusIQ — Built by Brijesh — contact: hello@nucleusiq.dev",
    }
    for key, val in contacts.items():
        if key in query.lower():
            return val
    return f"No contact found for: {query}"

def delete_database(table: str) -> str:
    """Delete a database table (dangerous!)."""
    return f"Deleted table: {table}"


add_tool = BaseTool.from_function(add, name="add", description="Add two numbers")
mul_tool = BaseTool.from_function(multiply, name="multiply", description="Multiply two numbers")
search_tool = BaseTool.from_function(search_contacts, name="search_contacts", description="Search company contacts")
delete_tool = BaseTool.from_function(delete_database, name="delete_database", description="Delete a database table")


# ==================================================================== #
# Demo 1: PII Guard + Tool Guard                                        #
# ==================================================================== #

async def demo_pii_and_tool_guard():
    print("\n" + "=" * 70)
    print("  Demo 1: PIIGuardPlugin + ToolGuardPlugin")
    print("  - PII (emails, SSN, phone) redacted before LLM sees them")
    print("  - delete_database tool is blocked")
    print("=" * 70)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    agent = Agent(
        name="SecureBot",
        role="Secure assistant",
        objective="Search contacts safely",
        llm=llm,
        tools=[search_tool, delete_tool, add_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            PIIGuardPlugin(
                pii_types=["email", "ssn", "phone"],
                strategy="redact",
            ),
            ToolGuardPlugin(
                blocked=["delete_database"],
                on_deny=lambda name, args: f"SECURITY: Tool '{name}' is forbidden.",
            ),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent.initialize()

    print("\n  Query: 'Look up Alice in our contacts'")
    result = await agent.execute({
        "id": "1",
        "objective": "Look up Alice in our contacts and tell me about her.",
    })
    print(f"\n  Response: {result}")
    print("\n  (PII in the tool result was redacted before the LLM saw it)")


# ==================================================================== #
# Demo 2: Model Fallback                                                #
# ==================================================================== #

async def demo_model_fallback():
    print("\n" + "=" * 70)
    print("  Demo 2: ModelFallbackPlugin")
    print("  - Primary model: gpt-4o-mini (works fine)")
    print("  - Fallbacks configured: [gpt-4o-mini] (ready if needed)")
    print("=" * 70)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    agent = Agent(
        name="FallbackBot",
        role="Resilient assistant",
        objective="Answer questions with fallback protection",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            ModelFallbackPlugin(
                fallbacks=["gpt-4o-mini"],
                retry_on=(Exception,),
            ),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent.initialize()

    print("\n  Query: 'What is 2+2? Answer in one word.'")
    result = await agent.execute({
        "id": "1",
        "objective": "What is 2+2? Answer in one word.",
    })
    print(f"  Response: {result}")
    print("\n  (ModelFallbackPlugin is silently protecting against model failures)")


# ==================================================================== #
# Demo 3: Human Approval (programmatic)                                 #
# ==================================================================== #

async def demo_human_approval():
    print("\n" + "=" * 70)
    print("  Demo 3: HumanApprovalPlugin (programmatic callback)")
    print("  - 'add' and 'multiply' auto-approved")
    print("  - 'delete_database' requires approval -> denied by callback")
    print("=" * 70)

    approval_log = []

    async def policy_callback(tool_name: str, tool_args: dict) -> bool:
        approved = tool_name in ("add", "multiply", "search_contacts")
        approval_log.append(f"{'APPROVED' if approved else 'DENIED'}: {tool_name}")
        return approved

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    agent = Agent(
        name="ApprovalBot",
        role="Calculator with approval gates",
        objective="Do math with approval",
        llm=llm,
        tools=[add_tool, mul_tool, delete_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            HumanApprovalPlugin(
                approval_callback=policy_callback,
                auto_approve=["add", "multiply"],
            ),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent.initialize()

    print("\n  Query: 'What is 15 + 25?'")
    result = await agent.execute({
        "id": "1",
        "objective": "What is 15 + 25?",
    })
    print(f"  Response: {result}")
    print(f"  Approval log: {approval_log}")


# ==================================================================== #
# Demo 4: Context Window + Memory                                       #
# ==================================================================== #

async def demo_context_window():
    print("\n" + "=" * 70)
    print("  Demo 4: ContextWindowPlugin + Memory")
    print("  - Memory stores all messages")
    print("  - ContextWindowPlugin trims to 20 messages max before each LLM call")
    print("=" * 70)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)

    agent = Agent(
        name="ContextBot",
        role="Conversational assistant with context management",
        objective="Chat with managed context window",
        llm=llm,
        memory=memory,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            ContextWindowPlugin(
                max_messages=20,
                keep_recent=8,
                placeholder="[Earlier conversation trimmed]",
            ),
            ModelCallLimitPlugin(max_calls=15),
        ],
    )
    await agent.initialize()

    print("\n  Sending 3 conversation turns...")
    r1 = await agent.execute({"id": "1", "objective": "My name is Brijesh and I'm building NucleusIQ."})
    print(f"  Turn 1: {str(r1)[:120]}...")

    r2 = await agent.execute({"id": "2", "objective": "NucleusIQ has a plugin system with 8 built-in plugins."})
    print(f"  Turn 2: {str(r2)[:120]}...")

    r3 = await agent.execute({"id": "3", "objective": "What's my name and what am I building?"})
    print(f"  Turn 3: {str(r3)[:120]}...")
    print("\n  (Context window keeps messages under control for long conversations)")


# ==================================================================== #
# Demo 5: All plugins combined                                          #
# ==================================================================== #

async def demo_all_combined():
    print("\n" + "=" * 70)
    print("  Demo 5: All 8 Plugins Combined")
    print("  - ModelCallLimit(15) + ToolCallLimit(20)")
    print("  - ToolRetry(2 retries) + ModelFallback([gpt-4o-mini])")
    print("  - PIIGuard(email,phone,ssn -> mask)")
    print("  - HumanApproval(auto-approve math tools)")
    print("  - ContextWindow(max 30 messages)")
    print("  - ToolGuard(block delete_database)")
    print("=" * 70)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.SLIDING_WINDOW, window_size=10)

    agent = Agent(
        name="FullStackBot",
        role="Fully protected assistant",
        objective="Search contacts and do math with all protections",
        llm=llm,
        memory=memory,
        tools=[add_tool, mul_tool, search_tool, delete_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ModelCallLimitPlugin(max_calls=15),
            ToolCallLimitPlugin(max_calls=20),
            ToolRetryPlugin(max_retries=2, base_delay=0.1),
            ModelFallbackPlugin(fallbacks=["gpt-4o-mini"]),
            PIIGuardPlugin(
                pii_types=["email", "phone", "ssn"],
                strategy="mask",
            ),
            HumanApprovalPlugin(
                approval_callback=lambda n, a: True,
                auto_approve=["add", "multiply", "search_contacts"],
            ),
            ContextWindowPlugin(max_messages=30, keep_recent=10),
            ToolGuardPlugin(blocked=["delete_database"]),
        ],
    )
    await agent.initialize()

    print("\n  Turn 1: Math")
    r1 = await agent.execute({"id": "1", "objective": "What is (7 + 3) * 5?"})
    print(f"  Result: {r1}\n")

    print("  Turn 2: Contact search (PII will be masked)")
    r2 = await agent.execute({"id": "2", "objective": "Look up Bob in contacts."})
    print(f"  Result: {r2}\n")

    print("  Turn 3: Recall from memory")
    r3 = await agent.execute({"id": "3", "objective": "What was the math result from earlier?"})
    print(f"  Result: {r3}\n")


# ==================================================================== #
# Main                                                                   #
# ==================================================================== #

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        return

    demos = [
        ("PII Guard + Tool Guard", demo_pii_and_tool_guard),
        ("Model Fallback", demo_model_fallback),
        ("Human Approval", demo_human_approval),
        ("Context Window + Memory", demo_context_window),
        ("All 8 Plugins Combined", demo_all_combined),
    ]

    for name, fn in demos:
        try:
            await fn()
            print(f"\n  [PASS] {name}\n")
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
