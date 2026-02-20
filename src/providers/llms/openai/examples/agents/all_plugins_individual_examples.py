"""
Individual Examples for All 8 Built-in Plugins — NucleusIQ

Each plugin gets its own focused demo that highlights its specific behavior.

Run with: python all_plugins_individual_examples.py
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

SEPARATOR = "=" * 60


# ====================================================================
# Shared tools
# ====================================================================

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def search_contacts(query: str) -> str:
    """Search the company contact directory."""
    contacts = {
        "alice": "Alice Smith - alice@company.com - Phone: 555-123-4567 - SSN: 123-45-6789",
        "bob": "Bob Jones - bob@company.com - IP: 192.168.1.50",
        "nucleusiq": "NucleusIQ - AI agent framework by Brijesh - hello@nucleusiq.dev",
    }
    for key, val in contacts.items():
        if key in query.lower():
            return val
    return f"No contact found for: {query}"

def delete_file(path: str) -> str:
    """Delete a file from the system."""
    return f"Deleted: {path}"

def flaky_tool(query: str) -> str:
    """A tool that fails sometimes."""
    if not hasattr(flaky_tool, "_call_count"):
        flaky_tool._call_count = 0
    flaky_tool._call_count += 1
    if flaky_tool._call_count % 3 != 0:
        raise ConnectionError("Service temporarily unavailable")
    return f"Result for: {query}"


add_tool = BaseTool.from_function(add, name="add", description="Add two numbers")
mul_tool = BaseTool.from_function(multiply, name="multiply", description="Multiply two numbers")
search_tool = BaseTool.from_function(search_contacts, name="search_contacts", description="Search company contacts")
delete_tool = BaseTool.from_function(delete_file, name="delete_file", description="Delete a file")


# ====================================================================
# 1. ModelCallLimitPlugin
# ====================================================================

async def example_model_call_limit():
    print(f"\n{SEPARATOR}")
    print("  1. ModelCallLimitPlugin")
    print(f"  Caps LLM calls to prevent runaway costs.")
    print(f"  Config: max_calls=5")
    print(SEPARATOR)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="LimitBot",
        role="Assistant with cost control",
        objective="Answer questions within call budget",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[ModelCallLimitPlugin(max_calls=5)],
    )
    await agent.initialize()

    result = await agent.execute({"id": "1", "objective": "What is the capital of France? One word."})
    print(f"  Result: {result}")
    print("  (Agent limited to 5 LLM calls per execution)")


# ====================================================================
# 2. ToolCallLimitPlugin
# ====================================================================

async def example_tool_call_limit():
    print(f"\n{SEPARATOR}")
    print("  2. ToolCallLimitPlugin")
    print(f"  Caps tool calls to prevent infinite tool loops.")
    print(f"  Config: max_calls=5")
    print(SEPARATOR)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="ToolLimitBot",
        role="Math assistant with tool limits",
        objective="Calculate with bounded tool usage",
        llm=llm,
        tools=[add_tool, mul_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ToolCallLimitPlugin(max_calls=5),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent.initialize()

    result = await agent.execute({"id": "1", "objective": "What is 3 + 7?"})
    print(f"  Result: {result}")
    print("  (Agent limited to 5 tool calls per execution)")


# ====================================================================
# 3. ToolRetryPlugin
# ====================================================================

async def example_tool_retry():
    print(f"\n{SEPARATOR}")
    print("  3. ToolRetryPlugin")
    print(f"  Retries failed tool calls with exponential backoff.")
    print(f"  Config: max_retries=2, base_delay=0.5s")
    print(SEPARATOR)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="RetryBot",
        role="Resilient assistant",
        objective="Use tools with retry protection",
        llm=llm,
        tools=[add_tool, mul_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ToolRetryPlugin(max_retries=2, base_delay=0.5, max_delay=5.0),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent.initialize()

    result = await agent.execute({"id": "1", "objective": "What is 10 + 20?"})
    print(f"  Result: {result}")
    print("  (Failed tool calls would be retried up to 2 times with 0.5s, 1.0s delays)")


# ====================================================================
# 4. ModelFallbackPlugin
# ====================================================================

async def example_model_fallback():
    print(f"\n{SEPARATOR}")
    print("  4. ModelFallbackPlugin")
    print(f"  Automatically tries backup models when the primary fails.")
    print(f"  Config: fallbacks=['gpt-4o-mini']")
    print(SEPARATOR)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="FallbackBot",
        role="Resilient assistant with model fallback",
        objective="Answer questions with failover protection",
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

    result = await agent.execute({"id": "1", "objective": "Name three programming languages in one sentence."})
    print(f"  Result: {result}")
    print("  (If primary model fails, gpt-4o-mini is tried automatically)")


# ====================================================================
# 5. PIIGuardPlugin
# ====================================================================

async def example_pii_guard():
    print(f"\n{SEPARATOR}")
    print("  5. PIIGuardPlugin")
    print(f"  Detects and handles PII before the LLM sees it.")
    print(f"  Config: types=[email, phone, ssn], strategy=redact")
    print(SEPARATOR)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="PIIBot",
        role="Privacy-aware assistant",
        objective="Help users while protecting PII",
        llm=llm,
        tools=[search_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            PIIGuardPlugin(
                pii_types=["email", "phone", "ssn"],
                strategy="redact",
                apply_to_input=True,
            ),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent.initialize()

    print("\n  --- Redact mode ---")
    result = await agent.execute({
        "id": "1",
        "objective": "Look up Alice in the contacts and tell me what you found.",
    })
    print(f"  Result: {result}")
    print("  (Email, phone, SSN were redacted with [REDACTED_*] before LLM saw them)")

    print("\n  --- Mask mode ---")
    agent2 = Agent(
        name="MaskBot",
        role="Privacy-aware assistant",
        objective="Help users while masking PII",
        llm=llm,
        tools=[search_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            PIIGuardPlugin(
                pii_types=["email", "phone", "ssn"],
                strategy="mask",
            ),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent2.initialize()

    result2 = await agent2.execute({
        "id": "2",
        "objective": "Search for Alice and share what you found.",
    })
    print(f"  Result: {result2}")
    print("  (PII was partially masked: a***@..., ***-***-4567, ***-**-6789)")

    print("\n  --- Custom pattern (API keys) ---")
    agent3 = Agent(
        name="APIKeyBot",
        role="API key protection",
        objective="Detect custom PII patterns",
        llm=llm,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            PIIGuardPlugin(
                custom_patterns={"api_key": r"sk-[a-zA-Z0-9]{20,}"},
                strategy="redact",
            ),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent3.initialize()

    result3 = await agent3.execute({
        "id": "3",
        "objective": "My OpenAI key is sk-abc123def456ghi789jklmno. Is it valid?",
    })
    print(f"  Result: {result3}")
    print("  (API key was redacted with [REDACTED_API_KEY] before LLM saw it)")


# ====================================================================
# 6. HumanApprovalPlugin
# ====================================================================

async def example_human_approval():
    print(f"\n{SEPARATOR}")
    print("  6. HumanApprovalPlugin")
    print(f"  Gates tool execution behind a programmatic approval callback.")
    print(f"  Config: auto_approve=[add, multiply], deny delete_file")
    print(SEPARATOR)

    approval_log = []

    async def approval_policy(tool_name: str, tool_args: dict) -> bool:
        safe_tools = {"add", "multiply", "search_contacts"}
        approved = tool_name in safe_tools
        status = "APPROVED" if approved else "DENIED"
        approval_log.append(f"{status}: {tool_name}({tool_args})")
        return approved

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="ApprovalBot",
        role="Calculator with approval gates",
        objective="Do math with human oversight",
        llm=llm,
        tools=[add_tool, mul_tool, delete_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            HumanApprovalPlugin(
                approval_callback=approval_policy,
                auto_approve=["add", "multiply"],
            ),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent.initialize()

    result = await agent.execute({"id": "1", "objective": "What is 25 + 17?"})
    print(f"  Result: {result}")
    print(f"  Approval log: {approval_log}")
    print("  (Math tools auto-approved, delete_file would be denied)")


# ====================================================================
# 7. ContextWindowPlugin
# ====================================================================

async def example_context_window():
    print(f"\n{SEPARATOR}")
    print("  7. ContextWindowPlugin")
    print(f"  Trims messages to prevent context window overflow.")
    print(f"  Config: max_messages=20, keep_recent=5")
    print(SEPARATOR)

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)

    agent = Agent(
        name="WindowBot",
        role="Conversational assistant with context management",
        objective="Chat with managed context",
        llm=llm,
        memory=memory,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            ContextWindowPlugin(
                max_messages=20,
                keep_recent=5,
                placeholder="[Earlier conversation has been summarized]",
            ),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent.initialize()

    conversations = [
        "Hi, I'm Brijesh.",
        "I'm building NucleusIQ, an AI agent framework.",
        "It has 8 built-in plugins now.",
        "The plugins use a typed request model with .with_() for immutability.",
        "What's my name and what am I building?",
    ]

    for i, msg in enumerate(conversations, 1):
        result = await agent.execute({"id": str(i), "objective": msg})
        print(f"  Turn {i}: {str(result)[:100]}...")

    print("\n  (Messages beyond 20 would be trimmed, keeping system + 5 most recent)")


# ====================================================================
# 8. ToolGuardPlugin
# ====================================================================

async def example_tool_guard():
    print(f"\n{SEPARATOR}")
    print("  8. ToolGuardPlugin")
    print(f"  Whitelist or blacklist tool access by name.")
    print(SEPARATOR)

    llm = BaseOpenAI(model_name="gpt-4o-mini")

    print("\n  --- Blocklist mode (block delete_file) ---")
    agent1 = Agent(
        name="BlockBot",
        role="Safe assistant",
        objective="Use tools safely",
        llm=llm,
        tools=[add_tool, mul_tool, search_tool, delete_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ToolGuardPlugin(
                blocked=["delete_file"],
                on_deny=lambda name, args: f"SECURITY: '{name}' is forbidden by policy.",
            ),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent1.initialize()

    result1 = await agent1.execute({"id": "1", "objective": "What is 5 + 3?"})
    print(f"  Math result: {result1}")
    print("  (add/multiply allowed, delete_file would be blocked)")

    print("\n  --- Allowlist mode (only allow add, multiply) ---")
    agent2 = Agent(
        name="AllowBot",
        role="Math-only assistant",
        objective="Only do math",
        llm=llm,
        tools=[add_tool, mul_tool, search_tool, delete_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ToolGuardPlugin(allowed=["add", "multiply"]),
            ModelCallLimitPlugin(max_calls=10),
        ],
    )
    await agent2.initialize()

    result2 = await agent2.execute({"id": "2", "objective": "Calculate 6 * 7."})
    print(f"  Math result: {result2}")
    print("  (Only add and multiply are allowed, everything else blocked)")


# ====================================================================
# Main
# ====================================================================

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        return

    examples = [
        ("1. ModelCallLimitPlugin", example_model_call_limit),
        ("2. ToolCallLimitPlugin", example_tool_call_limit),
        ("3. ToolRetryPlugin", example_tool_retry),
        ("4. ModelFallbackPlugin", example_model_fallback),
        ("5. PIIGuardPlugin", example_pii_guard),
        ("6. HumanApprovalPlugin", example_human_approval),
        ("7. ContextWindowPlugin", example_context_window),
        ("8. ToolGuardPlugin", example_tool_guard),
    ]

    passed = 0
    failed = 0

    for name, fn in examples:
        try:
            await fn()
            print(f"\n  [PASS] {name}\n")
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(examples)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
