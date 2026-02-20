"""
NucleusIQ Plugin System -- Comprehensive Demo
==============================================

Demonstrates ALL 8 built-in plugins across 11 scenarios:

PART A - Individual Plugin Demos (simple, focused):
  1. ModelCallLimitPlugin   [before_model]       - Caps LLM calls
  2. ToolCallLimitPlugin    [wrap_tool_call]      - Caps tool calls
  3. ToolRetryPlugin        [wrap_tool_call]      - Retries flaky tools
  4. ModelFallbackPlugin    [wrap_model_call]     - Falls back to another model
  5. PIIGuardPlugin         [before_model+after]  - Redacts PII
  6. HumanApprovalPlugin    [wrap_tool_call]      - Gates tool execution
  7. ContextWindowPlugin    [before_model]        - Trims old messages
  8. ToolGuardPlugin        [wrap_tool_call]      - Blocks dangerous tools

PART B - Complex Multi-Plugin Scenarios:
  9. Secure Agent     = PIIGuard + HumanApproval + ToolGuard
  10. Resilient Agent = ModelFallback + ModelCallLimit + ContextWindow + ToolRetry
  11. Fortress Agent  = ALL 8 plugins together

Run:  python plugin_comprehensive_demo.py
Requires: OPENAI_API_KEY environment variable
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

from nucleusiq.plugins.builtin import (
    ModelCallLimitPlugin,
    ToolCallLimitPlugin,
    ToolRetryPlugin,
    ModelFallbackPlugin,
    PIIGuardPlugin,
    HumanApprovalPlugin,
    ApprovalHandler,
    PolicyApprovalHandler,
    ContextWindowPlugin,
    ToolGuardPlugin,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("demo")

SEP = "=" * 65
SUBSEP = "-" * 65


# ====================================================================
# TOOL DEFINITIONS (all using BaseTool.from_function)
# ====================================================================

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

def search_contacts(query: str) -> str:
    """Search the company contact directory for a person."""
    contacts = {
        "alice": "Alice Smith - alice@company.com - Phone: 555-123-4567 - SSN: 123-45-6789",
        "bob": "Bob Jones - bob@company.com - IP: 192.168.1.50",
        "charlie": "Charlie Davis - charlie@corp.io - Credit Card: 4532015112830366",
    }
    for key, val in contacts.items():
        if key in query.lower():
            return val
    return f"No contact found for: {query}"

def delete_file(path: str) -> str:
    """Permanently delete a file from the filesystem."""
    return f"DELETED: {path}"

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'"

def deploy_to_production(service: str, version: str) -> str:
    """Deploy a service to production environment."""
    return f"Deployed {service} v{version} to PRODUCTION"

def restart_server(server_name: str) -> str:
    """Restart a server by name."""
    return f"Server '{server_name}' restarted successfully"

_flaky_counter = {"n": 0}
def flaky_api_call(query: str) -> str:
    """Call an external API that is unreliable."""
    _flaky_counter["n"] += 1
    if _flaky_counter["n"] % 3 != 0:
        raise ConnectionError(f"API timeout on attempt #{_flaky_counter['n']}")
    return f"API result for '{query}': 42 records found"

def read_file(path: str) -> str:
    """Read contents of a file."""
    return f"Contents of {path}: [sample data]"

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22C, sunny"


add_tool = BaseTool.from_function(add)
mul_tool = BaseTool.from_function(multiply)
search_tool = BaseTool.from_function(search_contacts)
delete_tool = BaseTool.from_function(delete_file)
email_tool = BaseTool.from_function(send_email)
deploy_tool = BaseTool.from_function(deploy_to_production)
restart_tool = BaseTool.from_function(restart_server)
flaky_tool = BaseTool.from_function(flaky_api_call)
read_tool = BaseTool.from_function(read_file)
weather_tool = BaseTool.from_function(get_weather)


def header(num, title, hook, description):
    print(f"\n{SEP}")
    print(f"  {num}. {title}")
    print(f"  Hook: {hook}")
    print(f"  {description}")
    print(SEP)


# ====================================================================
# PART A: INDIVIDUAL PLUGIN DEMOS
# ====================================================================


async def demo_1_model_call_limit():
    """ModelCallLimitPlugin [before_model] -- caps total LLM calls."""
    header(
        1, "ModelCallLimitPlugin", "before_model",
        "Prevents runaway costs by capping LLM calls to max_calls=3",
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="LimitBot",
        role="A math assistant",
        objective="Help with calculations",
        llm=llm,
        tools=[add_tool, mul_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[ModelCallLimitPlugin(max_calls=3)],
    )
    await agent.initialize()

    result = await agent.execute({
        "id": "1",
        "objective": "What is 10 + 20? Then what is 5 * 6?"
    })
    print(f"\n  Result: {str(result)[:150]}")
    print(f"  (Agent was limited to 3 LLM calls max)")


async def demo_2_tool_call_limit():
    """ToolCallLimitPlugin [wrap_tool_call] -- caps total tool calls."""
    header(
        2, "ToolCallLimitPlugin", "wrap_tool_call",
        "Limits tool executions to max_calls=2. Extra calls are blocked.",
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="ToolLimitBot",
        role="A math assistant",
        objective="Help with calculations",
        llm=llm,
        tools=[add_tool, mul_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ToolCallLimitPlugin(max_calls=2),
            ModelCallLimitPlugin(max_calls=8),
        ],
    )
    await agent.initialize()

    result = await agent.execute({
        "id": "2",
        "objective": "Calculate: 10+20, then 5*6, then 100+200"
    })
    print(f"\n  Result: {str(result)[:200]}")
    print(f"  (Only first 2 tool calls executed, 3rd was blocked)")


async def demo_3_tool_retry():
    """ToolRetryPlugin [wrap_tool_call] -- retries failed tool calls."""
    header(
        3, "ToolRetryPlugin", "wrap_tool_call",
        "Retries flaky tools with exponential backoff. max_retries=3, base_delay=0.1s",
    )
    _flaky_counter["n"] = 0

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="RetryBot",
        role="An API assistant",
        objective="Fetch data from APIs",
        llm=llm,
        tools=[flaky_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ToolRetryPlugin(max_retries=3, base_delay=0.1),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent.initialize()

    result = await agent.execute({
        "id": "3",
        "objective": "Search the API for 'python frameworks'"
    })
    print(f"\n  Result: {str(result)[:200]}")
    print(f"  Flaky tool was called {_flaky_counter['n']} times (retried on failures)")


async def demo_4_model_fallback():
    """ModelFallbackPlugin [wrap_model_call] -- falls back to another model on failure."""
    header(
        4, "ModelFallbackPlugin", "wrap_model_call",
        "If primary model fails, automatically tries fallback models.",
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="FallbackBot",
        role="A reliable assistant",
        objective="Always respond, even if primary model has issues",
        llm=llm,
        tools=[add_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ModelFallbackPlugin(fallbacks=["gpt-4o-mini", "gpt-3.5-turbo"]),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent.initialize()

    result = await agent.execute({
        "id": "4",
        "objective": "What is 7 + 13?"
    })
    print(f"\n  Result: {str(result)[:150]}")
    print(f"  (If gpt-4o-mini failed, it would try gpt-3.5-turbo next)")


async def demo_5_pii_guard():
    """PIIGuardPlugin [before_model + after_model] -- redacts PII from LLM traffic."""
    header(
        5, "PIIGuardPlugin", "before_model + after_model",
        "Detects and redacts emails, phones, SSNs, credit cards, IPs.",
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="PIIBot",
        role="A contact lookup assistant",
        objective="Search contacts safely",
        llm=llm,
        tools=[search_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            PIIGuardPlugin(
                pii_types=["email", "phone", "ssn", "credit_card", "ip_address"],
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
            ),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent.initialize()

    result = await agent.execute({
        "id": "5",
        "objective": "Look up Alice's contact information"
    })
    print(f"\n  Result: {str(result)[:250]}")
    print(f"  (PII like emails, SSNs, phones should be redacted in the response)")

    print(f"\n  {SUBSEP}")
    print(f"  PIIGuardPlugin strategies:")
    print(f"    'redact' -> alice@company.com becomes [EMAIL_REDACTED]")
    print(f"    'mask'   -> alice@company.com becomes a****@c******.com")
    print(f"    'block'  -> raises PluginHalt if PII detected")


async def demo_6_human_approval():
    """HumanApprovalPlugin [wrap_tool_call] -- gates tool execution behind approval.

    Shows TWO ways to configure approval:
      6A: Simple callback function (quick, one-off rules)
      6B: PolicyApprovalHandler class (structured, with audit log)
    """
    header(
        6, "HumanApprovalPlugin", "wrap_tool_call",
        "Two ways: callback function OR ApprovalHandler class",
    )

    # ------------------------------------------------------------------
    # 6A: Simple callback function
    # ------------------------------------------------------------------
    print(f"\n  {SUBSEP}")
    print(f"  6A: Using approval_callback (simple function)")
    print(f"  {SUBSEP}")

    def approval_policy(tool_name: str, tool_args: dict) -> bool:
        """Quick rule: math is safe, everything else denied."""
        return tool_name in {"add", "multiply"}

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent_a = Agent(
        name="CallbackBot",
        role="An assistant with callback-based approval",
        objective="Help with tasks",
        llm=llm,
        tools=[add_tool, delete_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            HumanApprovalPlugin(
                approval_callback=approval_policy,
                auto_approve=["add"],
                require_approval=["delete_file"],
            ),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent_a.initialize()

    print(f"\n  Config: approval_callback=approval_policy")
    print(f"          auto_approve=['add'], require_approval=['delete_file']")

    result = await agent_a.execute({"id": "6a1", "objective": "What is 25 + 17?"})
    print(f"\n  Task: 'What is 25+17?' -> {str(result)[:100]}")

    result = await agent_a.execute({"id": "6a2", "objective": "Delete /tmp/secrets.txt"})
    print(f"  Task: 'Delete file'     -> {str(result)[:150]}")

    # ------------------------------------------------------------------
    # 6B: PolicyApprovalHandler (class-based with audit log)
    # ------------------------------------------------------------------
    print(f"\n  {SUBSEP}")
    print(f"  6B: Using PolicyApprovalHandler (class with audit log)")
    print(f"  {SUBSEP}")

    policy_handler = PolicyApprovalHandler(
        safe_tools=["add", "multiply", "get_weather"],
        dangerous_tools=["delete_file", "deploy_to_production"],
        default_allow=False,
    )

    agent_b = Agent(
        name="PolicyBot",
        role="An assistant with policy-based approval and audit trail",
        objective="Help with tasks under strict policy",
        llm=llm,
        tools=[add_tool, delete_tool, deploy_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            HumanApprovalPlugin(approval_handler=policy_handler),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent_b.initialize()

    print(f"\n  Config: approval_handler=PolicyApprovalHandler(")
    print(f"            safe_tools=['add', 'multiply', 'get_weather'],")
    print(f"            dangerous_tools=['delete_file', 'deploy_to_production'],")
    print(f"          )")

    result = await agent_b.execute({"id": "6b1", "objective": "What is 50 + 50?"})
    print(f"\n  Task: 'What is 50+50?' -> {str(result)[:100]}")

    result = await agent_b.execute({"id": "6b2", "objective": "Delete /etc/passwd"})
    print(f"  Task: 'Delete file'     -> {str(result)[:150]}")

    print(f"\n  Audit log from PolicyApprovalHandler:")
    for entry in policy_handler.audit_log:
        status = "APPROVED" if entry["approved"] else "DENIED"
        print(f"    [{status}] {entry['tool']} (reason: {entry['reason']}, time: {entry['timestamp'][:19]})")


async def demo_7_context_window():
    """ContextWindowPlugin [before_model] -- trims messages to fit context window."""
    header(
        7, "ContextWindowPlugin", "before_model",
        "Trims older messages to prevent context overflow. max_messages=15, keep_recent=5",
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)

    agent = Agent(
        name="WindowBot",
        role="A conversational assistant with managed context window",
        objective="Chat while managing memory",
        llm=llm,
        memory=memory,
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
        plugins=[
            ContextWindowPlugin(
                max_messages=15,
                keep_recent=5,
                placeholder="[Earlier messages trimmed for context window]",
            ),
            ModelCallLimitPlugin(max_calls=8),
        ],
    )
    await agent.initialize()

    messages = [
        "My name is Brijesh and I'm building NucleusIQ.",
        "NucleusIQ is an AI agent framework with a plugin system.",
        "We have 8 built-in plugins covering model and tool hooks.",
        "The plugin system uses typed requests with immutable overrides.",
        "What have we been discussing? Summarize our conversation.",
    ]

    for i, msg in enumerate(messages, 1):
        print(f"\n  Turn {i}: {msg[:60]}...")
        result = await agent.execute({"id": f"7-{i}", "objective": msg})
        print(f"  Response: {str(result)[:120]}...")

    print(f"\n  (ContextWindowPlugin kept the conversation within 15 messages)")
    print(f"  (Oldest messages get replaced with placeholder when limit hit)")


async def demo_8_tool_guard():
    """ToolGuardPlugin [wrap_tool_call] -- blocks/allows tools by name."""
    header(
        8, "ToolGuardPlugin", "wrap_tool_call",
        "Whitelist/blacklist tools. blocked=['delete_file','restart_server']",
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="GuardBot",
        role="An assistant with restricted tool access",
        objective="Help users but block dangerous operations",
        llm=llm,
        tools=[add_tool, read_tool, delete_tool, restart_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ToolGuardPlugin(
                blocked=["delete_file", "restart_server"],
                on_deny="ACCESS DENIED: This tool is blocked by security policy.",
            ),
            ModelCallLimitPlugin(max_calls=5),
        ],
    )
    await agent.initialize()

    print(f"\n  Config:")
    print(f"    blocked = ['delete_file', 'restart_server']")
    print(f"    allowed = everything else (add, read_file)")

    print(f"\n  --- Task A: Allowed tool (add) ---")
    result = await agent.execute({"id": "8a", "objective": "What is 100 + 200?"})
    print(f"  Result: {str(result)[:150]}")

    print(f"\n  --- Task B: Blocked tool (delete_file) ---")
    result = await agent.execute({"id": "8b", "objective": "Delete /var/log/syslog"})
    print(f"  Result: {str(result)[:200]}")
    print(f"  (delete_file was blocked, agent got deny message)")


# ====================================================================
# PART B: COMPLEX MULTI-PLUGIN SCENARIOS
# ====================================================================


async def demo_9_secure_agent():
    """COMPLEX: Secure Agent = PIIGuard + HumanApproval + ToolGuard."""
    header(
        9, "Secure Agent (3 plugins)", "wrap_tool_call + before_model",
        "PIIGuard + HumanApproval + ToolGuard = defense in depth",
    )

    security_handler = PolicyApprovalHandler(
        safe_tools=["add", "multiply", "search_contacts", "get_weather"],
        dangerous_tools=["delete_file", "send_email"],
        default_allow=False,
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    agent = Agent(
        name="SecureBot",
        role="A secure assistant with layered protections",
        objective="Help safely with multiple security layers",
        llm=llm,
        tools=[add_tool, search_tool, delete_tool, email_tool, weather_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            PIIGuardPlugin(
                pii_types=["email", "phone", "ssn", "credit_card", "ip_address"],
                strategy="redact",
                apply_to_output=True,
            ),
            HumanApprovalPlugin(
                approval_handler=security_handler,
                auto_approve=["add", "get_weather"],
            ),
            ToolGuardPlugin(blocked=["restart_server"]),
            ModelCallLimitPlugin(max_calls=8),
        ],
    )
    await agent.initialize()

    print(f"\n  Security layers:")
    print(f"    Layer 1 - PIIGuard:        Redacts PII from all LLM traffic")
    print(f"    Layer 2 - PolicyApproval:  safe=['add','get_weather'], dangerous=['delete_file','send_email']")
    print(f"    Layer 3 - ToolGuard:       Blocks restart_server entirely")

    print(f"\n  --- Scenario: Look up contact (PII gets redacted) ---")
    result = await agent.execute({
        "id": "9a",
        "objective": "Find Alice's contact info"
    })
    print(f"  Result: {str(result)[:200]}")

    print(f"\n  --- Scenario: Try to delete (PolicyApprovalHandler denies) ---")
    result = await agent.execute({
        "id": "9b",
        "objective": "Delete /tmp/data.csv"
    })
    print(f"  Result: {str(result)[:200]}")

    print(f"\n  --- Scenario: Safe math (auto-approved, handler not called) ---")
    result = await agent.execute({
        "id": "9c",
        "objective": "What is 42 + 58?"
    })
    print(f"  Result: {str(result)[:120]}")

    print(f"\n  Audit trail from PolicyApprovalHandler:")
    for entry in security_handler.audit_log:
        status = "APPROVED" if entry["approved"] else "DENIED"
        print(f"    [{status}] {entry['tool']} -> {entry['reason']}")


async def demo_10_resilient_agent():
    """COMPLEX: Resilient Agent = ModelFallback + ModelCallLimit + ContextWindow + ToolRetry."""
    header(
        10, "Resilient Agent (4 plugins)", "wrap_model_call + before_model + wrap_tool_call",
        "ModelFallback + ModelCallLimit + ContextWindow + ToolRetry",
    )
    _flaky_counter["n"] = 0

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)

    agent = Agent(
        name="ResilientBot",
        role="A fault-tolerant assistant that handles failures gracefully",
        objective="Complete tasks despite API failures",
        llm=llm,
        memory=memory,
        tools=[add_tool, flaky_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            ModelFallbackPlugin(fallbacks=["gpt-4o-mini"]),
            ModelCallLimitPlugin(max_calls=10),
            ContextWindowPlugin(max_messages=20, keep_recent=8),
            ToolRetryPlugin(max_retries=3, base_delay=0.1),
        ],
    )
    await agent.initialize()

    print(f"\n  Resilience layers:")
    print(f"    ModelFallback:  If primary model fails -> try gpt-4o-mini")
    print(f"    ModelCallLimit: Max 10 LLM calls per task")
    print(f"    ContextWindow:  Keep conversation under 20 messages")
    print(f"    ToolRetry:      Retry flaky tools up to 3 times")

    print(f"\n  --- Task: Use flaky API (will retry automatically) ---")
    result = await agent.execute({
        "id": "10",
        "objective": "Search the API for 'machine learning trends'"
    })
    print(f"  Result: {str(result)[:200]}")
    print(f"  Flaky tool attempts: {_flaky_counter['n']}")


async def demo_11_fortress_agent():
    """COMPLEX: Fortress Agent = ALL 8 plugins together."""
    header(
        11, "Fortress Agent (ALL 8 plugins)", "ALL hooks active",
        "Every plugin running simultaneously - the ultimate protected agent",
    )
    _flaky_counter["n"] = 0

    fortress_handler = PolicyApprovalHandler(
        safe_tools=["add", "multiply", "search_contacts", "get_weather", "flaky_api_call"],
        dangerous_tools=["delete_file"],
        default_allow=False,
    )

    llm = BaseOpenAI(model_name="gpt-4o-mini")
    memory = MemoryFactory.create_memory(MemoryStrategy.FULL_HISTORY)

    agent = Agent(
        name="FortressBot",
        role="The most protected agent with all 8 security and reliability plugins",
        objective="Help users with maximum safety and reliability",
        llm=llm,
        memory=memory,
        tools=[add_tool, search_tool, delete_tool, weather_tool, flaky_tool],
        config=AgentConfig(execution_mode=ExecutionMode.STANDARD),
        plugins=[
            # --- Model-level plugins ---
            ModelCallLimitPlugin(max_calls=10),
            ModelFallbackPlugin(fallbacks=["gpt-4o-mini"]),
            PIIGuardPlugin(
                pii_types=["email", "phone", "ssn", "credit_card", "ip_address"],
                strategy="redact",
                apply_to_output=True,
            ),
            ContextWindowPlugin(max_messages=20, keep_recent=8),
            # --- Tool-level plugins ---
            ToolCallLimitPlugin(max_calls=5),
            ToolRetryPlugin(max_retries=2, base_delay=0.1),
            HumanApprovalPlugin(
                approval_handler=fortress_handler,
                auto_approve=["add", "get_weather"],
            ),
            ToolGuardPlugin(blocked=["restart_server", "deploy_to_production"]),
        ],
    )
    await agent.initialize()

    print(f"\n  All 8 plugins active:")
    print(f"    [before_model]    ModelCallLimitPlugin   max=10")
    print(f"    [wrap_model_call] ModelFallbackPlugin    fallback=gpt-4o-mini")
    print(f"    [before+after]    PIIGuardPlugin         strategy=redact")
    print(f"    [before_model]    ContextWindowPlugin    max_messages=20")
    print(f"    [wrap_tool_call]  ToolCallLimitPlugin    max=5")
    print(f"    [wrap_tool_call]  ToolRetryPlugin        retries=2")
    print(f"    [wrap_tool_call]  HumanApprovalPlugin    PolicyApprovalHandler")
    print(f"    [wrap_tool_call]  ToolGuardPlugin        blocks restart,deploy")

    print(f"\n  --- Task A: Safe math (auto-approved, handler not called) ---")
    result = await agent.execute({"id": "11a", "objective": "What is 33 + 67?"})
    print(f"  Result: {str(result)[:120]}")

    print(f"\n  --- Task B: Contact lookup (PII redacted) ---")
    result = await agent.execute({
        "id": "11b",
        "objective": "Look up Bob's contact information"
    })
    print(f"  Result: {str(result)[:200]}")

    print(f"\n  --- Task C: Delete attempt (PolicyApprovalHandler denies) ---")
    result = await agent.execute({
        "id": "11c",
        "objective": "Delete the file /etc/passwd"
    })
    print(f"  Result: {str(result)[:200]}")

    print(f"\n  Fortress audit trail:")
    for entry in fortress_handler.audit_log:
        status = "APPROVED" if entry["approved"] else "DENIED"
        print(f"    [{status}] {entry['tool']} -> {entry['reason']}")
    print(f"  Flaky tool attempts: {_flaky_counter['n']}")

    print(f"\n  Plugin execution order for a tool call:")
    print(f"    Request -> ToolCallLimit -> ToolRetry -> HumanApproval -> ToolGuard -> Tool")
    print(f"  Plugin execution order for a model call:")
    print(f"    Request -> ModelCallLimit -> PIIGuard -> ContextWindow -> ModelFallback -> LLM")


# ====================================================================
# RUNNER
# ====================================================================

async def main():
    print(f"\n{'#'*65}")
    print(f"  NucleusIQ Plugin System -- Comprehensive Demo")
    print(f"  8 built-in plugins x 11 scenarios")
    print(f"{'#'*65}")

    demos = [
        ("PART A: Individual Plugin Demos", None),
        ("1. ModelCallLimitPlugin",  demo_1_model_call_limit),
        ("2. ToolCallLimitPlugin",   demo_2_tool_call_limit),
        ("3. ToolRetryPlugin",       demo_3_tool_retry),
        ("4. ModelFallbackPlugin",   demo_4_model_fallback),
        ("5. PIIGuardPlugin",        demo_5_pii_guard),
        ("6. HumanApprovalPlugin",   demo_6_human_approval),
        ("7. ContextWindowPlugin",   demo_7_context_window),
        ("8. ToolGuardPlugin",       demo_8_tool_guard),
        ("PART B: Complex Multi-Plugin Scenarios", None),
        ("9. Secure Agent",          demo_9_secure_agent),
        ("10. Resilient Agent",      demo_10_resilient_agent),
        ("11. Fortress Agent",       demo_11_fortress_agent),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, fn in demos:
        if fn is None:
            print(f"\n\n{'#'*65}")
            print(f"  {name}")
            print(f"{'#'*65}")
            continue

        try:
            await fn()
            print(f"\n  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            errors.append(f"{name}: {e}")

    print(f"\n\n{'#'*65}")
    print(f"  FINAL RESULTS: {passed} passed, {failed} failed out of 11")
    if errors:
        print(f"\n  Errors:")
        for err in errors:
            print(f"    - {err}")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    asyncio.run(main())
