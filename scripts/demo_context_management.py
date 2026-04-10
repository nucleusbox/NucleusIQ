"""
Context Window Management — Phase 2 Real LLM Proof
====================================================

This script demonstrates NucleusIQ's Phase 2 context hygiene system
working with a REAL Gemini LLM, not mocks.

What it proves:
  1. optimal_budget (50K) drives compaction, not 1M model limit
  2. ObservationMasker (Tier 0) strips consumed tool results automatically
  3. post_response() fires after each LLM response
  4. Cost telemetry shows dollar savings
  5. ContextEngine handles the full lifecycle: ingest → prepare → post_response

Run:
    python scripts/demo_context_management.py
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "nucleusiq"))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src", "providers", "llms", "gemini"),
)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config.agent_config import AgentConfig
from nucleusiq.agents.context import ContextConfig, ContextStrategy
from nucleusiq.agents.task import Task
from nucleusiq.tools import tool

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
ctx_logger = logging.getLogger("nucleusiq.agents.context.engine")
ctx_logger.setLevel(logging.DEBUG)


# ------------------------------------------------------------------ #
# Custom tools that return LARGE results (simulating real-world data)  #
# ------------------------------------------------------------------ #


@tool(
    name="fetch_market_report",
    description="Fetch a comprehensive market analysis report for a given sector",
)
def fetch_market_report(sector: str) -> str:
    """Returns a large market report (simulating a real API response)."""
    print(f"\n  >>> [TOOL CALLED] fetch_market_report(sector='{sector}')")
    lines = [
        f"=== Market Analysis Report: {sector} ===",
        "Date: April 2026 | Source: MarketResearchPro API",
        "",
        "Executive Summary:",
        f"  The {sector} sector showed strong growth in Q1 2026, driven by",
        "  increased adoption of AI technologies and regulatory changes.",
        "",
        "Key Metrics:",
    ]
    for i in range(25):
        lines.append(
            f"  Company_{i:03d} | Revenue: ${10000 + i * 1500:,}M | "
            f"Growth: {5 + i * 0.3:.1f}% | Market Share: {2 + i * 0.4:.1f}% | "
            f"Notes: Detailed analysis of company {i} performance metrics including "
            f"quarterly breakdown, regional distribution, and competitive positioning "
            f"across multiple segments with forecast data for the next fiscal year"
        )
    lines.extend(
        [
            "",
            "Sector Trends:",
            "  1. AI adoption accelerating across enterprise",
            "  2. Regulatory compliance driving new investment",
            "  3. Cloud-native architecture becoming standard",
            "  4. Consolidation through M&A activity increasing",
            "",
            "Risk Factors:",
            "  - Supply chain disruptions (moderate)",
            "  - Interest rate sensitivity (high)",
            "  - Competitive pressure from new entrants (growing)",
        ]
    )

    result = "\n".join(lines)
    print(
        f"  >>> [TOOL RESULT] {len(result)} chars, ~{len(result) // 4} estimated tokens"
    )
    return result


@tool(
    name="fetch_financial_data",
    description="Fetch detailed financial data for multiple companies in a sector",
)
def fetch_financial_data(sector: str, num_companies: int = 20) -> str:
    """Returns structured financial data for companies."""
    print(
        f"\n  >>> [TOOL CALLED] fetch_financial_data(sector='{sector}', n={num_companies})"
    )
    records = []
    for i in range(num_companies):
        records.append(
            {
                "company": f"Corp_{i:03d}",
                "revenue_q1": 5000 + i * 800,
                "revenue_q2": 5200 + i * 900,
                "ebitda": 1200 + i * 200,
                "debt_ratio": round(0.3 + i * 0.02, 2),
                "employees": 1000 + i * 500,
                "description": f"Company {i} operates in the {sector} sector "
                f"with focus on enterprise solutions and has shown "
                f"consistent growth across all major markets",
            }
        )
    result = json.dumps({"sector": sector, "companies": records}, indent=2)
    print(
        f"  >>> [TOOL RESULT] {len(result)} chars, ~{len(result) // 4} estimated tokens"
    )
    return result


@tool(name="summarize_findings", description="Generate a brief summary of key findings")
def summarize_findings(topic: str) -> str:
    """Returns a short summary."""
    print(f"\n  >>> [TOOL CALLED] summarize_findings(topic='{topic}')")
    result = f"Summary of {topic}: Key findings indicate positive trends with moderate risk factors."
    print(f"  >>> [TOOL RESULT] {len(result)} chars (small)")
    return result


async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        return

    from nucleusiq_gemini import BaseGemini

    llm = BaseGemini(model_name="gemini-2.5-flash", api_key=api_key)

    print("=" * 70)
    print("  PHASE 2 PROOF: Context Hygiene for Quality, Cost, and Latency")
    print("=" * 70)
    print(f"\n  LLM: {llm.model_name}")
    print(f"  Context Window: {llm.get_context_window():,} tokens")
    print()

    # -------------------------------------------------------------- #
    # Step 1: Agent with Phase 2 context management                    #
    # -------------------------------------------------------------- #
    print("-" * 70)
    print("  STEP 1: Agent with Phase 2 context management")
    print("  optimal_budget=20K, observation masking ON, cost tracking at $3/M")
    print("-" * 70)

    from nucleusiq.prompts.zero_shot import ZeroShotPrompt

    agent_with = Agent(
        name="MarketAnalyst",
        role="Senior Market Research Analyst",
        objective="Analyze market sectors and provide investment insights",
        prompt=ZeroShotPrompt().configure(
            system="You fetch market reports and financial data, then synthesize findings.",
        ),
        llm=llm,
        tools=[fetch_market_report, fetch_financial_data, summarize_findings],
        config=AgentConfig(
            execution_mode="standard",
            context=ContextConfig(
                optimal_budget=20_000,
                response_reserve=4096,
                tool_result_threshold=500,
                strategy=ContextStrategy.PROGRESSIVE,
                enable_offloading=True,
                enable_observation_masking=True,
                cost_per_million_input=3.0,
                preserve_recent_turns=3,
            ),
            enable_tracing=True,
            verbose=True,
        ),
    )

    task = Task.from_dict(
        {
            "id": "market-analysis-001",
            "objective": (
                "Fetch the market report for the 'AI & Machine Learning' sector "
                "and the financial data for companies in that sector, "
                "then summarize the key investment findings."
            ),
        }
    )

    print("\n  Executing agent with Phase 2 context management...")
    print("  Watch: [TOOL CALLED], ObservationMasker logs, cost telemetry.\n")

    result = await agent_with.execute(task)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  Status       : {result.status}")
    print(f"  LLM Calls    : {len(result.llm_calls)}")
    tool_call_count = sum(1 for c in result.llm_calls if c.has_tool_calls)
    print(f"  Tool Calls   : {tool_call_count}")

    if result.context_telemetry:
        ct = result.context_telemetry
        print("\n  --- Context Telemetry (Phase 2) ---")
        print(f"  Optimal Budget   : {ct.optimal_budget:,} tokens")
        print(f"  Context Limit    : {ct.context_limit:,} tokens")
        print(f"  Response Reserve : {ct.response_reserve:,} tokens")
        print(f"  Peak Utilization : {ct.peak_utilization:.1%}")
        print(f"  Final Utilization: {ct.final_utilization:.1%}")
        print(f"  Compactions      : {ct.compaction_count}")
        print(f"  Tokens Freed     : {ct.tokens_freed_total:,}")
        print(f"  Artifacts Stored : {ct.artifacts_offloaded}")
        print("\n  --- Observation Masking (Tier 0) ---")
        print(f"  Observations Masked: {ct.observations_masked}")
        print(f"  Tokens Masked      : {ct.tokens_masked:,}")
        print("\n  --- Cost Estimation ---")
        print(f"  Cost w/o management: ${ct.estimated_cost_without_mgmt:.6f}")
        print(f"  Cost with management: ${ct.estimated_cost_with_mgmt:.6f}")
        print(f"  Savings            : {ct.estimated_savings_pct:.1f}%")
        if ct.region_breakdown:
            print("\n  Region Breakdown:")
            for region, tokens in sorted(ct.region_breakdown.items()):
                if tokens > 0:
                    print(f"    {region:15s}: {tokens:,} tokens")
        if ct.compaction_events:
            print("\n  Compaction Events:")
            for ev in ct.compaction_events:
                print(
                    f"    [{ev.strategy}] "
                    f"{ev.tokens_before:,} → {ev.tokens_after:,} "
                    f"(freed {ev.tokens_freed:,}, {ev.duration_ms:.2f}ms)"
                )
    else:
        print("\n  Context telemetry: None (engine may not have been triggered)")

    print("\n  --- Agent Output (first 500 chars) ---")
    output_str = str(result.output)
    print(f"  {output_str[:500]}...")

    print("\n  --- Full AgentResult.display() ---")
    print(result.display())

    # -------------------------------------------------------------- #
    # Step 2: Direct ObservationMasker demo                            #
    # -------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("  STEP 2: Direct ObservationMasker Demo — post_response()")
    print("=" * 70)

    from nucleusiq.agents.chat_models import ChatMessage
    from nucleusiq.agents.context import ContextEngine, DefaultTokenCounter

    engine = ContextEngine(
        config=ContextConfig(
            optimal_budget=10_000,
            tool_result_threshold=100,
            enable_offloading=True,
            enable_observation_masking=True,
            cost_per_million_input=3.0,
        ),
        token_counter=DefaultTokenCounter(),
        max_tokens=1_000_000,
    )

    tool_result_content = "A" * 2000
    messages = [
        ChatMessage(role="system", content="You are a research analyst."),
        ChatMessage(role="user", content="Analyze the market."),
        ChatMessage(
            role="tool",
            content=tool_result_content,
            name="market_search",
            tool_call_id="tc_demo_1",
        ),
        ChatMessage(
            role="assistant",
            content="Based on the market data, here are the key trends...",
        ),
        ChatMessage(
            role="tool",
            content="B" * 1500,
            name="financial_data",
            tool_call_id="tc_demo_2",
        ),
    ]

    print("\n  Before post_response():")
    print(f"    Tool result 1: {len(messages[2].content)} chars")
    print(f"    Tool result 2: {len(messages[4].content)} chars")
    print(f"    Store size: {engine.store.size}")

    masked = engine.post_response(messages)

    print("\n  After post_response():")
    print(
        f"    Tool result 1: {len(masked[2].content)} chars (was {len(tool_result_content)})"
    )
    print(f"    Tool result 2: {len(masked[4].content)} chars (unchanged — unconsumed)")
    print(f"    Store size: {engine.store.size}")
    print(f"    Masked marker: {masked[2].content[:80]}...")

    tel = engine.telemetry
    print("\n  Telemetry:")
    print(f"    Observations masked: {tel.observations_masked}")
    print(f"    Tokens masked: {tel.tokens_masked}")
    print(f"    Optimal budget: {tel.optimal_budget:,}")

    if engine.store.size > 0:
        key = engine.store.keys()[0]  # noqa: SIM118
        full = engine.store.retrieve(key)
        print(f"    Full content retrievable: {full == tool_result_content}")

    # -------------------------------------------------------------- #
    # Step 3: Show ingest_tool_result + cost telemetry                 #
    # -------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("  STEP 3: Cost Telemetry Demo")
    print("=" * 70)

    engine2 = ContextEngine(
        config=ContextConfig(
            optimal_budget=10_000,
            tool_result_threshold=100,
            enable_offloading=True,
            cost_per_million_input=3.0,
        ),
        token_counter=DefaultTokenCounter(),
        max_tokens=1_000_000,
    )

    big_lines = [
        f"Company_{i:03d} | Revenue: ${10000 + i * 1500:,}M | " * 5 for i in range(30)
    ]
    big_report = "\n".join(big_lines)
    print(f"\n  Original tool result: {len(big_report)} chars")

    compressed = engine2.ingest_tool_result(big_report, "fetch_market_report")
    print(f"  After ingest: {len(compressed)} chars")
    print(f"  Compression: {(1 - len(compressed) / len(big_report)):.1%}")

    msgs = [
        ChatMessage(role="system", content="You help."),
        ChatMessage(
            role="tool", content=compressed, name="report", tool_call_id="tc_cost"
        ),
    ]
    await engine2.prepare(msgs)
    tel2 = engine2.telemetry
    print("\n  Cost Telemetry:")
    print(f"    Cost w/o management: ${tel2.estimated_cost_without_mgmt:.6f}")
    print(f"    Cost with management: ${tel2.estimated_cost_with_mgmt:.6f}")
    print(f"    Savings: {tel2.estimated_savings_pct:.1f}%")

    print("\n" + "=" * 70)
    print("  PHASE 2 PROOF COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
