"""
Integration test: Memory persists across multiple agent.execute() calls.

Scenario (applied to **every** memory strategy):
  1. User says "Hello My name is Brijesh and I am a creator of NucleusIQ"
  2. User asks  "Who is the creator of NucleusIQ?"
  3. The answer MUST contain "Brijesh" — proving memory is read by the LLM.

Run with:
  pytest tests/memory/integration/ -m integration
  python tests/memory/integration/test_memory_integration.py   (standalone)

Requires: OPENAI_API_KEY environment variable.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[5]
_env_file = _repo_root / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file, override=False)

_openai_dir = str(Path(__file__).resolve().parents[3] / "providers" / "llms" / "openai")
if _openai_dir not in sys.path:
    sys.path.insert(0, _openai_dir)

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.memory.sliding_window import SlidingWindowMemory
from nucleusiq.memory.summary import SummaryMemory
from nucleusiq.memory.summary_window import SummaryWindowMemory
from nucleusiq.memory.token_budget import TokenBudgetMemory

_HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))

INTRO_MSG = "Hello My name is Brijesh and I am a creator of NucleusIQ"
RECALL_MSG = "Who is the creator of NucleusIQ?"


def _make_openai_llm():
    from nucleusiq_openai import BaseOpenAI

    return BaseOpenAI(model_name="gpt-4o-mini", temperature=0.0)


async def _run_two_turn_recall(memory, llm=None):
    """Run the standard two-turn recall scenario and return (result1, result2, memory)."""
    if llm is None:
        llm = _make_openai_llm()

    agent = Agent(
        name="MemoryBot",
        role="Assistant",
        objective="Answer user questions using conversation history.",
        narrative=(
            "You are a helpful assistant. "
            "Always use the conversation history to answer questions."
        ),
        llm=llm,
        memory=memory,
        config=AgentConfig(verbose=True),
    )
    await agent.initialize()

    result1 = await agent.execute(
        {
            "id": "intro",
            "objective": INTRO_MSG,
        }
    )
    assert result1 is not None, "First turn returned None"

    result2 = await agent.execute(
        {
            "id": "recall",
            "objective": RECALL_MSG,
        }
    )
    assert result2 is not None, "Second turn returned None"

    return result1, result2, memory


# ── 1. FullHistoryMemory ────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_full_history_memory_recall():
    """FullHistoryMemory keeps every message verbatim."""
    memory = FullHistoryMemory()
    _, result2, mem = await _run_two_turn_recall(memory)

    ctx = mem.get_context()
    roles = [m["role"] for m in ctx]
    assert "user" in roles, "User messages must be stored"
    assert "assistant" in roles, "Assistant messages must be stored"
    assert len(ctx) == 4, f"Expected 4 messages (2 user + 2 assistant), got {len(ctx)}"

    assert "brijesh" in str(result2).lower(), (
        f"FullHistoryMemory: Agent failed to recall 'Brijesh'. Got: {result2}"
    )


# ── 2. SlidingWindowMemory ──────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_sliding_window_memory_recall():
    """SlidingWindowMemory keeps last k messages — window=10 is plenty for 2 turns."""
    memory = SlidingWindowMemory(window_size=10)
    _, result2, mem = await _run_two_turn_recall(memory)

    ctx = mem.get_context()
    assert len(ctx) == 4, f"Expected 4 messages within window, got {len(ctx)}"

    assert "brijesh" in str(result2).lower(), (
        f"SlidingWindowMemory: Agent failed to recall 'Brijesh'. Got: {result2}"
    )


# ── 3. TokenBudgetMemory ────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_token_budget_memory_recall():
    """TokenBudgetMemory keeps messages within token limit — 4096 is generous for 2 turns."""
    memory = TokenBudgetMemory(max_tokens=4096)
    _, result2, mem = await _run_two_turn_recall(memory)

    ctx = mem.get_context()
    assert len(ctx) == 4, f"Expected 4 messages within budget, got {len(ctx)}"

    assert "brijesh" in str(result2).lower(), (
        f"TokenBudgetMemory: Agent failed to recall 'Brijesh'. Got: {result2}"
    )


# ── 4. SummaryMemory ────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_summary_memory_recall():
    """SummaryMemory condenses conversation into a running summary via LLM.

    The summary must retain the fact that Brijesh is the creator.
    """
    llm = _make_openai_llm()
    memory = SummaryMemory(llm=llm, llm_model="gpt-4o-mini")
    _, result2, mem = await _run_two_turn_recall(memory, llm=llm)

    ctx = mem.get_context()
    assert len(ctx) >= 1, "Summary context must not be empty"
    summary_text = ctx[0]["content"]
    assert "brijesh" in summary_text.lower(), (
        f"SummaryMemory: Summary lost 'Brijesh'. Summary: {summary_text}"
    )

    assert "brijesh" in str(result2).lower(), (
        f"SummaryMemory: Agent failed to recall 'Brijesh'. Got: {result2}"
    )


# ── 5. SummaryWindowMemory ──────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_summary_window_memory_recall():
    """SummaryWindowMemory keeps recent messages verbatim + older ones summarized.

    With window_size=10, all 4 messages stay in the window (no summarization
    triggered). The key assertion is that the agent can still recall Brijesh.
    """
    llm = _make_openai_llm()
    memory = SummaryWindowMemory(window_size=10, llm=llm, llm_model="gpt-4o-mini")
    _, result2, mem = await _run_two_turn_recall(memory, llm=llm)

    ctx = mem.get_context()
    assert len(ctx) >= 4, f"Expected at least 4 context entries, got {len(ctx)}"

    assert "brijesh" in str(result2).lower(), (
        f"SummaryWindowMemory: Agent failed to recall 'Brijesh'. Got: {result2}"
    )


# ── 6. SummaryWindowMemory with overflow ─────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_summary_window_memory_with_overflow():
    """SummaryWindowMemory with small window forces summarization of older messages.

    window_size=2 means after turn 1 (2 messages: user + assistant), when turn 2
    starts the window overflows and the oldest messages get summarized.
    The agent should still recall Brijesh from the summary.
    """
    llm = _make_openai_llm()
    memory = SummaryWindowMemory(window_size=2, llm=llm, llm_model="gpt-4o-mini")
    _, result2, mem = await _run_two_turn_recall(memory, llm=llm)

    ctx = mem.get_context()
    has_summary = any(m.get("role") == "system" for m in ctx)
    assert has_summary, "Expected a summary system message after window overflow"

    assert "brijesh" in str(result2).lower(), (
        f"SummaryWindowMemory (overflow): Agent failed to recall 'Brijesh'. Got: {result2}"
    )


# ── 7. SlidingWindowMemory with tight window ─────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_sliding_window_memory_tight_window():
    """SlidingWindowMemory with window_size=2 only keeps the last 2 messages.

    This means the intro message gets evicted, and the agent should NOT be able
    to recall Brijesh — verifying eviction works correctly.
    """
    memory = SlidingWindowMemory(window_size=2)
    _, result2, mem = await _run_two_turn_recall(memory)

    ctx = mem.get_context()
    assert len(ctx) == 2, f"Tight window should keep exactly 2 messages, got {len(ctx)}"

    user_contents = [m["content"] for m in ctx if m["role"] == "user"]
    assert not any("brijesh" in c.lower() for c in user_contents), (
        "Tight window should have evicted the intro message containing 'Brijesh'"
    )


# ── 8. MemoryFactory integration ─────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_memory_factory_creates_all_strategies():
    """MemoryFactory.create_memory works for every built-in strategy."""
    from nucleusiq.memory.factory import MemoryFactory, MemoryStrategy

    llm = _make_openai_llm()

    strategies = [
        (MemoryStrategy.FULL_HISTORY, {}),
        (MemoryStrategy.SLIDING_WINDOW, {"window_size": 10}),
        (MemoryStrategy.TOKEN_BUDGET, {"max_tokens": 4096}),
        (MemoryStrategy.SUMMARY, {"llm": llm, "llm_model": "gpt-4o-mini"}),
        (
            MemoryStrategy.SUMMARY_WINDOW,
            {"window_size": 10, "llm": llm, "llm_model": "gpt-4o-mini"},
        ),
    ]

    for strategy, kwargs in strategies:
        memory = MemoryFactory.create_memory(strategy, **kwargs)
        _, result2, _ = await _run_two_turn_recall(memory, llm=llm)
        assert "brijesh" in str(result2).lower(), (
            f"MemoryFactory({strategy.value}): Agent failed to recall 'Brijesh'. Got: {result2}"
        )


# ── Standalone runner ────────────────────────────────────────────────────

if __name__ == "__main__":

    async def _main():
        if not _HAS_KEY:
            print("SKIP: OPENAI_API_KEY not set")
            return

        tests = [
            ("FullHistoryMemory", test_full_history_memory_recall),
            ("SlidingWindowMemory", test_sliding_window_memory_recall),
            ("TokenBudgetMemory", test_token_budget_memory_recall),
            ("SummaryMemory", test_summary_memory_recall),
            ("SummaryWindowMemory", test_summary_window_memory_recall),
            (
                "SummaryWindowMemory (overflow)",
                test_summary_window_memory_with_overflow,
            ),
            ("SlidingWindowMemory (tight)", test_sliding_window_memory_tight_window),
            (
                "MemoryFactory (all strategies)",
                test_memory_factory_creates_all_strategies,
            ),
        ]

        passed = 0
        for name, fn in tests:
            try:
                await fn()
                print(f"  PASS: {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {name} — {e}")

        print(f"\n{passed}/{len(tests)} tests passed")
        if passed == len(tests):
            print("All memory integration tests passed!")
        else:
            print("Some tests failed. Review output above.")

    asyncio.run(_main())
