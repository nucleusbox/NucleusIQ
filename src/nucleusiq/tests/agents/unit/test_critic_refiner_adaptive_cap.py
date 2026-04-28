"""End-to-end wiring tests for the v0.7.8 adaptive tool-result cap.

The individual ``compute_per_tool_cap`` numerics are tested in
``tests/unit/context/test_adaptive_tool_cap.py``; this module checks
the *glue* between the helper and the Critic/Refiner runners:

* ``CriticRunner._compute_critic_per_tool_cap`` — reads the agent's
  LLM context window and the count of tool results to compute the cap.
* ``RefinerRunner._compute_refiner_char_caps`` — mirrors the Critic
  with its own (larger) overhead/reserve defaults.
* The Critic's ``_extract_reasoning_trace`` honours
  ``per_tool_char_cap`` instead of the legacy fixed
  ``CriticLimits.tool_result``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.components.critic import Critic
from nucleusiq.agents.context.config import ContextConfig
from nucleusiq.agents.modes.autonomous.critic_runner import (
    _compute_critic_per_tool_cap,
)
from nucleusiq.agents.modes.autonomous.refiner_runner import (
    _compute_refiner_char_caps,
)


class _FakeLLM:
    """Minimal LLM double that lets tests vary ``get_context_window``."""

    model_name = "fake-model"
    is_reasoning_model = False

    def __init__(self, ctx_window: int) -> None:
        self._ctx = ctx_window

    def get_context_window(self) -> int:
        return self._ctx


def _make_agent(
    *,
    ctx_window: int = 128_000,
    context_cfg: ContextConfig | None = None,
):
    agent = MagicMock()
    agent.llm = _FakeLLM(ctx_window)
    agent.config = MagicMock()
    agent.config.context = context_cfg or ContextConfig()
    return agent


class TestCriticRunnerCapWiring:
    def test_large_window_gives_generous_cap(self):
        agent = _make_agent(ctx_window=400_000)
        messages = [ChatMessage(role="tool", content="x") for _ in range(10)]
        cap = _compute_critic_per_tool_cap(agent, messages)
        # 400K - 5K - 8K = 387K tokens / 10 tools = 38.7K tokens
        # → 154K chars → clamped to default max 50K.
        assert cap == 50_000

    def test_small_window_many_tools_reduces_cap(self):
        agent = _make_agent(ctx_window=32_000)
        messages = [ChatMessage(role="tool", content="x") for _ in range(60)]
        cap = _compute_critic_per_tool_cap(agent, messages)
        # (32K - 13K) / 60 = 316 tokens = 1264 chars.
        assert cap == 1264

    def test_no_context_config_returns_none(self):
        agent = MagicMock()
        agent.llm = _FakeLLM(128_000)
        agent.config.context = None
        messages = [ChatMessage(role="tool", content="x")]
        assert _compute_critic_per_tool_cap(agent, messages) is None

    def test_no_llm_returns_none(self):
        agent = MagicMock()
        agent.llm = None
        agent.config = MagicMock()
        agent.config.context = ContextConfig()
        messages = [ChatMessage(role="tool", content="x")]
        assert _compute_critic_per_tool_cap(agent, messages) is None

    def test_zero_tool_messages_does_not_explode(self):
        agent = _make_agent(ctx_window=128_000)
        messages = [ChatMessage(role="user", content="hi")]
        cap = _compute_critic_per_tool_cap(agent, messages)
        assert cap == 50_000  # hits ceiling because num_tool_results=0→1


class TestRefinerRunnerCapWiring:
    def test_returns_per_tool_and_total_caps(self):
        agent = _make_agent(ctx_window=128_000)
        messages = [ChatMessage(role="tool", content="x") for _ in range(20)]
        per_tool, total = _compute_refiner_char_caps(agent, messages)
        assert per_tool is not None and total is not None
        assert per_tool > 0
        # total should be ≥ per_tool * num_tools (it's the budget cap).
        assert total >= per_tool * 20

    def test_refiner_reserves_more_than_critic(self):
        agent = _make_agent(ctx_window=128_000)
        messages = [ChatMessage(role="tool", content="x") for _ in range(20)]
        critic_cap = _compute_critic_per_tool_cap(agent, messages)
        refiner_cap, _ = _compute_refiner_char_caps(agent, messages)
        # Defaults: refiner prompt_overhead=8K, reserve=16K vs critic 5K/8K.
        # So the Refiner gets a strictly *smaller* per-tool slice at same
        # window — it's paying for the space to write the full answer.
        assert refiner_cap is not None and critic_cap is not None
        assert refiner_cap <= critic_cap


class TestCriticHonoursExplicitCap:
    """The Critic must use the explicit cap, not the legacy default."""

    def test_explicit_cap_overrides_limits_tool_result(self):
        critic = Critic()
        long_result = "Z" * 10_000
        messages = [
            ChatMessage(role="tool", content=long_result, name="t"),
        ]
        trace = critic._extract_reasoning_trace(
            messages,
            per_tool_char_cap=500,  # far below default 3_000
        )
        result_line = [line for line in trace.split("\n") if "[Tool Result]" in line][0]
        content = result_line.replace("[Tool Result] ", "")
        assert len(content) <= 510  # 500 + ellipsis

    def test_no_explicit_cap_falls_back_to_limits(self):
        """Backward compat: callers that don't pass per_tool_char_cap
        still get the legacy fixed ``limits.tool_result``."""
        critic = Critic()
        long_result = "Z" * 10_000
        messages = [
            ChatMessage(role="tool", content=long_result, name="t"),
        ]
        trace = critic._extract_reasoning_trace(messages)  # no cap
        result_line = [line for line in trace.split("\n") if "[Tool Result]" in line][0]
        content = result_line.replace("[Tool Result] ", "")
        # STANDARD_LIMITS.tool_result = 3_000
        assert 2_990 <= len(content) <= 3_010
