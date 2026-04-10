"""Tests for MemorySnapshot capture in Agent._build_result()."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.memory.base import BaseMemory
from nucleusiq.tests.conftest import make_test_prompt


class _Msg:
    """Lightweight message object with .role and .content."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _TestMemory(BaseMemory):
    """Minimal BaseMemory subclass that exposes ``messages`` and
    ``token_count`` for MemorySnapshot capture tests."""

    _messages: list = []
    _token_count: int | None = None

    def model_post_init(self, __context: Any) -> None:
        self._messages = []
        self._token_count = None

    @property
    def strategy_name(self) -> str:
        return "test_memory"

    @property
    def messages(self) -> list:
        return self._messages

    @property
    def token_count(self) -> int | None:
        return self._token_count

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self._messages.append(_Msg(role, content))

    def get_context(
        self, query: str | None = None, **kwargs: Any
    ) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        self._messages.clear()


def _make_agent(memory=None, tracing=True):
    return Agent(
        name="MemTest",
        role="tester",
        objective="test memory snapshots",
        prompt=make_test_prompt(),
        llm=MockLLM(),
        memory=memory,
        config=AgentConfig(
            execution_mode="direct",
            verbose=False,
            enable_tracing=tracing,
        ),
    )


class TestMemorySnapshotCapture:
    """Verify that Agent._build_result() captures a MemorySnapshot
    when both a tracer and a memory are present."""

    @pytest.mark.asyncio
    async def test_memory_snapshot_captured_via_execute(self):
        mem = _TestMemory()
        mem.add_message("user", "hello")
        mem.add_message("assistant", "hi")
        mem._token_count = 100

        agent = _make_agent(memory=mem, tracing=True)
        await agent.initialize()
        result = await agent.execute(Task(id="t1", objective="say hi"))

        assert result.memory_snapshot is not None
        assert result.memory_snapshot.strategy == "_TestMemory"
        assert result.memory_snapshot.message_count >= 2
        assert result.memory_snapshot.token_count == 100
        assert len(result.memory_snapshot.messages) >= 2

    @pytest.mark.asyncio
    async def test_no_memory_no_snapshot(self):
        agent = _make_agent(memory=None, tracing=True)
        await agent.initialize()
        result = await agent.execute(Task(id="t2", objective="no memory"))

        assert result.memory_snapshot is None

    @pytest.mark.asyncio
    async def test_no_tracer_no_snapshot(self):
        mem = _TestMemory()
        mem.add_message("user", "hello")

        agent = _make_agent(memory=mem, tracing=False)
        await agent.initialize()
        result = await agent.execute(Task(id="t3", objective="no tracer"))

        assert result.memory_snapshot is None

    @pytest.mark.asyncio
    async def test_long_messages_truncated(self):
        mem = _TestMemory()
        for i in range(20):
            mem.add_message("user", "x" * 500)

        agent = _make_agent(memory=mem, tracing=True)
        await agent.initialize()
        result = await agent.execute(Task(id="t4", objective="truncation"))

        assert result.memory_snapshot is not None
        assert len(result.memory_snapshot.messages) <= 10
        for msg in result.memory_snapshot.messages:
            assert len(msg["content"]) <= 200

    @pytest.mark.asyncio
    async def test_snapshot_strategy_name_matches_class(self):
        mem = _TestMemory()
        mem.add_message("system", "init")

        agent = _make_agent(memory=mem, tracing=True)
        await agent.initialize()
        result = await agent.execute(Task(id="t5", objective="strategy"))

        assert result.memory_snapshot is not None
        assert result.memory_snapshot.strategy == "_TestMemory"
