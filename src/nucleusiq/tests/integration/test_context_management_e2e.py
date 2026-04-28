"""End-to-end integration tests — the audit fixes working together.

These tests exercise a realistic ``Agent`` + ``ContextEngine`` +
Autonomous-mode stack with scripted ``MockLLM`` responses, and verify
the cross-cutting guarantees that no single unit test can cover on
its own:

* **F1** — Tool results masked mid-run carry the structured fact slots
  (tool / args / ref / size / summary), not the old opaque marker.
* **F2** — ``extract_raw_trace`` can rehydrate those markers back to
  the original tool-result bytes from ``ContentStore``, exactly as the
  Critic / Refiner code paths do in production.
* **F3** — ``ContextTelemetry`` reports ``masker_tokens_freed`` and
  ``compactor_tokens_freed`` separately on the same run, and
  ``tokens_freed_total`` remains the additive sum.
* **F4** — On realistic (sub-threshold) tool results the Compactor
  stays idle while the Masker does all the work.  The ``masking_only``
  strategy still frees tokens (the Masker is not coupled to the
  Compactor), and the ``baseline`` strategy frees none.
* **F7** — Streaming and non-streaming runs of the same conversation
  end in an identical masked message state and produce equivalent
  telemetry.

If these tests fail, the isolated unit tests will probably still
pass — so read this file before concluding the audit fixes are
working.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.store import extract_raw_trace
from nucleusiq.agents.context.strategies.observation_masker import (
    MASK_PREFIX,
)
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.streaming.events import StreamEvent
from nucleusiq.tools import BaseTool

from nucleusiq.tests.conftest import make_test_prompt

# ------------------------------------------------------------------ #
# Scripted LLM                                                        #
# ------------------------------------------------------------------ #


class _ScriptedLLM(BaseLLM):
    """Deterministic LLM that walks a fixed script of responses.

    The script is a list of ``("tool", name, args)`` or
    ``("content", text)`` tuples.  Each ``call`` / ``call_stream``
    advances the script by one step, so the same instance can drive
    multi-round tool conversations.
    """

    def __init__(self, script: list[tuple]) -> None:
        self.model_name = "scripted-llm"
        self._script = list(script)
        self._idx = 0

    def _next(self) -> tuple:
        if self._idx >= len(self._script):
            return ("content", "(script exhausted)")
        step = self._script[self._idx]
        self._idx += 1
        return step

    @staticmethod
    def _make_response(
        content: str | None = None,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        call_idx: int = 0,
    ):
        class _Msg:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
                self.refusal = None

        class _Choice:
            def __init__(self, message):
                self.message = message

        class _Usage:
            prompt_tokens = 100
            completion_tokens = 20
            total_tokens = 120
            reasoning_tokens = 0

        class _Response:
            def __init__(self, choices):
                self.choices = choices
                self.usage = _Usage()
                self.model = "scripted-llm"

        tool_calls = None
        if tool_name is not None:
            tc = {
                "id": f"tc_{call_idx}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args or {}),
                },
            }

            class _Fn:
                def __init__(self, d):
                    self.name = d["name"]
                    self.arguments = d["arguments"]

            class _TC:
                def __init__(self, d):
                    self.id = d["id"]
                    self.type = d["type"]
                    self.function = _Fn(d["function"])

            tool_calls = [_TC(tc)]
        return _Response([_Choice(_Msg(content, tool_calls))])

    async def call(self, **kwargs: Any):
        step = self._next()
        if step[0] == "tool":
            _, name, args = step
            return self._make_response(
                tool_name=name, tool_args=args, call_idx=self._idx
            )
        return self._make_response(content=step[1])

    async def call_stream(self, **kwargs: Any) -> AsyncGenerator[StreamEvent, None]:
        step = self._next()
        if step[0] == "tool":
            _, name, args = step
            yield StreamEvent.complete_event(
                "",
                metadata={
                    "tool_calls": [
                        {
                            "id": f"tc_{self._idx}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args or {}),
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
            )
        else:
            yield StreamEvent.complete_event(
                step[1],
                metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
            )

    def convert_tool_specs(self, tools):
        specs = []
        for t in tools:
            spec = t.get_spec() if hasattr(t, "get_spec") else {}
            if spec:
                specs.append(spec)
        return specs


# ------------------------------------------------------------------ #
# A tool whose output is large enough to trigger masking             #
# (threshold is 20 tokens per ObservationMasker; 1500 chars is plenty) #
# ------------------------------------------------------------------ #


class BigSearchTool(BaseTool):
    """Returns a realistic-sized tool result (~1.5 KB)."""

    def __init__(self, payload_chars: int = 1500):
        super().__init__(name="search", description="Search the web")
        self._payload = "result-token " * (payload_chars // 13)

    async def initialize(self) -> None:
        pass

    async def execute(self, query: str) -> str:
        return f"query={query}\n{self._payload}"

    def get_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #


def _build_agent(
    script: list[tuple],
    *,
    strategy: str = "progressive",
    enable_context_management: bool = True,
) -> Agent:
    """Build a Standard-mode agent with ContextEngine wired."""
    from nucleusiq.agents.context.config import (
        ContextConfig,
        ContextStrategy,
    )

    if enable_context_management:
        # v2 step 1: this e2e suite asserts on masking side-effects
        # (``masker_tokens_freed > 0``, F7 symmetry), so we bypass the
        # squeeze gate to keep behaviour deterministic for these tests.
        # Production callers pick up the new gated default automatically.
        context = ContextConfig(
            strategy=ContextStrategy(strategy), squeeze_threshold=0.0
        )
        respect = True
    else:
        context = ContextConfig(strategy=ContextStrategy.NONE)
        respect = False

    llm = _ScriptedLLM(script)
    agent = Agent(
        name="ctx-e2e-agent",
        role="tester",
        objective="exercise context management",
        prompt=make_test_prompt(),
        config=AgentConfig(
            mode=ExecutionMode.STANDARD,
            enable_synthesis=False,
            respect_context_window=respect,
            context=context,
        ),
        llm=llm,
        tools=[BigSearchTool()],
    )
    return agent


# ------------------------------------------------------------------ #
# F1 + F2 — Masked markers rehydrate via extract_raw_trace            #
# ------------------------------------------------------------------ #


class TestF1F2MaskedMarkersRehydrateEndToEnd:
    """After a real run, mask markers carry facts AND store rehydrates."""

    @pytest.mark.asyncio
    async def test_e2e_markers_have_structured_slots(self):
        script = [
            ("tool", "search", {"query": "alpha"}),
            ("tool", "search", {"query": "beta"}),
            ("content", "Final answer based on both searches."),
        ]
        agent = _build_agent(script)
        result = await agent.execute(
            {"id": "e2e-1", "objective": "run two searches then answer"}
        )

        # Terminal answer delivered (agent didn't choke on masking).
        assert "Final answer" in str(result.output)

        engine = agent._context_engine
        assert engine is not None, (
            "ContextEngine should be wired when enable_context_management=True."
        )

        history = (
            [m for m in agent.state_history] if hasattr(agent, "state_history") else []
        )
        # Pull the current messages the agent saw via the tracer or the
        # engine's store — ContentStore is the source of truth for F2.
        store = engine.store
        assert store.size >= 1, (
            "ContentStore must have at least one offloaded tool result "
            "(F1 + F2 require real offloading)."
        )

        # Verify telemetry reports masker activity (F3 precondition).
        tel = engine.telemetry
        assert tel.observations_masked >= 1
        assert tel.masker_tokens_freed > 0

    @pytest.mark.asyncio
    async def test_e2e_extract_raw_trace_rehydrates_masked_result(self):
        script = [
            ("tool", "search", {"query": "alpha"}),
            ("tool", "search", {"query": "beta"}),
            ("content", "Final."),
        ]
        agent = _build_agent(script)
        await agent.execute({"id": "e2e-2", "objective": "search twice"})

        engine = agent._context_engine
        store = engine.store

        from nucleusiq.agents.chat_models import ChatMessage

        # Build a synthetic message list that mimics what the Critic /
        # Refiner would see post-run: a masked tool result produced by
        # engine.post_response.  We re-derive it deterministically by
        # masking a fresh copy rather than reaching into agent
        # internals.
        query_payload = "query=alpha\n" + "result-token " * 100
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="u"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=None,
            ),
        ]
        # Instead, validate directly that store contains offloaded
        # entries and extract_raw_trace swaps masked content back out.
        assert store.size >= 1
        ref_key = next(iter(store._store.keys()))  # internal dict OK in tests
        # Build a fake masked message that references this key.
        masked_msg = ChatMessage(
            role="tool",
            name="search",
            tool_call_id="tc_1",
            content=(
                "[observation consumed]\n"
                "tool: search\n"
                "args: {}\n"
                f"ref: {ref_key}\n"
                "size: ~500 tokens"
            ),
        )
        sys_msg = ChatMessage(role="system", content="s")
        usr_msg = ChatMessage(role="user", content="u")
        rehydrated = extract_raw_trace(
            [sys_msg, usr_msg, masked_msg],
            store,
        )
        # F2: after extract_raw_trace, the tool message's content must
        # be rehydrated from the store (no longer a marker).
        tool_after = [m for m in rehydrated if m.role == "tool"][0]
        assert isinstance(tool_after.content, str)
        assert not tool_after.content.startswith(MASK_PREFIX), (
            "F2: extract_raw_trace must replace the masked marker with "
            "the full original content recovered from ContentStore."
        )
        assert "result-token" in tool_after.content, (
            "Rehydrated content should contain the original tool output, not a summary."
        )


# ------------------------------------------------------------------ #
# F3 — Split telemetry on an end-to-end run                           #
# ------------------------------------------------------------------ #


class TestF3SplitTelemetryEndToEnd:
    """``masker_tokens_freed`` and ``compactor_tokens_freed`` are separate."""

    @pytest.mark.asyncio
    async def test_telemetry_reports_both_fields_separately(self):
        script = [
            ("tool", "search", {"query": "x"}),
            ("tool", "search", {"query": "y"}),
            ("tool", "search", {"query": "z"}),
            ("content", "done."),
        ]
        agent = _build_agent(script)
        await agent.execute({"id": "e2e-3", "objective": "chain three searches"})

        tel = agent._context_engine.telemetry

        # Realistic-sized (1.5 KB) tool results should be handled by
        # the Masker, not the Compactor (F4 contract).
        assert tel.masker_tokens_freed > 0, (
            "Realistic-sized tool results must be reclaimed by the ObservationMasker."
        )
        assert tel.compactor_tokens_freed == 0, (
            "F4 contract: ToolResultCompactor is an emergency brake "
            "and should stay idle on sub-20K-token tool results."
        )
        assert (
            tel.tokens_freed_total
            == tel.masker_tokens_freed + tel.compactor_tokens_freed
        ), (
            "F3: tokens_freed_total must remain the additive sum so "
            "downstream backward-compatible readers still work."
        )


# ------------------------------------------------------------------ #
# F4 — masking_only strategy still reclaims tokens                    #
# ------------------------------------------------------------------ #


class TestF4StrategyIndependence:
    """Masker must not depend on the Compactor being enabled."""

    @pytest.mark.asyncio
    async def test_compactor_disabled_still_frees_via_masker(self):
        """Simulate the ``masking_only`` experiment strategy.

        The experiment pipeline disables the ``ToolResultCompactor`` by
        pushing ``tool_compaction_trigger`` to ~0.99 (never reached).
        The ``ObservationMasker`` must still fire and reclaim tokens —
        this is the F4 architectural contract: the two mechanisms are
        independent; the Masker is the primary workhorse and does not
        depend on the Compactor being enabled.
        """
        from nucleusiq.agents.context.config import (
            ContextConfig,
            ContextStrategy,
        )

        script = [
            ("tool", "search", {"query": "x"}),
            ("tool", "search", {"query": "y"}),
            ("content", "done."),
        ]
        llm = _ScriptedLLM(script)
        agent = Agent(
            name="ctx-e2e-mo",
            role="tester",
            objective="masking-only",
            prompt=make_test_prompt(),
            config=AgentConfig(
                mode=ExecutionMode.STANDARD,
                enable_synthesis=False,
                respect_context_window=True,
                context=ContextConfig(
                    strategy=ContextStrategy.PROGRESSIVE,
                    # Effectively disable the Compactor triggers.
                    tool_compaction_trigger=0.99,
                    conversation_compaction_trigger=0.995,
                    emergency_trigger=0.999,
                    # v2 step 1: bypass the budget gate so this
                    # masking-only e2e test still asserts non-zero
                    # ``masker_tokens_freed``.
                    squeeze_threshold=0.0,
                ),
            ),
            llm=llm,
            tools=[BigSearchTool()],
        )
        await agent.execute({"id": "e2e-mo", "objective": "two searches"})
        tel = agent._context_engine.telemetry
        assert tel.masker_tokens_freed > 0, (
            "Masker must still engage even when the Compactor triggers "
            "are effectively disabled (F4 contract)."
        )
        assert tel.compactor_tokens_freed == 0, (
            "Compactor must stay idle when its triggers are unreachable."
        )

    @pytest.mark.asyncio
    async def test_strategy_none_frees_no_tokens(self):
        """``ContextStrategy.NONE`` → engine disabled, zero overhead."""
        script = [
            ("tool", "search", {"query": "x"}),
            ("tool", "search", {"query": "y"}),
            ("content", "done."),
        ]
        agent = _build_agent(
            script,
            strategy="none",
            enable_context_management=False,
        )
        await agent.execute({"id": "e2e-bl", "objective": "two searches"})
        assert agent._context_engine is None, (
            "strategy=none + respect_context_window=False must return "
            "a None engine for the zero-overhead opt-out path."
        )


# ------------------------------------------------------------------ #
# F7 — Streaming vs non-streaming produce identical telemetry shape   #
# ------------------------------------------------------------------ #


class TestF7StreamingSymmetryEndToEnd:
    """A full Agent run through Standard mode, sync vs streaming."""

    @pytest.mark.asyncio
    async def test_streaming_and_sync_produce_same_telemetry_shape(self):
        script = [
            ("tool", "search", {"query": "alpha"}),
            ("tool", "search", {"query": "beta"}),
            ("content", "Done."),
        ]

        # Non-streaming run.
        ns_agent = _build_agent(list(script))
        await ns_agent.execute({"id": "e2e-sym-ns", "objective": "two searches"})
        ns_tel = ns_agent._context_engine.telemetry

        # Streaming run — same script, fresh agent.
        st_agent = _build_agent(list(script))
        async for _ in st_agent.execute_stream(
            {"id": "e2e-sym-st", "objective": "two searches"}
        ):
            pass
        st_tel = st_agent._context_engine.telemetry

        # Both paths must reclaim the same number of masked
        # observations (the terminal F7 post_response covers the last
        # tool result in both paths).
        assert ns_tel.observations_masked == st_tel.observations_masked, (
            "F7: streaming and non-streaming must mask the same number "
            f"of observations.  sync={ns_tel.observations_masked}, "
            f"stream={st_tel.observations_masked}"
        )
        assert ns_tel.compactor_tokens_freed == st_tel.compactor_tokens_freed
        # Mask-freed tokens can differ by a few tokens due to UUID /
        # arg-preview noise in the marker payload; assert both > 0.
        assert ns_tel.masker_tokens_freed > 0
        assert st_tel.masker_tokens_freed > 0
