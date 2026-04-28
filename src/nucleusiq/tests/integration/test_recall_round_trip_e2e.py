"""End-to-end integration test — Context Mgmt v2 — Step 2 recall loop.

This test covers the full "retrieval, not eviction" round-trip
through a real :class:`Agent` + :class:`ContextEngine`:

    Round 1.  LLM → search(alpha)
    Round 2.  LLM → search(beta)
    Round 3.  LLM → search(gamma)
              ─► ``post_response`` for this round masks the ``alpha``
              tool message because there is now an assistant turn
              between it and the latest assistant turn.  The masked
              marker carries a ``ref:`` line.
    Round 4.  LLM → ``recall_tool_result(ref=...)`` ─► engine returns
              the full ``alpha`` content; recall is logged in
              :class:`RecallTracker`.
    Round 5.  LLM → final answer that cites the recalled evidence.

Why three searches before the recall? Because ``post_response`` runs
*before* the assistant turn is appended for the round it just
finished — the freshly-completed round's tool result has no later
assistant message yet, so the masker leaves it alone.  The masker
needs an assistant message *after* a tool result to consider the
result "consumed".  In a multi-tool conversation that means tool N
becomes maskable only at the start of round N+2.  Three searches
is the minimum that produces a masked, recall-eligible marker the
agent can then act on.

What the test asserts (the §6 + §9 invariants of
``CONTEXT_MANAGEMENT_V2_REDESIGN.md``):

* The original search content is offloaded into ``ContentStore``
  and replaced by a marker that carries a ``ref:`` line — i.e. the
  recall hint is actionable.
* The auto-injected ``recall_tool_result`` tool is callable from
  the agent loop and returns the full original content.
* :class:`RecallTracker` records the recall (``recall_count``,
  ``total_recalled_tokens``).
* Recall calls do NOT count against ``max_tool_calls`` — the
  recall round is "free" from a budget point of view (§6.4).
* The final answer comes back without errors so the recall path is
  fully integrated through the streaming/sync pipelines.

Why this is an *integration* test rather than four unit tests: the
bug surface this paper is closing (Task E refusals from gpt-5.2)
lives in the seams between Engine, Modes, and Tools.  Verifying any
one of those in isolation would not catch a regression where, say,
the masker ran but the recall tool was never injected, or where the
recall succeeded but the model never saw the result.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator

import pytest
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.agents.context.config import ContextConfig, ContextStrategy
from nucleusiq.agents.context.policy import ContextPolicy
from nucleusiq.agents.context.recall_tools import RECALL_TOOL_NAME
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.streaming.events import StreamEvent
from nucleusiq.tools import BaseTool

from nucleusiq.tests.conftest import make_test_prompt

# ====================================================================== #
# Helpers                                                                  #
# ====================================================================== #


_REF_LINE_RE = re.compile(r"^ref:\s*(\S+)\s*$", re.MULTILINE)


def _extract_first_ref(messages: list[dict]) -> str | None:
    """Pull the first ``ref:`` value out of a masked tool message.

    Used by the scripted LLM below so its "recall" step can issue a
    real ``recall_tool_result(ref=…)`` call against whatever ref the
    masker happened to mint at runtime.
    """
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content.startswith(
            "[observation consumed"
        ):
            continue
        match = _REF_LINE_RE.search(content)
        if match:
            return match.group(1)
    return None


# ====================================================================== #
# A search tool whose output is large enough to trip masker (>20 tokens) #
# ====================================================================== #


class _BigSearchTool(BaseTool):
    """Returns ~3 KB of unique content per query so we can verify
    that the *exact* original bytes (not a preview) survive recall.

    Declared ``context_policy=EVIDENCE`` explicitly so the test does
    not depend on the heuristic classifier's defaults — those defaults
    have their own unit-test coverage in ``test_policy.py``.  Here we
    are exercising the *recall round-trip*, not the heuristic.
    """

    def __init__(self) -> None:
        super().__init__(
            name="search",
            description="Search the web",
            context_policy=ContextPolicy.EVIDENCE,
        )

    async def initialize(self) -> None:
        pass

    async def execute(self, query: str) -> str:
        # ~3 KB of unique-ish bytes — large enough that masking is
        # an obvious win for the v1 token budget but, more importantly,
        # large enough that the recall must return the *real* content
        # rather than a slim preview.
        body = f"hit-{query}-" * 250
        return f"query={query}\n{body}"

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


# ====================================================================== #
# A scripted LLM that can inspect prior messages to issue a recall       #
# ====================================================================== #


class _RecallAwareScriptedLLM(BaseLLM):
    """Walks a fixed script.  One step type — ``("recall",)`` — is
    *dynamic*: it scans the inbound ``messages`` for a masked marker,
    pulls out the ``ref`` line, and emits a real
    ``recall_tool_result`` call.  Other steps are static.
    """

    def __init__(self, script: list[tuple]) -> None:
        self.model_name = "scripted-recall-llm"
        self._script = list(script)
        self._idx = 0
        self.captured_ref: str | None = None

    def _next(self) -> tuple:
        if self._idx >= len(self._script):
            return ("content", "(script exhausted)")
        step = self._script[self._idx]
        self._idx += 1
        return step

    @staticmethod
    def _make_response(
        *,
        content: str | None = None,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        call_idx: int = 0,
    ) -> Any:
        """Build a Pydantic-free OpenAI-ish response object.

        Mirrors the shape used by the existing context-management e2e
        suite so we don't have to teach the agent loop a second
        response shape.
        """

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
                self.model = "scripted-recall-llm"

        tool_calls = None
        if tool_name is not None:
            tc_dict = {
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

            tool_calls = [_TC(tc_dict)]
        return _Response([_Choice(_Msg(content, tool_calls))])

    async def call(self, **kwargs: Any) -> Any:
        step = self._next()
        kind = step[0]

        if kind == "tool":
            _, name, args = step
            return self._make_response(
                tool_name=name, tool_args=args, call_idx=self._idx
            )

        if kind == "recall":
            messages = kwargs.get("messages") or []
            ref = _extract_first_ref(messages)
            assert ref is not None, (
                "Recall step expected to find a [observation consumed] "
                "marker carrying a ref: line, but none was present.  "
                "This means the masker did not run between the two "
                "search rounds."
            )
            self.captured_ref = ref
            return self._make_response(
                tool_name=RECALL_TOOL_NAME,
                tool_args={"ref": ref},
                call_idx=self._idx,
            )

        # ("content", text)
        return self._make_response(content=step[1])

    async def call_stream(
        self, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:  # pragma: no cover — sync only
        raise NotImplementedError(
            "Recall e2e exercises the sync path; streaming has its own coverage"
        )

    def convert_tool_specs(self, tools):
        out = []
        for t in tools:
            spec = t.get_spec() if hasattr(t, "get_spec") else {}
            if spec:
                out.append(spec)
        return out


# ====================================================================== #
# Agent factory                                                            #
# ====================================================================== #


def _build_recall_agent(script: list[tuple]) -> Agent:
    """Standard-mode agent with squeeze gate disabled so masker fires."""
    llm = _RecallAwareScriptedLLM(script)
    agent = Agent(
        name="recall-e2e",
        role="tester",
        objective="exercise recall round-trip",
        prompt=make_test_prompt(),
        config=AgentConfig(
            mode=ExecutionMode.STANDARD,
            enable_synthesis=False,
            respect_context_window=True,
            context=ContextConfig(
                strategy=ContextStrategy.PROGRESSIVE,
                # v2 step 1: bypass the budget gate so the masker runs
                # on every post_response.  The default 0.70 gate keeps
                # the masker idle on tiny conversations like this one.
                squeeze_threshold=0.0,
            ),
        ),
        llm=llm,
        tools=[_BigSearchTool()],
    )
    return agent


# ====================================================================== #
# Tests                                                                    #
# ====================================================================== #


class TestRecallRoundTrip:
    """The headline test — Step 2 closes Task E's failure mode."""

    @pytest.mark.asyncio
    async def test_agent_recalls_offloaded_evidence_across_rounds(self):
        """Three searches → recall → final answer cites recalled content."""
        script = [
            ("tool", "search", {"query": "alpha"}),
            ("tool", "search", {"query": "beta"}),
            ("tool", "search", {"query": "gamma"}),
            ("recall",),  # dynamic: extracts ref from masked marker
            ("content", "Final answer based on recalled evidence."),
        ]
        agent = _build_recall_agent(script)

        result = await agent.execute(
            {"id": "recall-e2e-1", "objective": "two searches then recall"}
        )

        # The agent must finish without errors and surface the answer.
        assert "Final answer" in str(result.output), (
            "Agent did not reach the final synthesis turn — recall "
            "loop likely broken upstream."
        )

        engine = agent._context_engine
        assert engine is not None

        # ---------- 1. Masker actually offloaded round-1 search ----------
        assert engine.store.size >= 1, (
            "ContentStore must hold the offloaded round-1 search result "
            "(masker did not run)."
        )

        # ---------- 2. Recall tool was invoked with that ref -------------
        captured = agent.llm.captured_ref  # type: ignore[attr-defined]
        assert captured is not None, "Scripted LLM never issued a recall."
        assert captured in engine.store, (
            "Recall ref must point at a real ContentStore key."
        )

        # ---------- 3. RecallTracker recorded the recall -----------------
        assert engine.recall_tracker.recall_count >= 1, (
            "RecallTracker must record every recall_tool_result invocation."
        )
        assert engine.recall_tracker.total_recalled_tokens > 0, (
            "Tracker must accumulate recalled-token totals (telemetry)."
        )

        # ---------- 4. Telemetry surfaces recall counters ----------------
        tel = engine.telemetry
        assert tel.recall_count >= 1
        assert tel.recall_tokens > 0

    @pytest.mark.asyncio
    async def test_recalled_content_matches_original_bytes(self):
        """Round-trip fidelity: ``ContentStore.retrieve(ref) == original``.

        The other tests prove that the recall *path* is wired up.
        This one proves the recall *payload* is exactly what the tool
        produced — not a head-truncated preview, not a summary, but
        the literal output bytes.

        Why this matters: Task E refusals from gpt-5.2 originated in
        the v1 design where the model could only see the first ~200
        chars of an offloaded result via the (now removed) ``summary``
        slot.  The model had no way to recover the rest.  v2 step 2's
        whole pitch is "recall returns the full content"; if that
        ever silently regressed to a preview, evidence-heavy tasks
        would degrade in exactly the way Task E showed.
        """
        script = [
            ("tool", "search", {"query": "alpha"}),
            ("tool", "search", {"query": "beta"}),
            ("tool", "search", {"query": "gamma"}),
            ("recall",),
            ("content", "Done."),
        ]
        agent = _build_recall_agent(script)
        await agent.execute(
            {"id": "recall-fidelity", "objective": "verify recall fidelity"}
        )

        engine = agent._context_engine
        captured = agent.llm.captured_ref  # type: ignore[attr-defined]
        assert captured is not None and captured in engine.store

        # The store always holds the *original* bytes the tool produced.
        # Our search tool generates "query={query}\n" + "hit-{query}-"*250.
        stored = engine.store.retrieve(captured)
        assert stored is not None, "ContentStore must still hold the entry"
        assert isinstance(stored, str)

        # First search was alpha — masker masks the *oldest* consumed
        # tool result first, so the recalled ref should map to alpha.
        assert "query=alpha" in stored
        assert stored.count("hit-alpha-") == 250, (
            "Recalled content must contain every byte the tool produced "
            "(no head truncation, no summary substitution)."
        )

    @pytest.mark.asyncio
    async def test_recall_does_not_count_against_max_tool_calls(self):
        """Recall is a memory operation, not a tool action (§6.4).

        We set ``max_tool_calls=4`` and run exactly three search
        rounds, one recall round, and one final-answer round.  This is
        the *boundary* case for the §6.4 semantics:

        * If recall were charged against the budget, the loop would
          terminate after the recall (3 searches + 1 recall = 4 ==
          budget) and the final answer round would never run.
        * Because §6.4 makes recalls free, ``tool_call_count`` stays
          at 3 across the recall round, the loop continues, and the
          final answer is delivered.

        Picking ``max_tool_calls=4`` instead of a roomier number
        means a regression that re-introduces budget accounting for
        recalls would flip this test red on its very next CI run,
        rather than hiding behind generous headroom.
        """
        script = [
            ("tool", "search", {"query": "alpha"}),
            ("tool", "search", {"query": "beta"}),
            ("tool", "search", {"query": "gamma"}),
            ("recall",),  # must be *free* — doesn't count vs max_tool_calls
            ("content", "Done."),
        ]
        llm = _RecallAwareScriptedLLM(script)
        agent = Agent(
            name="recall-budget-test",
            role="tester",
            objective="prove recall is budget-free",
            prompt=make_test_prompt(),
            config=AgentConfig(
                mode=ExecutionMode.STANDARD,
                enable_synthesis=False,
                respect_context_window=True,
                max_tool_calls=4,
                context=ContextConfig(
                    strategy=ContextStrategy.PROGRESSIVE,
                    squeeze_threshold=0.0,
                ),
            ),
            llm=llm,
            tools=[_BigSearchTool()],
        )

        result = await agent.execute(
            {"id": "recall-budget", "objective": "three searches and recall"}
        )

        assert llm.captured_ref is not None, (
            "Recall must execute even when the user-tool budget "
            "(max_tool_calls=4) leaves only one slot beyond the "
            "three searches — that slot must NOT have been spent on "
            "the recall (§6.4)."
        )
        assert "Done" in str(result.output)
        assert agent._context_engine.recall_tracker.recall_count >= 1
