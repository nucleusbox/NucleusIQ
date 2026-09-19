# src/nucleusiq/agents/agent.py
"""
Agent — Thin orchestrator for NucleusIQ agents.

Routes execution to mode strategies (Direct, Standard, Autonomous)
via a pluggable registry.  All heavy logic lives in:

- ``modes/``       — execution strategies
- ``components/``  — executor, decomposer, critic, refiner, validation
- ``messaging/``   — LLM message construction
"""

import asyncio
import contextlib
import inspect
import json
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, ClassVar

from nucleusiq.agents.agent_result import (
    AbstentionSignal,
    AgentResult,
    AutonomousDetail,
    ResultStatus,
)
from nucleusiq.agents.builder.base_agent import BaseAgent
from nucleusiq.agents.components.executor import Executor
from nucleusiq.agents.config.agent_config import AgentMetrics, AgentState
from nucleusiq.agents.context.budgets import FALLBACK_WINDOW
from nucleusiq.agents.errors import AgentConfigError
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

# Mode imports
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.direct_mode import DirectMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.observability import DefaultExecutionTracer
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.structured_output.handler import StructuredOutputHandler
from nucleusiq.agents.task import Task
from nucleusiq.agents.usage.usage_tracker import UsageSummary, UsageTracker
from nucleusiq.llms.llm_params import LLMParams
from nucleusiq.plugins.base import AgentContext, BasePlugin
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.plugins.manager import PluginManager
from nucleusiq.streaming.events import StreamEvent, StreamEventType
from pydantic import Field, PrivateAttr


def _declares_window(llm: Any) -> bool:
    """True when the provider overrides ``BaseLLM.get_context_window``.

    The base implementation is a constant 128K guess; only an override
    (provider model registry, user subclass) counts as a declared window.
    """
    try:
        from nucleusiq.llms.base_llm import BaseLLM

        return (
            getattr(type(llm), "get_context_window", None)
            is not BaseLLM.get_context_window
        )
    except Exception:
        return True


class Agent(BaseAgent):
    """
    Concrete implementation of an agent in the NucleusIQ framework.

    This is a thin orchestrator that delegates execution to mode strategies
    (DirectMode, StandardMode, AutonomousMode) via a pluggable registry.

    Execution Modes (Gearbox Strategy):
    - "direct": Fast, optional tools, max 25 tool calls (Gear 1)
    - "standard": Tool-enabled loop, max 80 tool calls (Gear 2) - default
    - "autonomous": Orchestration + Critic/Refiner, max 300 tool calls (Gear 3)

    Prompt (required):
    - ``prompt`` — a ``BasePrompt`` instance defining the system message
      and optional user preamble for the LLM.  Use
      ``PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)``
      or any BasePrompt subclass.

    Labels (for logging / documentation only — NOT sent to LLM):
    - ``role`` — short human-readable label (default "Agent")
    - ``objective`` — short description of purpose

    Example::

        agent = Agent(
            name="CalculatorBot",
            role="Calculator",  # label only
            objective="Perform math ops",  # label only
            prompt=PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT).configure(
                system="You are a helpful calculator assistant.",
                user="Answer questions accurately.",
            ),
            llm=llm,
            config=AgentConfig(execution_mode="standard"),
        )
    """

    # ------------------------------------------------------------------ #
    # Mode registry (Open/Closed Principle)                               #
    # ------------------------------------------------------------------ #

    _mode_registry: ClassVar[dict[str, type[BaseExecutionMode]]] = {
        "direct": DirectMode,
        "standard": StandardMode,
        "autonomous": AutonomousMode,
    }

    @classmethod
    def register_mode(cls, name: str, mode_class: type[BaseExecutionMode]) -> None:
        """
        Register a new execution mode without modifying Agent.

        Args:
            name: Mode name (used in AgentConfig.execution_mode)
            mode_class: Class implementing BaseExecutionMode
        """
        cls._mode_registry[name] = mode_class

    # ------------------------------------------------------------------ #
    # Plugin system                                                        #
    # ------------------------------------------------------------------ #

    plugins: list[BasePlugin] = Field(
        default_factory=list,
        description="List of plugins to hook into the agent execution pipeline",
    )

    # ------------------------------------------------------------------ #
    # Private attributes (initialised in initialize())                    #
    # ------------------------------------------------------------------ #

    _executor: Executor | None = PrivateAttr(default=None)
    _plugin_manager: PluginManager | None = PrivateAttr(default=None)
    _structured_output: StructuredOutputHandler = PrivateAttr(
        default_factory=StructuredOutputHandler
    )
    _usage_tracker: UsageTracker = PrivateAttr(default_factory=UsageTracker)
    _tracer: DefaultExecutionTracer | None = PrivateAttr(default=None)
    _context_engine: Any = PrivateAttr(default=None)
    _workspace: Any = PrivateAttr(default=None)
    _evidence_dossier: Any = PrivateAttr(default=None)
    _document_corpus: Any = PrivateAttr(default=None)
    _phase_controller: Any = PrivateAttr(default=None)
    _evidence_gate: Any = PrivateAttr(default=None)
    _context_state_activator: Any = PrivateAttr(default=None)
    _last_synthesis_package: Any = PrivateAttr(default=None)
    _last_messages: list | None = PrivateAttr(default=None)
    _tool_dedup_cache: dict[tuple[str, str], str] = PrivateAttr(default_factory=dict)
    _execution_progress: Any = PrivateAttr(default=None)
    _sub_agent_context_tels: list = PrivateAttr(default_factory=list)
    # Always-on per-run diagnostics collector (see agents/diagnostics).
    _run_recorder: Any = PrivateAttr(default=None)
    _last_run_report: Any = PrivateAttr(default=None)
    # Latest ``SchemaCheck`` recorded by a mode / validation layer when
    # ``response_format`` is set (WS-8).  ``_build_result`` re-checks the
    # final output anyway; this is kept for diagnostics.
    _last_schema_check: Any = PrivateAttr(default=None)
    # Wall-clock deadline (``time.monotonic()``) for the current run.
    _run_deadline: Any = PrivateAttr(default=None)
    # Preflight fitness decision for the current run (dict) and the mode
    # actually used when preflight downgraded the requested gear.
    _preflight: Any = PrivateAttr(default=None)
    _effective_mode_value: Any = PrivateAttr(default=None)
    # Shared evidence (WS-3): a sub-agent's read-only view of its parent's
    # store / corpus / dossier, set by ``Decomposer.create_sub_agent``.
    _parent_evidence: Any = PrivateAttr(default=None)
    # Which declared ``Task.resources`` tool traffic has touched (WS-4).
    _resource_tracker: Any = PrivateAttr(default=None)
    # The single bounded coverage follow-up a run may spend (WS-4):
    # ``{"kind": "child" | "retry", ...}`` once used, else ``None``.
    _coverage_followup: Any = PrivateAttr(default=None)
    # Material handed to the generator that is neither the task nor a tool
    # result — the sub-agent findings a COMPLEX synthesis reads (I-10).
    # ``{"label", "text", "complete"}`` while set; the Critic shows it too.
    _generator_inputs: Any = PrivateAttr(default=None)
    # ExpandableTool adapters (e.g., MCPTool from nucleusiq-mcp) kept for
    # cleanup at shutdown.  See ``initialize()`` / ``_cleanup_expandable_tools``.
    _expandable_tools: list = PrivateAttr(default_factory=list)

    # ------------------------------------------------------------------ #
    # LIFECYCLE                                                           #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Initialize agent components and resources.

        Tool expansion (ExpandableTool protocol):
            Any item in ``self.tools`` that satisfies
            :class:`nucleusiq.tools.protocols.ExpandableTool` is treated
            as a *factory* — its ``connect()`` is called (in parallel
            with other adapters via ``asyncio.gather``), then
            ``expand(existing_names=...)`` returns the concrete
            :class:`BaseTool` instances that replace the factory in
            ``self.tools``.  See ``MCP_INTEGRATION_DESIGN.md`` §9.1 / §10.

            The original factory objects are retained in
            ``self._expandable_tools`` so that
            :meth:`_cleanup_expandable_tools` can disconnect them at
            shutdown.
        """
        from nucleusiq.tools.protocols import ExpandableTool

        self._logger.info(f"Initializing agent: {self.name}")

        try:
            # Initialize plugin manager
            self._plugin_manager = PluginManager(self.plugins)
            if self.plugins:
                self._logger.debug(
                    "Plugin manager initialized with %d plugins",
                    len(self.plugins),
                )

            # Phase A: Identify ExpandableTool adapters vs concrete tools.
            adapters: list[Any] = [
                t for t in self.tools if isinstance(t, ExpandableTool)
            ]
            direct: list[Any] = [
                t for t in self.tools if not isinstance(t, ExpandableTool)
            ]
            self._expandable_tools = adapters

            # Phase B: Connect all adapters in parallel.  This is the
            # major latency win when users add multiple MCP / A2A
            # servers — N × RTT becomes max(RTT).
            #
            # We use ``return_exceptions=True`` so that one failing
            # adapter cannot orphan in-flight ``connect()`` calls from
            # peers (default ``gather`` would propagate the first error
            # while leaving siblings running unattended — subprocesses,
            # HTTP connections — until they eventually fail and dangle).
            # On any failure we raise the first exception here so the
            # outer ``except`` block runs ``_cleanup_expandable_tools``,
            # which disconnects every adapter (successful or not — the
            # adapter's own ``disconnect`` is idempotent on
            # already-disconnected state).
            if adapters:
                self._logger.debug(
                    "Connecting %d expandable tool adapter(s) in parallel",
                    len(adapters),
                )
                results = await asyncio.gather(
                    *(a.connect() for a in adapters),
                    return_exceptions=True,
                )
                for adapter, res in zip(adapters, results, strict=True):
                    if isinstance(res, BaseException):
                        self._logger.error(
                            "ExpandableTool adapter %r failed to connect: %s",
                            adapter,
                            res,
                        )
                        raise res

            # Phase C: Expand each adapter into concrete BaseTool instances.
            # ``existing_names`` lets adapters detect / prefix collisions
            # consistently across the whole agent's tool registry.
            existing_names: set[str] = set()
            for t in direct:
                n = getattr(t, "name", None)
                if isinstance(n, str):
                    existing_names.add(n)
            expanded_tools: list[Any] = list(direct)
            for adapter in adapters:
                bound = await adapter.expand(existing_names=existing_names)
                for t in bound:
                    if getattr(t, "name", None) is not None:
                        existing_names.add(t.name)
                expanded_tools.extend(bound)

            # Replace ``self.tools`` with the expanded list so executors,
            # plugins, tracer, and ContextEngine all see real BaseTools.
            self.tools = expanded_tools

            # Initialize Executor component (always needed for tool execution)
            if self.llm:
                self._executor = Executor(self.llm, self.tools)
                self._logger.debug("Executor component initialized")
            else:
                self._executor = None
                self._logger.debug("Executor not initialized (no LLM)")

            # Initialize memory if provided
            if self.memory:
                await self.memory.ainitialize()
                self._logger.debug("Memory system initialized")

            # Initialize prompt if provided
            if self.prompt:
                prompt_text = self.prompt.format_prompt()
                self._logger.debug(f"Prompt system initialized \n {prompt_text}")

            # Initialize tools (expanded MCPBoundTool.initialize is a no-op
            # since the session is already connected via the adapter).
            for tool in self.tools:
                await tool.initialize()
            if self.tools:
                self._logger.debug("Initialised %d tools", len(self.tools))

            # Initialization succeeded
            self.state = AgentState.INITIALIZING
            self._logger.info("Agent initialization completed successfully")

        except BaseException as e:
            self.state = AgentState.ERROR
            self._logger.error(f"Agent initialization failed: {e!s}")
            # Best-effort cleanup of any adapters that connected before
            # the failure so we don't leak subprocesses / sessions.
            # We catch BaseException (not just Exception) so that
            # KeyboardInterrupt / CancelledError still triggers cleanup.
            try:
                await self._cleanup_expandable_tools()
            except BaseException:  # noqa: BLE001 — cleanup must never mask original
                self._logger.exception(
                    "Adapter cleanup failed during initialize rollback"
                )
            raise

    async def _cleanup_expandable_tools(self) -> None:
        """Disconnect all :class:`ExpandableTool` adapters in parallel.

        Best-effort: uses ``return_exceptions=True`` so one failing
        adapter does not block the others' shutdown.  Idempotent —
        adapters' ``disconnect()`` must tolerate being called on an
        already-disconnected session.

        Called from :meth:`initialize` on failure (to roll back partial
        state) and from the agent's shutdown / ``__aexit__`` path.
        """
        adapters = getattr(self, "_expandable_tools", None) or []
        if not adapters:
            return
        results = await asyncio.gather(
            *(a.disconnect() for a in adapters),
            return_exceptions=True,
        )
        for adapter, res in zip(adapters, results, strict=True):
            if isinstance(res, Exception):
                self._logger.warning(
                    "ExpandableTool adapter disconnect failed: %r",
                    adapter,
                    exc_info=res,
                )
        # Clear the list so a subsequent initialize starts clean.
        self._expandable_tools = []

    # ------------------------------------------------------------------ #
    # PLAN CREATION (simple default)                                      #
    # ------------------------------------------------------------------ #

    async def plan(self, task: Task | dict[str, Any]) -> Plan:
        """
        Create an execution plan for the given task.

        By default, returns a simple one-step plan that executes the task
        directly.  Override this method for custom multi-step plan creation.

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys

        Returns:
            Plan instance with steps
        """
        # Convert dict to Task if needed (backward compatibility)
        if isinstance(task, dict):
            task = Task.from_dict(task)

        # Create default one-step plan
        step = PlanStep(step=1, action="execute", task=task)
        return Plan(steps=[step], task=task)

    # ------------------------------------------------------------------ #
    # EXECUTION — thin dispatcher via mode registry                       #
    # ------------------------------------------------------------------ #

    def _resolve_llm_params(
        self,
        per_execute: LLMParams | None = None,
    ) -> dict[str, Any]:
        """
        Merge LLM parameter overrides and return a kwargs dict.

        Merge chain (highest priority wins):
            LLM defaults (in __init__) < AgentConfig.llm_params < per-execute llm_params

        Only non-None values are included in the result.

        Args:
            per_execute: Optional per-task LLM parameter overrides.

        Returns:
            Dict of merged LLM call kwargs (may be empty).
        """
        config_params = getattr(self.config, "llm_params", None)
        if config_params is None and per_execute is None:
            return {}
        if config_params is not None and per_execute is not None:
            return config_params.merge(per_execute).to_call_kwargs()
        if config_params is not None:
            return config_params.to_call_kwargs()
        assert per_execute is not None
        return per_execute.to_call_kwargs()

    # ------------------------------------------------------------------ #
    # EXECUTION LIFECYCLE — shared setup (DRY)                             #
    # ------------------------------------------------------------------ #

    def _resolve_mode(self) -> BaseExecutionMode:
        """Look up and instantiate the configured execution mode."""
        execution_mode = self.config.execution_mode
        mode_value = (
            execution_mode.value
            if hasattr(execution_mode, "value")
            else str(execution_mode)
        )
        self._logger.info(
            "Agent '%s' executing in %s mode",
            self.name,
            mode_value.upper(),
        )
        mode_class = self._mode_registry.get(mode_value)
        if not mode_class:
            raise AgentConfigError(
                f"Unknown execution mode: {execution_mode}",
                mode=mode_value,
            )
        return mode_class()

    def _create_context_engine(self) -> Any:
        """Create a ContextEngine if context management is configured.

        Returns ``None`` when context management is disabled (zero overhead).
        Auto-creates with defaults when ``respect_context_window=True``
        and ``config.context`` is ``None``.
        """
        try:
            from nucleusiq.agents.context.config import ContextConfig
            from nucleusiq.agents.context.engine import ContextEngine

            ctx_config = self.config.context

            if ctx_config is None and self.config.respect_context_window:
                mode_val = (
                    self.config.execution_mode.value
                    if hasattr(self.config.execution_mode, "value")
                    else str(self.config.execution_mode)
                )
                ctx_config = ContextConfig.for_mode(mode_val)

            if ctx_config is None or ctx_config.strategy == "none":
                return None

            max_tokens = ctx_config.max_context_tokens
            window_is_fallback = False
            if max_tokens is None and self.llm:
                try:
                    raw = self.llm.get_context_window()
                    max_tokens = int(raw) if isinstance(raw, (int, float)) else None
                except Exception:
                    max_tokens = None
                # ``BaseLLM.get_context_window`` returns a constant; only a
                # provider override is a *declared* window.
                window_is_fallback = max_tokens is None or not _declares_window(
                    self.llm
                )
            elif max_tokens is None:
                window_is_fallback = True

            counter = self._build_token_counter()

            if window_is_fallback:
                # Same last-resort number as ``BaseLLM.get_context_window``;
                # surfaced so the run report can explain a mis-sized run.
                self._logger.warning(
                    "Context window unknown for %s — assuming %d tokens. Set "
                    "ContextConfig.max_context_tokens to the model's real window.",
                    getattr(self.llm, "model_name", "llm"),
                    FALLBACK_WINDOW,
                )

            store = None
            parent_view = getattr(self, "_parent_evidence", None)
            parent_store = getattr(parent_view, "store", None)
            if parent_store is not None:
                from nucleusiq.agents.context.shared_evidence import (
                    LayeredContentStore,
                )

                store = LayeredContentStore(parent_store)

            return ContextEngine(
                config=ctx_config,
                token_counter=counter,
                max_tokens=max_tokens or FALLBACK_WINDOW,
                tracer=self._tracer,
                max_output_tokens=int(self.config.llm_max_output_tokens),
                window_is_fallback=window_is_fallback,
                store=store,
            )
        except Exception:
            self._logger.debug("Context engine creation failed, proceeding without it")
            return None

    def _inject_recall_tools_for_execution(self) -> None:
        """Append auto-injected context-management tools to ``self.tools``.

        Context Mgmt v2 — Step 2 (§6.2 of the redesign): the recall
        tools (``recall_tool_result``, ``list_recalled_evidence``)
        are auto-discovered by the model whenever a
        :class:`ContextEngine` is attached to this agent.  Discovery
        works because every LLM call serialises ``self.tools`` into
        the tool-spec list; appending here makes the tools visible
        without any explicit user wiring.

        Idempotent across executions: any pre-existing context-management tools
        from a previous ``execute()`` call are stripped first
        (because their engine/workspace binding is stale), then fresh tools
        are built against the new run state and appended.

        Executor wiring is conditional. The :class:`Executor` is
        created lazily by some execution modes (e.g. Standard's
        ``_ensure_executor``) on the *first* tool call, which happens
        after ``_setup_execution`` returns.  When ``_executor`` is
        already set we register the context tools in its tool table
        directly; when it is ``None`` we still append to
        ``self.tools`` so the lazy constructor — which iterates
        ``self.tools`` — picks them up.  Either path leaves the
        executor with the context tools available, which is the only
        invariant the agent loop cares about.
        """
        from nucleusiq.agents.context.document_corpus_tools import (
            build_document_corpus_tools,
        )
        from nucleusiq.agents.context.evidence_tools import build_evidence_tools
        from nucleusiq.agents.context.recall_tools import build_recall_tools
        from nucleusiq.agents.context.workspace_tools import (
            build_workspace_tools,
            is_context_management_tool_name,
        )

        # Always strip stale context tools (their run-local binding is tied to
        # the previous execution).
        self.tools = [
            t
            for t in self.tools
            if not is_context_management_tool_name(getattr(t, "name", None))
        ]
        if self._executor is not None:
            self._executor.tools = {
                name: tool
                for name, tool in self._executor.tools.items()
                if not is_context_management_tool_name(name)
            }

        # Context tools only make sense when the agent has user tools. Without
        # user tools, helper tools can confuse naive LLMs and simple mocks that
        # blindly pick tools[0].
        if not self.tools:
            self._logger.debug(
                "Skipping context tool injection: agent has no user tools"
            )
            return

        context_tools = []
        if self._context_engine is not None:
            context_tools.extend(build_recall_tools(self._context_engine))
        if self._workspace is not None:
            context_tools.extend(build_workspace_tools(self._workspace))
        if self._evidence_dossier is not None:
            context_tools.extend(build_evidence_tools(self._evidence_dossier))
        if self._document_corpus is not None and self._evidence_dossier is not None:
            context_tools.extend(
                build_document_corpus_tools(
                    self._document_corpus,
                    evidence=self._evidence_dossier,
                )
            )

        if not context_tools:
            return

        self.tools.extend(context_tools)
        if self._executor is not None:
            for t in context_tools:
                self._executor.tools[t.name] = t

        self._logger.debug(
            "Auto-injected %d context tool(s): %s",
            len(context_tools),
            [t.name for t in context_tools],
        )

    def _build_token_counter(self) -> Any:
        """Build a TokenCounter from the LLM's estimate_tokens method."""
        from nucleusiq.agents.context.counter import DefaultTokenCounter

        if self.llm is None:
            return DefaultTokenCounter()

        class _LLMTokenCounter:
            """Adapter: wraps BaseLLM.estimate_tokens() as a TokenCounter."""

            def __init__(self, llm: Any) -> None:
                self._llm = llm

            def count(self, text: str) -> int:
                return self._llm.estimate_tokens(text)

            def count_messages(self, messages: list) -> int:
                total = 0
                for msg in messages:
                    total += 4
                    content = msg.content if hasattr(msg, "content") else ""
                    if isinstance(content, str):
                        total += self.count(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                text = part.get("text", "")
                                if text:
                                    total += self.count(text)
                    if hasattr(msg, "name") and msg.name:
                        total += self.count(msg.name)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            total += self.count(str(tc))
                return total

        return _LLMTokenCounter(self.llm)

    @property
    def workspace(self) -> Any:
        """Run-local in-memory workspace for the current execution."""
        if self._workspace is None:
            from nucleusiq.agents.context.workspace import InMemoryWorkspace

            self._workspace = InMemoryWorkspace()
        return self._workspace

    @property
    def evidence_dossier(self) -> Any:
        """Run-local in-memory evidence dossier for the current execution."""
        if self._evidence_dossier is None:
            from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier

            self._evidence_dossier = InMemoryEvidenceDossier()
        return self._evidence_dossier

    def build_synthesis_package(
        self,
        *,
        task: str,
        output_shape: str = "",
        recalled_snippets: tuple[str, ...] = (),
        max_chars: int | None = None,
        role: str = "synthesis_package",
    ) -> Any:
        """Build a bounded synthesis package from this run's curated state.

        ``max_chars`` defaults to the window-derived budget for ``role``
        (see :class:`BudgetResolver`); pass an explicit value only to
        override it.  The package carries the run's resource coverage so
        every consumer sees, as a harness-verified fact, which declared
        resources the tools actually read.
        """
        from nucleusiq.agents.context.synthesis_package import build_synthesis_package

        if max_chars is None:
            max_chars = self._package_budget_chars(role)
        tracker = getattr(self, "_resource_tracker", None)
        coverage = None
        if tracker is not None and getattr(tracker, "resources", None):
            with contextlib.suppress(Exception):
                coverage = tracker.to_dict()
        return build_synthesis_package(
            task=task,
            output_shape=output_shape,
            workspace=self.workspace,
            evidence=self.evidence_dossier,
            recalled_snippets=recalled_snippets,
            coverage=coverage,
            max_chars=max_chars,
        )

    def _package_budget_chars(self, role: str) -> int:
        """Window-derived char budget for a synthesis-package consumer."""
        from nucleusiq.agents.context.budgets import budgets_for
        from nucleusiq.agents.context.synthesis_package import DEFAULT_PACKAGE_CHARS

        try:
            return budgets_for(self).handoff_chars(role)  # type: ignore[arg-type]
        except Exception:
            return DEFAULT_PACKAGE_CHARS

    @property
    def document_corpus(self) -> Any:
        """Run-local in-memory document corpus for L5 retrieval."""
        if self._document_corpus is None:
            from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus

            self._document_corpus = InMemoryDocumentCorpus()
        return self._document_corpus

    @property
    def phase_controller(self) -> Any:
        """Run-local phase telemetry controller for L6."""
        if self._phase_controller is None:
            from nucleusiq.agents.context.phase_control import PhaseController

            self._phase_controller = PhaseController()
        return self._phase_controller

    @property
    def evidence_gate(self) -> Any:
        """Run-local evidence completeness gate for L6."""
        if self._evidence_gate is None:
            from nucleusiq.agents.context.phase_control import EvidenceGate

            self._evidence_gate = EvidenceGate(
                required_tags=tuple(self.config.evidence_gate_required_tags),
                enforce=self.config.evidence_gate_enforce,
            )
        return self._evidence_gate

    def _has_context_state(self) -> bool:
        """Return True when workspace/evidence has state worth packaging."""
        try:
            if self.workspace.stats().entry_count > 0:
                return True
        except Exception:
            pass
        try:
            if self.evidence_dossier.stats().item_count > 0:
                return True
        except Exception:
            pass
        return False

    def _build_synthesis_messages_from_context(
        self,
        *,
        task: str,
        output_shape: str = "",
        max_chars: int | None = None,
        role: str = "synthesis_package",
    ) -> list[Any] | None:
        """Build package-based synthesis messages when curated state exists.

        The package is sized for ``role`` through :class:`BudgetResolver`
        unless ``max_chars`` is given.  When the package had to omit items
        its own visibility note is prepended so the consumer never mistakes
        a cut list for a complete one.
        """
        if not self._has_context_state():
            return None

        from nucleusiq.agents.chat_models import ChatMessage

        package = self.build_synthesis_package(
            task=task,
            output_shape=output_shape,
            max_chars=max_chars,
            role=role,
        )
        self._last_synthesis_package = package
        recorder = getattr(self, "_run_recorder", None)
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.record_decision(
                    f"package:{role}",
                    {
                        "chars": package.metadata.get("char_count"),
                        "max_chars": package.metadata.get("max_chars"),
                        "complete": package.metadata.get("complete"),
                        "omitted_items": {
                            k: v
                            for k, v in (
                                package.metadata.get("omitted_items") or {}
                            ).items()
                            if v
                        },
                        "omitted_sections": list(
                            package.metadata.get("omitted_sections") or []
                        ),
                    },
                )
        phase_controller = getattr(self, "_phase_controller", None)
        if phase_controller is not None:
            phase_controller.enter("ORGANIZE_EVIDENCE")
            phase_controller.enter("SYNTHESIZE")
            evidence_gate = getattr(self, "_evidence_gate", None)
            if evidence_gate is not None:
                try:
                    decision = evidence_gate.evaluate(
                        self.evidence_dossier,
                        record_gaps=bool(evidence_gate.required_tags),
                    )
                    phase_controller.record_evidence_gate(decision)
                except Exception:
                    pass
        activator = getattr(self, "_context_state_activator", None)
        if activator is not None:
            activator.metrics.synthesis_package_used = True
            activator.metrics.synthesis_package_char_count = package.metadata.get(
                "char_count", len(package.text)
            )
        if phase_controller is not None:
            phase_controller.synthesis_used_package = True

        note = package.visibility_note()
        preamble = f"{note}\n\n" if note else ""
        return [
            ChatMessage(
                role="user",
                content=(
                    f"{preamble}{package.text}\n\n"
                    "Using only the curated package above, produce the complete "
                    "final answer requested by the task. Clearly qualify any known gaps."
                ),
            )
        ]

    def _activate_context_state_for_tool_result(
        self,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
        tool_result: Any,
        tool_args: dict[str, Any] | None = None,
    ) -> None:
        """Internal L4.5 route from business tool result to context state."""
        tracker = getattr(self, "_resource_tracker", None)
        if tracker is not None:
            with contextlib.suppress(Exception):
                tracker.observe(
                    tool_name=tool_name, tool_args=tool_args, tool_result=tool_result
                )
        activator = getattr(self, "_context_state_activator", None)
        if activator is None:
            return
        phase_controller = getattr(self, "_phase_controller", None)
        if phase_controller is not None:
            phase_controller.enter("RESEARCH")
        try:
            activator.activate_tool_result(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_result=tool_result,
                tool_args=tool_args,
            )
        except Exception as exc:
            self._logger.debug("Context state activation skipped: %s", exc)

    async def _setup_execution(
        self,
        task: Task | dict[str, Any],
        llm_params: LLMParams | None = None,
    ) -> tuple:
        """Shared lifecycle setup for ``execute()`` and ``execute_stream()``.

        Steps:
            1. Convert dict → Task
            2. Resolve merged LLM params
            3. Set current task
            4. Ensure plugin manager + reset counters
            5. Reset usage tracker and execution tracer for this run
            6. Run BEFORE_AGENT hook (may raise ``PluginHalt``)
            7. Validate tool count against mode limit
            8. Resolve execution mode

        Returns:
            ``(task, mode, agent_ctx)``

        Raises:
            PluginHalt: If a plugin aborts execution early.
            ValueError: If tool count exceeds mode limit or mode is unknown.
        """
        if isinstance(task, dict):
            task = Task.from_dict(task)

        self._current_llm_overrides = self._resolve_llm_params(per_execute=llm_params)
        self._logger.debug("Starting execution for task %s", task.id)
        self._current_task = task.to_dict()

        if self._plugin_manager is None:
            self._plugin_manager = PluginManager(self.plugins)
        self._plugin_manager.reset_counters()

        self._usage_tracker.reset()
        self._tracer = (
            DefaultExecutionTracer() if self.config.effective_tracing else None
        )
        from nucleusiq.agents.diagnostics.run_report import RunRecorder

        self._run_recorder = RunRecorder()
        self._last_schema_check = None
        self._effective_mode_value = None
        self._preflight = None
        # Wall clock (WS-6 6.3): monotonic deadline checked at every loop
        # boundary.  ``0`` means unlimited.
        budget_s = int(getattr(self.config, "max_execution_time", 0) or 0)
        self._run_deadline = time.monotonic() + budget_s if budget_s > 0 else None

        if self._plugin_manager is not None and self._tracer is not None:
            self._plugin_manager._tracer = self._tracer

        # Context window management — create ContextEngine if configured
        self._context_engine = self._create_context_engine()
        self._sub_agent_context_tels = []
        from nucleusiq.agents.context.document_search import InMemoryDocumentCorpus
        from nucleusiq.agents.context.evidence import InMemoryEvidenceDossier
        from nucleusiq.agents.context.phase_control import EvidenceGate, PhaseController
        from nucleusiq.agents.context.state_activator import ContextStateActivator
        from nucleusiq.agents.context.workspace import InMemoryWorkspace

        self._workspace = InMemoryWorkspace()
        parent_view = getattr(self, "_parent_evidence", None)
        if parent_view is not None:
            # Sub-agent: read through to the parent's evidence, write locally.
            from nucleusiq.agents.context.shared_evidence import (
                LayeredDocumentCorpus,
                LayeredEvidenceDossier,
            )

            self._evidence_dossier = LayeredEvidenceDossier(
                getattr(parent_view, "dossier", None)
            )
            self._document_corpus = LayeredDocumentCorpus(
                getattr(parent_view, "corpus", None)
            )
        else:
            self._evidence_dossier = InMemoryEvidenceDossier()
            self._document_corpus = InMemoryDocumentCorpus()
        from nucleusiq.agents.context.coverage import ResourceTouchTracker

        self._resource_tracker = ResourceTouchTracker(task.effective_resources())
        self._coverage_followup = None
        self._generator_inputs = None
        self._phase_controller = PhaseController()
        self._evidence_gate = EvidenceGate(
            required_tags=tuple(self.config.evidence_gate_required_tags),
            enforce=self.config.evidence_gate_enforce,
        )
        self._phase_controller.enter("PLAN")
        self._context_state_activator = ContextStateActivator(
            workspace=self._workspace,
            evidence=self._evidence_dossier,
            document_corpus=self._document_corpus,
            required_tags=tuple(self.config.evidence_gate_required_tags),
            max_corpus_index_chars=self.config.context_tool_result_corpus_max_chars,
            ingest_min_chars=self.config.context_activation_ingest_min_chars,
        )
        self._last_synthesis_package = None

        # Context Mgmt v2 — Step 2: auto-inject the recall tools so
        # the model can rehydrate offloaded evidence on demand.  The
        # tools are bound to the engine just created above; on a
        # subsequent execute() call the engine is replaced and we
        # re-inject fresh tools.  This is the only place that mutates
        # ``self.tools`` after construction, kept here so the
        # max-tools check below sees the final list.
        self._inject_recall_tools_for_execution()

        agent_ctx = AgentContext(
            agent_name=self.name,
            task=task,
            state=self.state,
            config=self.config,
            memory=self.memory,
        )
        agent_ctx = await self._plugin_manager.run_before_agent(agent_ctx)

        max_tools = self.config.get_effective_max_tool_calls()
        # Auto-injected context-management tools must not count against the user's
        # tool budget — the user did not opt into them, the framework
        # added them.  See ``workspace_tools.is_context_management_tool_name`` for the
        # canonical list.
        from nucleusiq.agents.context.workspace_tools import (
            is_context_management_tool_name,
        )

        user_tool_count = sum(
            1
            for t in self.tools
            if not is_context_management_tool_name(getattr(t, "name", None))
        )
        if user_tool_count > max_tools:
            mode_value = (
                self.config.execution_mode.value
                if hasattr(self.config.execution_mode, "value")
                else str(self.config.execution_mode)
            )
            raise AgentConfigError(
                f"Agent '{self.name}' has {user_tool_count} tools but "
                f"{mode_value.upper()} mode allows max {max_tools}. "
                f"Reduce tools or switch to a higher execution mode.",
                mode=mode_value,
            )

        mode = self._resolve_mode()
        mode = self._run_preflight(mode)
        return task, mode, agent_ctx

    # ------------------------------------------------------------------ #
    # Preflight fitness (WS-6 6.7)                                         #
    # ------------------------------------------------------------------ #

    def _measure_fixed_costs(self) -> tuple[int, int]:
        """Tokens every call pays before any evidence: system prompt + tool schemas."""
        engine = self._context_engine
        counter = getattr(engine, "token_counter", None)
        if counter is None:
            counter = self._build_token_counter()

        def _count(text: str) -> int:
            try:
                return int(counter.count(text))
            except Exception:
                return max(1, len(text) // 4)

        system_text = ""
        prompt = getattr(self, "prompt", None)
        for attr in ("system", "system_prompt"):
            value = getattr(prompt, attr, None)
            if isinstance(value, str) and value.strip():
                system_text = value
                break
        system_text += f"\n{self.role or ''}\n{self.objective or ''}"
        system_tokens = _count(system_text)

        tool_tokens = 0
        for tool in self.tools or []:
            spec_fn = getattr(tool, "get_spec", None)
            if not callable(spec_fn):
                continue
            try:
                tool_tokens += _count(json.dumps(spec_fn(), default=str))
            except Exception:
                tool_tokens += 50
        return system_tokens, tool_tokens

    def _run_preflight(self, mode: BaseExecutionMode) -> BaseExecutionMode:
        """Decide whether the resolved window can carry the requested gear.

        ``working = window − response_reserve − system − tool_schemas`` is
        what a single call can spend on evidence and reasoning.  Below
        ``preflight_min_working_tokens`` Autonomous mode cannot fit its
        Critic/Refiner hand-offs or sub-agent synthesis, so the run is
        downgraded to STANDARD (unless ``preflight_downgrade=False``).
        The numbers always land in the run report.
        """
        engine = self._context_engine
        if engine is None:
            return mode
        try:
            system_tokens, tool_tokens = self._measure_fixed_costs()
            engine.set_fixed_costs(
                system_tokens=system_tokens, tool_schema_tokens=tool_tokens
            )
            budgets = engine.budgets
        except Exception as exc:
            self._logger.debug("Preflight skipped: %s", exc)
            return mode

        mode_value = (
            self.config.execution_mode.value
            if hasattr(self.config.execution_mode, "value")
            else str(self.config.execution_mode)
        )
        working = budgets.usable_tokens
        cfg = self.config
        preflight: dict[str, Any] = {
            "mode_requested": mode_value,
            "window": budgets.window,
            "window_is_fallback": bool(getattr(engine, "window_is_fallback", False)),
            "response_reserve": budgets.response_reserve,
            "system_tokens": system_tokens,
            "tool_schema_tokens": tool_tokens,
            "working_tokens": working,
            "fitness": "ok",
            "action": "none",
        }

        if mode_value == "autonomous":
            min_ok = int(cfg.preflight_min_working_tokens)
            marginal = int(cfg.preflight_marginal_working_tokens)
            if min_ok > 0 and working < min_ok:
                preflight["fitness"] = "unfit"
                breakdown = (
                    f"window {budgets.window} − reserve {budgets.response_reserve} − "
                    f"system {system_tokens} − tool schemas {tool_tokens} = "
                    f"{working} working tokens (< {min_ok})"
                )
                if cfg.preflight_downgrade:
                    preflight["action"] = "downgraded_to_standard"
                    self._logger.warning(
                        "Preflight: %s. Autonomous mode cannot fit its Critic/"
                        "Refiner hand-offs in this window — running in STANDARD "
                        "mode instead (set preflight_downgrade=False to force).",
                        breakdown,
                    )
                    mode = StandardMode()
                    self._effective_mode_value = "standard"
                    recorder = getattr(self, "_run_recorder", None)
                    if recorder is not None:
                        with contextlib.suppress(Exception):
                            recorder.record_event("preflight_downgraded", breakdown)
                else:
                    preflight["action"] = "forced"
                    self._logger.warning(
                        "Preflight: %s. Autonomous mode forced "
                        "(preflight_downgrade=False); expect heavy compaction.",
                        breakdown,
                    )
            elif marginal > 0 and working < marginal:
                preflight["fitness"] = "marginal"
                self._logger.info(
                    "Preflight: %d working tokens (window %d, reserve %d, system %d, "
                    "tool schemas %d) — Autonomous mode will run with tight hand-off "
                    "budgets.",
                    working,
                    budgets.window,
                    budgets.response_reserve,
                    system_tokens,
                    tool_tokens,
                )
        elif mode_value == "standard":
            floor = int(cfg.preflight_standard_min_working_tokens)
            if floor > 0 and working < floor:
                preflight["fitness"] = "marginal"
                self._logger.warning(
                    "Preflight: only %d working tokens per call (window %d, reserve "
                    "%d, system %d, tool schemas %d). Reduce tools or use a larger "
                    "model window.",
                    working,
                    budgets.window,
                    budgets.response_reserve,
                    system_tokens,
                    tool_tokens,
                )

        self._preflight = preflight
        recorder = getattr(self, "_run_recorder", None)
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.record_decision("preflight", dict(preflight))
        return mode

    # ------------------------------------------------------------------ #
    # EXECUTION — non-streaming                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_framework_error_output(output: Any) -> bool:
        """Return True for legacy mode error-string sentinels."""
        return isinstance(output, str) and output.strip().startswith("Error:")

    @staticmethod
    def _classify_framework_error(output: Any) -> str:
        """Map legacy error-string sentinels to stable ``AgentResult`` types."""
        text = str(output)
        if "Maximum tool calls" in text:
            return "ToolCallLimitError"
        if "LLM did not respond" in text:
            return "EmptyLLMResponseError"
        if "Tool '" in text and "execution failed" in text:
            return "ToolExecutionError"
        return "AgentRuntimeError"

    async def execute(
        self,
        task: Task | dict[str, Any],
        llm_params: LLMParams | None = None,
    ) -> AgentResult:
        """Execute a task using the agent's capabilities.

        Execution Flow (Gearbox Strategy):
        - Direct mode: Fast, optional tools (max 25 tool calls)
        - Standard mode: Tool-enabled loop (max 80 tool calls) — default
        - Autonomous mode: Orchestration + Critic/Refiner (max 300 tool calls)

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys
            llm_params: Optional type-safe per-task LLM parameter overrides.
                Accepts :class:`LLMParams` or any provider subclass
                (e.g. ``OpenAILLMParams``).  These override both the LLM-level
                defaults and the ``AgentConfig.llm_params`` for this single
                execution only.

        Returns:
            :class:`AgentResult` — immutable execution result. Backward
            compatible: ``str(result)`` returns the output text.
        """
        t0 = time.perf_counter()
        task_obj: Task | None = None

        try:
            try:
                task_obj, mode, agent_ctx = await self._setup_execution(
                    task, llm_params
                )
            except PluginHalt as halt:
                status = ResultStatus.HALTED
                output = halt.result
                if task_obj is None:
                    task_obj = task if isinstance(task, Task) else Task.from_dict(task)
                self._record_termination("plugin_halt", "halted before execution")
                return self._build_result(task_obj, status, output, None, None, t0)

            status = ResultStatus.SUCCESS
            abstention_reason: str | None = None
            abstention_code: str | None = None
            try:
                output = await mode.run(self, task_obj)
            except PluginHalt as halt:
                status = ResultStatus.HALTED
                output = halt.result
                self._record_termination("plugin_halt", "plugin halted execution")
            except AbstentionSignal as signal:
                # F2: Autonomous mode exhausted retries with Critic still
                # failing. Surface as a first-class outcome rather than
                # silently returning a bad answer.
                # F5: also carry the machine-readable abstain_reason so
                # programmatic callers can react without string-matching
                # free-form feedback.
                status = ResultStatus.ABSTAINED
                output = signal.best_candidate
                abstention_reason = signal.reason
                abstention_code = getattr(signal, "abstain_reason", None)
                self._record_termination(
                    "critic_abstain", f"{abstention_code or ''} {signal.reason}".strip()
                )

            if self._plugin_manager is not None:
                output = await self._plugin_manager.run_after_agent(agent_ctx, output)

            error: str | None = None
            error_type: str | None = None
            if (
                status == ResultStatus.SUCCESS
                and self.state == AgentState.ERROR
                and self._is_framework_error_output(output)
            ):
                status = ResultStatus.ERROR
                error = str(output)
                error_type = self._classify_framework_error(output)

            return self._build_result(
                task_obj,
                status,
                output,
                error,
                error_type,
                t0,
                abstention_reason,
                abstention_code,
            )

        except Exception as exc:
            if task_obj is None:
                task_obj = task if isinstance(task, Task) else Task.from_dict(task)
            self._record_termination("error", f"{type(exc).__name__}: {exc}")
            return self._build_result(
                task_obj,
                ResultStatus.ERROR,
                None,
                str(exc),
                type(exc).__name__,
                t0,
            )
        finally:
            self._current_llm_overrides = {}

    # ------------------------------------------------------------------ #
    # Diagnostics helpers                                                  #
    # ------------------------------------------------------------------ #

    # Reasons a mode sets *before* raising; the generic ``error`` recorded
    # by ``execute()``'s catch-all must not paper over them.
    _STICKY_TERMINATIONS: ClassVar[frozenset[str]] = frozenset(
        {"context_overflow", "deadline", "llm_timeout"}
    )

    def _record_termination(self, reason: str, message: str = "") -> None:
        recorder = getattr(self, "_run_recorder", None)
        if recorder is None:
            return
        with contextlib.suppress(Exception):
            if reason == "error":
                current = getattr(recorder, "termination_reason", None)
                current_value = getattr(current, "value", current)
                if current_value in self._STICKY_TERMINATIONS:
                    return
            recorder.set_termination(reason, message)

    @property
    def last_run_report(self) -> Any:
        """The :class:`RunReport` of the most recent ``execute()`` (or ``None``)."""
        return self._last_run_report

    def _resolve_structured_result(
        self, output: Any, status: ResultStatus
    ) -> tuple[Any, dict[str, Any] | None]:
        """Type the final output against ``response_format`` (WS-8).

        Returns ``(parsed, structured)``.  ``parsed`` is the validated
        schema instance or ``None``; ``structured`` is the verdict dict,
        or ``None`` when no ``response_format`` is configured.  A final
        answer that still fails the schema after the finalizer records
        ``termination_reason="schema_invalid"`` — the caller keeps the
        raw text in ``result.output`` and can decide what to do.
        """
        if self.response_format is None:
            return None, None
        from nucleusiq.agents.modes.base_mode import BaseExecutionMode

        contract = BaseExecutionMode.structured_contract(self)
        if contract is None:
            return None, None
        if status in (ResultStatus.ERROR, ResultStatus.HALTED):
            return None, None

        payload = output
        if isinstance(output, dict) and "schema" in output and "mode" in output:
            payload = output.get("output")
        try:
            check = contract.check(payload)
        except Exception as exc:
            self._logger.debug("Schema check skipped: %s", exc)
            return None, None
        self._last_schema_check = check

        recorder = getattr(self, "_run_recorder", None)
        finalizer_runs = 0
        if recorder is not None:
            with contextlib.suppress(Exception):
                finalizer_runs = int(recorder.counters.finalizer_runs)
        structured = contract.describe(check, finalizer_runs=finalizer_runs)
        if not check.valid:
            self._logger.warning(
                "Final output does not satisfy schema %s: %s",
                contract.schema_name,
                (check.errors or "")[:200],
            )
            if status == ResultStatus.SUCCESS:
                self._record_termination(
                    "schema_invalid",
                    f"Final output failed schema {contract.schema_name}: "
                    f"{(check.errors or '')[:200]}",
                )
        return (check.value if check.valid else None), structured

    def _resolved_config_for_report(self) -> dict[str, Any]:
        """Snapshot of the numbers that actually governed this run."""
        cfg = self.config
        out: dict[str, Any] = {
            "mode": (
                cfg.execution_mode.value
                if hasattr(cfg.execution_mode, "value")
                else str(cfg.execution_mode)
            ),
            "max_tool_calls": cfg.get_effective_max_tool_calls(),
            "max_context_tool_calls": cfg.get_effective_max_context_tool_calls(),
            "max_retries": cfg.max_retries,
            "llm_max_output_tokens": cfg.llm_max_output_tokens,
            "enable_decomposition": getattr(cfg, "enable_decomposition", True),
            "max_sub_agents": cfg.max_sub_agents,
            "enable_synthesis": cfg.enable_synthesis,
            "tracing": bool(cfg.effective_tracing),
            "response_format_set": self.response_format is not None,
        }
        from nucleusiq.agents.context.workspace_tools import (
            is_context_management_tool_name,
        )

        user_tools = [
            t
            for t in (self.tools or [])
            if not is_context_management_tool_name(getattr(t, "name", None))
        ]
        out["tool_count"] = len(user_tools)
        out["idempotent_tool_count"] = sum(
            1 for t in user_tools if bool(getattr(t, "idempotent", False))
        )
        engine = getattr(self, "_context_engine", None)
        if engine is not None:
            with contextlib.suppress(Exception):
                out["context_window"] = int(engine.resolved_max_tokens)
            try:
                ecfg = engine.config
                out["response_reserve"] = int(
                    getattr(engine, "resolved_response_reserve", ecfg.response_reserve)
                )
                strategy = getattr(ecfg, "strategy", None)
                out["strategy"] = (
                    strategy.value if hasattr(strategy, "value") else str(strategy)
                )
            except Exception:
                pass
            with contextlib.suppress(Exception):
                out["optimal_budget"] = int(engine.resolved_optimal_budget)
            with contextlib.suppress(Exception):
                out["window_is_fallback"] = bool(engine.window_is_fallback)
            with contextlib.suppress(Exception):
                out["budgets"] = engine.budgets.to_dict()
        elif self.llm is not None:
            try:
                out["context_window"] = int(self.llm.get_context_window())
                out["strategy"] = "none"
            except Exception:
                pass
        out["max_execution_time"] = int(getattr(cfg, "max_execution_time", 0) or 0)
        for field in ("llm_call_timeout", "step_timeout"):
            if field in getattr(cfg, "model_fields_set", set()):
                out[field] = getattr(cfg, field)
        preflight = getattr(self, "_preflight", None)
        if isinstance(preflight, dict):
            out["preflight"] = dict(preflight)
        return out

    def _finalize_coverage(self, output: Any) -> Any:
        """Reconcile ``Task.resources`` against what the run touched (WS-4).

        Returns a ``CoverageReport`` (also stored on the run recorder) or
        ``None`` when the task declared no resources.
        """
        tracker = getattr(self, "_resource_tracker", None)
        if tracker is None or not getattr(tracker, "resources", None):
            return None
        try:
            from nucleusiq.agents.context.coverage import build_coverage

            corpus = getattr(self, "_document_corpus", None)
            doc_ids: list[str] = []
            lister = getattr(corpus, "list_documents", None)
            if callable(lister):
                with contextlib.suppress(Exception):
                    doc_ids = [str(getattr(d, "id", "")) for d in lister()]
            text = output.get("output") if isinstance(output, dict) else output
            coverage = build_coverage(
                tracker,
                answer_text=str(text) if text is not None else "",
                followup=getattr(self, "_coverage_followup", None),
                enforce=bool(getattr(self.config, "evidence_gate_enforce", False)),
                corpus_document_ids=doc_ids,
            )
        except Exception as exc:
            self._logger.debug("Coverage reconciliation skipped: %s", exc)
            return None
        recorder = getattr(self, "_run_recorder", None)
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.coverage = coverage.to_dict()
            if coverage.unprocessed:
                with contextlib.suppress(Exception):
                    recorder.record_event(
                        "coverage_gap",
                        f"{len(coverage.unprocessed)}/{len(coverage.resources)} "
                        "resource(s) unprocessed: "
                        + ", ".join(coverage.unprocessed[:5]),
                    )
        return coverage

    def _build_run_report(
        self, task: Task, status: ResultStatus, mode_value: str, model_name: str | None
    ) -> Any:
        from nucleusiq.agents.diagnostics.run_report import (
            RunRecorder,
            TerminationReason,
        )

        recorder = getattr(self, "_run_recorder", None)
        if not isinstance(recorder, RunRecorder):
            return None

        # Compaction counters come from the engine's telemetry so the
        # report agrees with ``context_telemetry`` byte for byte.
        engine = getattr(self, "_context_engine", None)
        if engine is not None:
            try:
                tel = engine.telemetry
                by: dict[str, int] = {}
                for ev in getattr(tel, "compaction_events", ()) or ():
                    name = getattr(ev, "strategy", "unknown")
                    by[name] = by.get(name, 0) + 1
                recorder.counters.compactions_by_strategy = by
                recorder.counters.emergency_count = int(
                    getattr(engine, "emergency_count", 0) or 0
                )
                recorder.counters.unseen_evidence_evicted = sum(
                    int(getattr(ev, "unseen_evicted", 0) or 0)
                    for ev in getattr(tel, "compaction_events", ()) or ()
                )
            except Exception:
                pass

        default_reason = {
            ResultStatus.SUCCESS: TerminationReason.COMPLETED,
            ResultStatus.ERROR: TerminationReason.ERROR,
            ResultStatus.HALTED: TerminationReason.PLUGIN_HALT,
            ResultStatus.ABSTAINED: TerminationReason.CRITIC_ABSTAIN,
        }.get(status, TerminationReason.COMPLETED)

        try:
            from nucleusiq import __version__ as framework_version
        except Exception:
            framework_version = ""

        try:
            from nucleusiq.agents.structured_output.resolver import (
                get_provider_from_llm,
            )

            provider = get_provider_from_llm(self.llm) if self.llm else None
        except Exception:
            provider = None

        try:
            report = recorder.build(
                framework_version=str(framework_version),
                provider=provider,
                model=model_name,
                mode=mode_value,
                agent_name=self.name,
                task_id=task.id,
                status=status.value,
                config_resolved=self._resolved_config_for_report(),
                default_reason=default_reason,
            )
        except Exception as exc:
            self._logger.debug("Run report build skipped: %s", exc)
            return None

        # Analyzer is pure and never raises, but stay defensive: a report
        # without findings is still worth more than no report.
        try:
            from nucleusiq.agents.diagnostics.analyzer import attach_findings

            report = attach_findings(report)
        except Exception as exc:
            self._logger.debug("Run report analysis skipped: %s", exc)
        return report

    def _build_result(
        self,
        task: Task,
        status: ResultStatus,
        output: Any,
        error: str | None,
        error_type: str | None,
        t0: float,
        abstention_reason: str | None = None,
        abstention_code: str | None = None,
    ) -> AgentResult:
        """Construct a frozen :class:`AgentResult` from execution data."""
        from nucleusiq.agents.agent_result import MemorySnapshot

        mode_value = (
            self.config.execution_mode.value
            if hasattr(self.config.execution_mode, "value")
            else str(self.config.execution_mode)
        )
        # Preflight may have downgraded the gear; the result says which
        # one actually ran (the request is in ``diagnostics.decisions``).
        effective = getattr(self, "_effective_mode_value", None)
        if isinstance(effective, str) and effective:
            mode_value = effective
        model_name: str | None = None
        if self.llm is not None:
            model_name = getattr(self.llm, "model", None) or getattr(
                self.llm, "model_name", None
            )

        usage_dict: dict[str, Any] | None = None
        try:
            usage_dict = self._usage_tracker.summary.summary()
        except Exception:
            pass

        tracer = getattr(self, "_tracer", None)

        if tracer is not None and self.memory is not None:
            try:
                strategy_name = type(self.memory).__name__
                messages_raw = getattr(self.memory, "messages", [])
                msg_count = len(messages_raw) if messages_raw else 0
                token_count = getattr(self.memory, "token_count", None)
                messages_snapshot: tuple[dict[str, str], ...] = ()
                if messages_raw:
                    messages_snapshot = tuple(
                        {
                            "role": getattr(m, "role", "unknown"),
                            "content": str(getattr(m, "content", ""))[:200],
                        }
                        for m in messages_raw[-10:]
                    )
                tracer.set_memory_snapshot(
                    MemorySnapshot(
                        strategy=strategy_name,
                        message_count=msg_count,
                        token_count=token_count,
                        messages=messages_snapshot,
                    )
                )
            except Exception:
                pass

        tool_calls_t: tuple = ()
        llm_calls_t: tuple = ()
        plugin_events_t: tuple = ()
        warnings_t: tuple = ()
        memory_snap = None
        autonomous_out: AutonomousDetail | None = None

        if tracer is not None:
            tool_calls_t = tuple(tracer.tool_calls)
            llm_calls_t = tuple(tracer.llm_calls)
            plugin_events_t = tuple(tracer.plugin_events)
            warnings_t = tuple(tracer.warnings)
            memory_snap = tracer.memory_snapshot
            ad = tracer.autonomous_detail
            if ad:
                try:
                    autonomous_out = AutonomousDetail.model_validate(ad)
                except Exception:
                    autonomous_out = None

        # Context telemetry (merge sub-agent telemetries for autonomous mode)
        context_tel = None
        engine = getattr(self, "_context_engine", None)
        if engine is not None:
            try:
                context_tel = engine.telemetry
            except Exception:
                pass

        sub_tels = getattr(self, "_sub_agent_context_tels", [])
        if sub_tels:
            try:
                from nucleusiq.agents.context.telemetry import ContextTelemetry

                context_tel = ContextTelemetry.merge(context_tel, sub_tels)
            except Exception:
                pass

        metadata: dict[str, Any] = {}
        phase_controller = getattr(self, "_phase_controller", None)
        if phase_controller is not None:
            try:
                phase_controller.finish()
            except Exception:
                pass
        workspace = getattr(self, "_workspace", None)
        if workspace is not None:
            try:
                metadata["workspace"] = workspace.stats().to_dict()
            except Exception:
                pass
        evidence_dossier = getattr(self, "_evidence_dossier", None)
        if evidence_dossier is not None:
            try:
                metadata["evidence"] = evidence_dossier.stats().to_dict()
            except Exception:
                pass
        document_corpus = getattr(self, "_document_corpus", None)
        if document_corpus is not None:
            try:
                metadata["document_search"] = document_corpus.stats().to_dict()
            except Exception:
                pass
        if phase_controller is not None:
            try:
                metadata["phase_control"] = phase_controller.stats().to_dict()
            except Exception:
                pass
        activator = getattr(self, "_context_state_activator", None)
        if activator is not None:
            try:
                metadata["context_activation"] = activator.metrics.to_dict()
            except Exception:
                pass
        package = getattr(self, "_last_synthesis_package", None)
        if package is not None:
            try:
                metadata["synthesis_package"] = dict(package.metadata)
            except Exception:
                pass

        parsed, structured = self._resolve_structured_result(output, status)
        if structured is not None:
            metadata["raw_output"] = (
                output.get("output") if isinstance(output, dict) else output
            )

        coverage = self._finalize_coverage(output)
        if coverage is not None:
            metadata["coverage"] = coverage.to_dict()
            if coverage.blocked and status == ResultStatus.SUCCESS:
                # ``evidence_gate_enforce``: an answer that leaves declared
                # resources unprocessed is not certified.  The output is
                # kept so the caller can still inspect it.
                status = ResultStatus.ABSTAINED
                abstention_code = abstention_code or "coverage_incomplete"
                abstention_reason = abstention_reason or (
                    f"{len(coverage.unprocessed)} declared resource(s) were "
                    "never processed: " + ", ".join(coverage.unprocessed[:5])
                )
                self._logger.warning("Coverage gate blocked: %s", abstention_reason)

        report = self._build_run_report(task, status, mode_value, model_name)
        self._last_run_report = report
        termination_reason: str | None = None
        if report is not None:
            termination_reason = report.termination.reason.value
            self._logger.info(
                "Run ended: reason=%s rounds=%d llm_calls=%d tool_calls=%d "
                "(context %d, dedup %d, recall_errors %d) compactions=%d "
                "emergency=%d findings=%s",
                termination_reason,
                report.counters.rounds,
                report.counters.llm_calls,
                report.counters.tool_calls_business,
                report.counters.tool_calls_context,
                report.counters.dedup_banners,
                report.counters.recall_errors,
                sum(report.counters.compactions_by_strategy.values()),
                report.counters.emergency_count,
                [f.code for f in report.findings] or "none",
            )

        return AgentResult(
            agent_id=str(self.id),
            agent_name=self.name,
            task_id=task.id,
            mode=mode_value,
            model=model_name,
            output=output,
            status=status,
            error=error,
            error_type=error_type,
            duration_ms=(time.perf_counter() - t0) * 1000,
            abstention_reason=abstention_reason,
            abstention_code=abstention_code,
            termination_reason=termination_reason,
            parsed=parsed,
            structured=structured,
            usage=usage_dict,
            tool_calls=tool_calls_t,
            llm_calls=llm_calls_t,
            plugin_events=plugin_events_t,
            memory_snapshot=memory_snap,
            autonomous=autonomous_out,
            context_telemetry=context_tel,
            diagnostics=report,
            warnings=warnings_t,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # EXECUTION — streaming                                                #
    # ------------------------------------------------------------------ #

    async def execute_stream(
        self,
        task: Task | dict[str, Any],
        llm_params: LLMParams | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream task execution as ``StreamEvent`` objects.

        Mirrors ``execute()`` lifecycle exactly (LLM params, plugins,
        memory, mode routing) but yields events instead of returning
        a single result.

        Event protocol::

            LLM_CALL_START → TOKEN... → LLM_CALL_END
              → TOOL_CALL_START → TOOL_CALL_END → (loop)
            → COMPLETE (final text)

        Autonomous mode additionally emits ``THINKING`` events for
        internal verification steps (Critic, Refiner).

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys
            llm_params: Optional type-safe per-task LLM parameter overrides.

        Yields:
            StreamEvent objects representing the execution progress.

        Example::

            async for event in agent.execute_stream(task):
                if event.type == "token":
                    print(event.token, end="", flush=True)
                elif event.type == "complete":
                    print()  # newline after stream
                elif event.type == "error":
                    print(f"Error: {event.message}")
        """
        try:
            task, mode, agent_ctx = await self._setup_execution(task, llm_params)
        except PluginHalt as halt:
            self._record_termination("plugin_halt", "halted before execution")
            yield StreamEvent.complete_event(str(halt.result) if halt.result else "")
            return
        except AgentConfigError:
            # Misconfiguration is a programmer error — fail fast, same as
            # before.  Runtime failures below become stream events.
            raise
        except Exception as exc:
            self._record_termination("error", f"{type(exc).__name__}: {exc}")
            self._logger.error("Streaming setup failed: %s", exc)
            yield StreamEvent.error_event(f"{type(exc).__name__}: {exc}")
            return

        final_result: str | None = None

        try:
            try:
                async for event in mode.run_stream(self, task):
                    if event.type == StreamEventType.COMPLETE:
                        final_result = event.content
                    yield event
            except PluginHalt as halt:
                final_result = str(halt.result) if halt.result else ""
                self._record_termination("plugin_halt", "plugin halted execution")
                yield StreamEvent.complete_event(final_result)
            except AbstentionSignal as signal:
                # F2: surface abstention as a terminal stream event.
                # We emit the best candidate as complete_event content so
                # UIs that only consume final text still work, but wrap
                # the signal's reason in an error_event so abstention is
                # distinguishable from a clean pass.
                # F5: prefix the error_event with the structured reason
                # code (e.g. "budget_exhausted") so programmatic stream
                # consumers can branch without string-matching feedback.
                final_result = (
                    str(signal.best_candidate) if signal.best_candidate else ""
                )
                code = getattr(signal, "abstain_reason", None)
                self._record_termination(
                    "critic_abstain", f"{code or ''} {signal.reason}".strip()
                )
                yield StreamEvent.complete_event(final_result)
                prefix = f"ABSTAINED[{code}]" if code else "ABSTAINED"
                yield StreamEvent.error_event(f"{prefix}: {signal.reason}")
            except Exception as exc:
                # Same contract as ``execute()``: every failure becomes a
                # typed result.  ``BaseException`` (cancellation, ^C) still
                # propagates so callers can stop the run.
                self._record_termination("error", f"{type(exc).__name__}: {exc}")
                self._logger.error("Streaming execution failed: %s", exc)
                self.state = AgentState.ERROR
                yield StreamEvent.error_event(f"{type(exc).__name__}: {exc}")

            if self._plugin_manager and final_result is not None:
                await self._plugin_manager.run_after_agent(agent_ctx, final_result)
        finally:
            self._current_llm_overrides = {}
            try:
                mode_value = (
                    self.config.execution_mode.value
                    if hasattr(self.config.execution_mode, "value")
                    else str(self.config.execution_mode)
                )
                model_name = None
                if self.llm is not None:
                    model_name = getattr(self.llm, "model", None) or getattr(
                        self.llm, "model_name", None
                    )
                status = (
                    ResultStatus.ERROR
                    if self.state == AgentState.ERROR
                    else ResultStatus.SUCCESS
                )
                # Streaming has no AgentResult to carry ``metadata["coverage"]``;
                # reconcile here so the run report still records it.
                with contextlib.suppress(Exception):
                    self._finalize_coverage(final_result)
                self._last_run_report = self._build_run_report(
                    task if isinstance(task, Task) else Task.from_dict(task),
                    status,
                    mode_value,
                    model_name,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # STRUCTURED OUTPUT HELPERS (cross-cutting, used by all modes)        #
    # ------------------------------------------------------------------ #

    def _resolve_response_format(self):
        """Resolve response_format to an OutputSchema (or None).

        Delegates to ``StructuredOutputHandler``.
        """
        return self._structured_output.resolve_response_format(
            self.response_format, self.llm
        )

    def _get_structured_output_kwargs(self, output_config: Any) -> dict[str, Any]:
        """Build LLM call kwargs for structured output.

        Delegates to ``StructuredOutputHandler``.
        """
        return self._structured_output.get_call_kwargs(
            output_config, self.response_format, self.llm
        )

    def _wrap_structured_output_result(self, response, output_config) -> Any:
        """Wrap LLM response with structured-output metadata.

        Delegates to ``StructuredOutputHandler``.
        """
        return self._structured_output.wrap_result(response, output_config)

    # ------------------------------------------------------------------ #
    # USAGE TRACKING                                                      #
    # ------------------------------------------------------------------ #

    @property
    def last_usage(self) -> UsageSummary:
        """Return the accumulated usage summary from the most recent execution.

        Returns a :class:`UsageSummary` Pydantic model with typed fields:
        ``total``, ``call_count``, ``by_purpose``, ``by_origin``.

        Access fields via attribute (``agent.last_usage.total.prompt_tokens``)
        or convert to a plain dict with ``agent.last_usage.model_dump()``.
        """
        return self._usage_tracker.summary

    @property
    def usage_tracker(self) -> UsageTracker:
        """Direct access to the underlying UsageTracker (for advanced use)."""
        return self._usage_tracker

    # ------------------------------------------------------------------ #
    # UTILITY METHODS (stay on Agent)                                     #
    # ------------------------------------------------------------------ #

    async def _process_result(self, result: Any) -> Any:
        """Process and store execution results."""
        try:
            if self.memory:
                summary = str(result)[:500] if result else ""
                await self.memory.aadd_message("assistant", summary)

            # Process through prompt if available and method exists
            if self.prompt:
                process_result = getattr(self.prompt, "process_result", None)
                if process_result and callable(process_result):
                    if inspect.iscoroutinefunction(process_result):
                        result = await process_result(result)
                    else:
                        result = process_result(result)

            return result

        except Exception as e:
            self._logger.error(f"Result processing failed: {str(e)}")
            raise

    def _validate_task(self, task: dict[str, Any]) -> bool:
        """Validate task format and requirements."""
        required_fields = ["id", "objective"]
        return all(field in task for field in required_fields)

    async def _execute_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Execute a specific tool with parameters."""
        from nucleusiq.tools.errors import ToolNotFoundError

        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ToolNotFoundError(
                f"Tool not found: {tool_name}",
                tool_name=tool_name,
            )

        self.state = AgentState.WAITING_FOR_TOOLS
        try:
            return await tool.execute(**params)
        finally:
            self.state = AgentState.EXECUTING

    async def _handle_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Handle execution errors with appropriate logging and recovery."""
        self._logger.error(f"Error during execution: {str(error)}")

        if self.memory:
            await self.memory.aadd_message(
                "system",
                f"Error: {error}",
            )

        self.metrics.error_count += 1
        self.state = AgentState.ERROR

    async def save_state(self) -> dict[str, Any]:
        """Save agent's current state."""
        state = {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "metrics": self.metrics.model_dump(),
            "current_task": self._current_task,
            "timestamp": datetime.now().isoformat(),
        }

        if self.memory:
            state["memory"] = await self.memory.aexport_state()

        return state

    async def load_state(self, state: dict[str, Any]) -> None:
        """Load agent's saved state."""
        self.state = state["state"]
        self.metrics = AgentMetrics(**state["metrics"])
        self._current_task = state["current_task"]

        if self.memory and "memory" in state:
            await self.memory.aimport_state(state["memory"])

        self._logger.info(f"Loaded agent state from {state['timestamp']}")

    async def delegate_task(
        self, task: dict[str, Any], target_agent: "BaseAgent"
    ) -> Any:
        """Delegate a task to another agent."""
        self._logger.info(
            f"Delegating task to agent to perfoming the task: {target_agent.name}"
        )
        self.state = AgentState.WAITING_FOR_HUMAN

        try:
            return await target_agent.execute(task)
        finally:
            self.state = AgentState.EXECUTING
