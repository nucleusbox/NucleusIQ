# src/nucleusiq/agents/agent.py
"""
Agent — Thin orchestrator for NucleusIQ agents.

Routes execution to mode strategies (Direct, Standard, Autonomous)
via a pluggable registry.  All heavy logic lives in:

- ``modes/``       — execution strategies
- ``components/``  — executor, decomposer, critic, refiner, validation
- ``messaging/``   — LLM message construction
"""

import inspect
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

    # ------------------------------------------------------------------ #
    # LIFECYCLE                                                           #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Initialize agent components and resources."""
        self._logger.info(f"Initializing agent: {self.name}")

        try:
            # Initialize plugin manager
            self._plugin_manager = PluginManager(self.plugins)
            if self.plugins:
                self._logger.debug(
                    "Plugin manager initialized with %d plugins",
                    len(self.plugins),
                )

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

            # Initialize tools
            for tool in self.tools:
                await tool.initialize()
            if self.tools:
                self._logger.debug("Initialised %d tools", len(self.tools))

            # Initialization succeeded
            self.state = AgentState.INITIALIZING
            self._logger.info("Agent initialization completed successfully")

        except Exception as e:
            self.state = AgentState.ERROR
            self._logger.error(f"Agent initialization failed: {str(e)}")
            raise

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
            if max_tokens is None and self.llm:
                try:
                    raw = self.llm.get_context_window()
                    max_tokens = int(raw) if isinstance(raw, (int, float)) else None
                except Exception:
                    max_tokens = None

            counter = self._build_token_counter()

            return ContextEngine(
                config=ctx_config,
                token_counter=counter,
                max_tokens=max_tokens or 128_000,
                tracer=self._tracer,
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
        max_chars: int = 12_000,
    ) -> Any:
        """Build a bounded synthesis package from this run's curated state."""
        from nucleusiq.agents.context.synthesis_package import build_synthesis_package

        return build_synthesis_package(
            task=task,
            output_shape=output_shape,
            workspace=self.workspace,
            evidence=self.evidence_dossier,
            recalled_snippets=recalled_snippets,
            max_chars=max_chars,
        )

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
        max_chars: int = 12_000,
    ) -> list[Any] | None:
        """Build package-based synthesis messages when curated state exists."""
        if not self._has_context_state():
            return None

        from nucleusiq.agents.chat_models import ChatMessage

        package = self.build_synthesis_package(
            task=task,
            output_shape=output_shape,
            max_chars=max_chars,
        )
        self._last_synthesis_package = package
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

        return [
            ChatMessage(
                role="user",
                content=(
                    f"{package.text}\n\n"
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
        self._evidence_dossier = InMemoryEvidenceDossier()
        self._document_corpus = InMemoryDocumentCorpus()
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
        return task, mode, agent_ctx

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
                return self._build_result(task_obj, status, output, None, None, t0)

            status = ResultStatus.SUCCESS
            abstention_reason: str | None = None
            abstention_code: str | None = None
            try:
                output = await mode.run(self, task_obj)
            except PluginHalt as halt:
                status = ResultStatus.HALTED
                output = halt.result
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
            usage=usage_dict,
            tool_calls=tool_calls_t,
            llm_calls=llm_calls_t,
            plugin_events=plugin_events_t,
            memory_snapshot=memory_snap,
            autonomous=autonomous_out,
            context_telemetry=context_tel,
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
            yield StreamEvent.complete_event(str(halt.result) if halt.result else "")
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
                yield StreamEvent.complete_event(final_result)
                code = getattr(signal, "abstain_reason", None)
                prefix = f"ABSTAINED[{code}]" if code else "ABSTAINED"
                yield StreamEvent.error_event(f"{prefix}: {signal.reason}")

            if self._plugin_manager and final_result is not None:
                await self._plugin_manager.run_after_agent(agent_ctx, final_result)
        finally:
            self._current_llm_overrides = {}

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
