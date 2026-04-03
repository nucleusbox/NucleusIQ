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
from typing import Any, ClassVar, Dict, List, Type

from nucleusiq.agents.agent_result import AgentResult, AutonomousDetail, ResultStatus
from nucleusiq.agents.builder.base_agent import BaseAgent
from nucleusiq.agents.components.executor import Executor
from nucleusiq.agents.errors import AgentConfigError
from nucleusiq.agents.observability import DefaultExecutionTracer
from nucleusiq.agents.usage.usage_tracker import UsageSummary, UsageTracker
from nucleusiq.agents.config.agent_config import AgentMetrics, AgentState
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode

# Mode imports
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.direct_mode import DirectMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.structured_output.handler import StructuredOutputHandler
from nucleusiq.agents.task import Task
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
    - "direct": Fast, optional tools, max 5 tool calls (Gear 1)
    - "standard": Tool-enabled loop, max 30 tool calls (Gear 2) - default
    - "autonomous": Orchestration + Critic/Refiner, max 100 tool calls (Gear 3)

    Prompt Precedence:
    - If ``prompt`` is provided, it takes precedence over ``role``/``objective``
      for LLM message construction during execution.
    - If ``prompt`` is None, ``role`` and ``objective`` are used to construct
      the system message: "You are a {role}. Your objective is to {objective}."

    Example:
        # With prompt (prompt takes precedence)
        agent = Agent(
            name="CalculatorBot",
            role="Calculator",
            objective="Perform calculations",
            prompt=PromptFactory.create_prompt().configure(
                system="You are a helpful calculator assistant.",
                user="Answer questions accurately."
            ),
            llm=llm,
            config=AgentConfig(execution_mode="standard")
        )

        # Without prompt (role/objective used)
        agent = Agent(
            name="CalculatorBot",
            role="Calculator",              # Used to build system message
            objective="Perform calculations", # Used to build system message
            prompt=None,
            llm=llm,
            config=AgentConfig(execution_mode="direct")
        )
    """

    # ------------------------------------------------------------------ #
    # Mode registry (Open/Closed Principle)                               #
    # ------------------------------------------------------------------ #

    _mode_registry: ClassVar[Dict[str, Type[BaseExecutionMode]]] = {
        "direct": DirectMode,
        "standard": StandardMode,
        "autonomous": AutonomousMode,
    }

    @classmethod
    def register_mode(cls, name: str, mode_class: Type[BaseExecutionMode]) -> None:
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

    plugins: List[BasePlugin] = Field(
        default_factory=list,
        description="List of plugins to hook into the agent execution pipeline",
    )

    # ------------------------------------------------------------------ #
    # Private attributes (initialised in initialize())                    #
    # ------------------------------------------------------------------ #

    _executor: Executor | None = PrivateAttr(default=None)
    _plugin_manager: PluginManager = PrivateAttr(default=None)
    _structured_output: StructuredOutputHandler = PrivateAttr(
        default_factory=StructuredOutputHandler
    )
    _usage_tracker: UsageTracker = PrivateAttr(default_factory=UsageTracker)
    _tracer: DefaultExecutionTracer | None = PrivateAttr(default=None)

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

    async def plan(self, task: Task | Dict[str, Any]) -> Plan:
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
    ) -> Dict[str, Any]:
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

    async def _setup_execution(
        self,
        task: Task | Dict[str, Any],
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
        self._tracer = DefaultExecutionTracer() if self.config.enable_tracing else None

        agent_ctx = AgentContext(
            agent_name=self.name,
            task=task,
            state=self.state,
            config=self.config,
            memory=self.memory,
        )
        agent_ctx = await self._plugin_manager.run_before_agent(agent_ctx)

        max_tools = self.config.get_effective_max_tool_calls()
        if len(self.tools) > max_tools:
            mode_value = (
                self.config.execution_mode.value
                if hasattr(self.config.execution_mode, "value")
                else str(self.config.execution_mode)
            )
            raise AgentConfigError(
                f"Agent '{self.name}' has {len(self.tools)} tools but "
                f"{mode_value.upper()} mode allows max {max_tools}. "
                f"Reduce tools or switch to a higher execution mode.",
                mode=mode_value,
            )

        mode = self._resolve_mode()
        return task, mode, agent_ctx

    # ------------------------------------------------------------------ #
    # EXECUTION — non-streaming                                            #
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        task: Task | Dict[str, Any],
        llm_params: LLMParams | None = None,
    ) -> AgentResult:
        """Execute a task using the agent's capabilities.

        Execution Flow (Gearbox Strategy):
        - Direct mode: Fast, optional tools (max 5 tool calls)
        - Standard mode: Tool-enabled loop (max 30 tool calls) — default
        - Autonomous mode: Orchestration + Critic/Refiner (max 100 tool calls)

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
            try:
                output = await mode.run(self, task_obj)
            except PluginHalt as halt:
                status = ResultStatus.HALTED
                output = halt.result

            output = await self._plugin_manager.run_after_agent(agent_ctx, output)
            return self._build_result(task_obj, status, output, None, None, t0)

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
    ) -> AgentResult:
        """Construct a frozen :class:`AgentResult` from execution data."""
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

        tool_calls_t: tuple = ()
        llm_calls_t: tuple = ()
        plugin_events_t: tuple = ()
        warnings_t: tuple = ()
        memory_snap = None
        autonomous_out: AutonomousDetail | None = None

        tracer = getattr(self, "_tracer", None)
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
            usage=usage_dict,
            tool_calls=tool_calls_t,
            llm_calls=llm_calls_t,
            plugin_events=plugin_events_t,
            memory_snapshot=memory_snap,
            autonomous=autonomous_out,
            warnings=warnings_t,
        )

    # ------------------------------------------------------------------ #
    # EXECUTION — streaming                                                #
    # ------------------------------------------------------------------ #

    async def execute_stream(
        self,
        task: Task | Dict[str, Any],
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

    def _get_structured_output_kwargs(self, output_config: Any) -> Dict[str, Any]:
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

    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task format and requirements."""
        required_fields = ["id", "objective"]
        return all(field in task for field in required_fields)

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
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

    async def _handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle execution errors with appropriate logging and recovery."""
        self._logger.error(f"Error during execution: {str(error)}")

        if self.memory:
            await self.memory.aadd_message(
                "system",
                f"Error: {error}",
            )

        self.metrics.error_count += 1
        self.state = AgentState.ERROR

    async def save_state(self) -> Dict[str, Any]:
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

    async def load_state(self, state: Dict[str, Any]) -> None:
        """Load agent's saved state."""
        self.state = state["state"]
        self.metrics = AgentMetrics(**state["metrics"])
        self._current_task = state["current_task"]

        if self.memory and "memory" in state:
            await self.memory.aimport_state(state["memory"])

        self._logger.info(f"Loaded agent state from {state['timestamp']}")

    async def delegate_task(
        self, task: Dict[str, Any], target_agent: "BaseAgent"
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
