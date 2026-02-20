# src/nucleusiq/agents/agent.py
"""
Agent — Thin orchestrator for NucleusIQ agents.

Routes execution to mode strategies (Direct, Standard, Autonomous)
via a pluggable registry.  All heavy logic lives in:

- ``modes/``       — execution strategies
- ``planning/``    — plan creation & execution
- ``messaging/``   — LLM message construction
"""

from typing import Any, ClassVar, Dict, List, Optional, Type, Union
from datetime import datetime
import asyncio
import inspect
from pydantic import Field, PrivateAttr

from nucleusiq.agents.builder.base_agent import BaseAgent
from nucleusiq.agents.config.agent_config import AgentState, AgentMetrics
from nucleusiq.agents.task import Task
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.components.executor import Executor
from nucleusiq.agents.structured_output.handler import StructuredOutputHandler
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.llms.llm_params import LLMParams
from nucleusiq.plugins.base import BasePlugin, AgentContext
from nucleusiq.plugins.manager import PluginManager
from nucleusiq.plugins.errors import PluginHalt

# Mode imports
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.direct_mode import DirectMode
from nucleusiq.agents.modes.standard_mode import StandardMode
from nucleusiq.agents.modes.autonomous_mode import AutonomousMode


class Agent(BaseAgent):
    """
    Concrete implementation of an agent in the NucleusIQ framework.

    This is a thin orchestrator that delegates execution to mode strategies
    (DirectMode, StandardMode, AutonomousMode) via a pluggable registry.

    Execution Modes (Gearbox Strategy):
    - "direct": Fast, simple, no tools (Gear 1)
    - "standard": Tool-enabled, linear execution (Gear 2) - default
    - "autonomous": Full reasoning loop with planning and self-correction (Gear 3)

    Prompt Precedence:
    - If ``prompt`` is provided, it takes precedence over ``role``/``objective``
      for LLM message construction during execution.
    - If ``prompt`` is None, ``role`` and ``objective`` are used to construct
      the system message: "You are a {role}. Your objective is to {objective}."
    - ``role`` and ``objective`` are always used for planning context, even
      when prompt exists.

    Example:
        # With prompt (prompt takes precedence)
        agent = Agent(
            name="CalculatorBot",
            role="Calculator",              # Used for planning context only
            objective="Perform calculations", # Used for planning context only
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
    def register_mode(
        cls, name: str, mode_class: Type[BaseExecutionMode]
    ) -> None:
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

    _executor: Optional[Executor] = PrivateAttr(default=None)
    _plugin_manager: PluginManager = PrivateAttr(default=None)
    _structured_output: StructuredOutputHandler = PrivateAttr(
        default_factory=StructuredOutputHandler
    )

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
                self._logger.debug(
                    f"Prompt system initialized \n {prompt_text}"
                )

            # Initialize tools
            for tool in self.tools:
                await tool.initialize()
            if self.tools:
                self._logger.debug(
                    "Initialised %d tools", len(self.tools)
                )

            # Initialization succeeded
            self.state = AgentState.INITIALIZING
            self._logger.info(
                "Agent initialization completed successfully"
            )

        except Exception as e:
            self.state = AgentState.ERROR
            self._logger.error(
                f"Agent initialization failed: {str(e)}"
            )
            raise

    # ------------------------------------------------------------------ #
    # PLANNING (very simple by default)                                   #
    # ------------------------------------------------------------------ #

    async def plan(self, task: Union[Task, Dict[str, Any]]) -> Plan:
        """
        Create an execution plan for the given task.

        By default, returns a simple one-step plan that executes the task
        directly.  Override this method or use LLM-based planning
        (``_create_llm_plan``) for more sophisticated multi-step planning.

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
        per_execute: Optional[LLMParams] = None,
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

    async def execute(
        self,
        task: Union[Task, Dict[str, Any]],
        llm_params: Optional[LLMParams] = None,
    ) -> Any:
        """
        Execute a task using the agent's capabilities.

        Execution Flow (Gearbox Strategy):
        - Direct mode: Fast, simple, no tools
        - Standard mode: Tool-enabled, linear execution (default)
        - Autonomous mode: Full reasoning loop with planning and self-correction

        The execution uses:
        - Task: User's request (what to do) - from task.objective
        - Prompt: Agent's instructions (how to behave) - from self.prompt
        - Plan: Task decomposition (how to break down) - optional, from plan()

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys
            llm_params: Optional type-safe per-task LLM parameter overrides.
                Accepts :class:`LLMParams` or any provider subclass
                (e.g. ``OpenAILLMParams``).  These override both the LLM-level
                defaults and the ``AgentConfig.llm_params`` for this single
                execution only.

        Returns:
            Execution result (final answer or tool result)
        """
        # Convert dict to Task if needed (backward compatibility)
        if isinstance(task, dict):
            task = Task.from_dict(task)

        # Resolve merged LLM params for this execution
        self._current_llm_overrides = self._resolve_llm_params(
            per_execute=llm_params
        )

        self._logger.debug("Starting execution for task %s", task.id)
        self._current_task = task.to_dict()  # Store as dict for compat

        # Ensure plugin manager exists (even if initialize() was not called)
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager(self.plugins)
        self._plugin_manager.reset_counters()

        # --- BEFORE_AGENT hook ---
        agent_ctx = AgentContext(
            agent_name=self.name,
            task=task,
            state=self.state,
            config=self.config,
            memory=self.memory,
        )
        try:
            agent_ctx = await self._plugin_manager.run_before_agent(agent_ctx)
        except PluginHalt as halt:
            return halt.result

        # Store user input in memory before execution
        if self.memory:
            user_input = (
                task.objective
                if hasattr(task, "objective")
                else task.to_dict().get("objective", "")
            )
            if user_input:
                await self.memory.aadd_message("user", user_input)

        # Route to appropriate execution mode (Gearbox Strategy)
        from nucleusiq.agents.config.agent_config import ExecutionMode

        execution_mode = self.config.execution_mode

        # Get mode value (handle both enum and string for backward compat)
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

        # Look up mode from registry
        mode_class = self._mode_registry.get(mode_value)
        if not mode_class:
            raise ValueError(f"Unknown execution mode: {execution_mode}")

        try:
            try:
                result = await mode_class().run(self, task)
            except PluginHalt as halt:
                result = halt.result

            # --- AFTER_AGENT hook ---
            result = await self._plugin_manager.run_after_agent(
                agent_ctx, result
            )
            return result
        finally:
            # Clean up per-execute overrides
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
                process_result = getattr(
                    self.prompt, "process_result", None
                )
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

    async def _execute_tool(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Any:
        """Execute a specific tool with parameters."""
        tool = next(
            (t for t in self.tools if t.name == tool_name), None
        )
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        self.state = AgentState.WAITING_FOR_TOOLS
        try:
            return await tool.execute(**params)
        finally:
            self.state = AgentState.EXECUTING

    async def _handle_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> None:
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

        self._logger.info(
            f"Loaded agent state from {state['timestamp']}"
        )

    async def delegate_task(
        self, task: Dict[str, Any], target_agent: "BaseAgent"
    ) -> Any:
        """Delegate a task to another agent."""
        self._logger.info(
            "Delegating task to agent to perfoming the task: "
            f"{target_agent.name}"
        )
        self.state = AgentState.WAITING_FOR_HUMAN

        try:
            return await target_agent.execute(task)
        finally:
            self.state = AgentState.EXECUTING

