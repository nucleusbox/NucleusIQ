# src/nucleusiq/agents/builder/base_agent.py
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from nucleusiq.agents.config.agent_config import AgentConfig, AgentMetrics, AgentState
from nucleusiq.agents.plan import Plan
from nucleusiq.agents.task import Task
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.memory.base import BaseMemory
from nucleusiq.prompts.base import BasePrompt
from pydantic import UUID4, BaseModel, ConfigDict, Field, PrivateAttr


class BaseAgent(ABC, BaseModel):
    """
    Abstract base class defining the core functionality for all agents in NucleusIQ.

    This class establishes the fundamental structure and capabilities that every
    agent must implement, ensuring consistency across the framework.

    Agent Identity (WHO the agent is - set at creation time):
    - role: Agent's role (e.g., "Calculator", "Assistant")
            Used as system prompt when `prompt` is not provided.
    - objective: Agent's general purpose (e.g., "Perform calculations")
                 Used as system prompt when `prompt` is not provided.
    - narrative: Agent's description/personality (optional, for documentation)

    Prompt Precedence:
    - If `prompt` is provided, it takes precedence over `role`/`objective`
      for LLM message construction during execution.
    - If `prompt` is None, `role` and `objective` are used to construct
      the system message: "You are a {role}. Your objective is to {objective}."
    - `role` and `objective` are always used for:
      - Planning context (even when prompt exists)
      - Logging and identification
      - Fallback planning prompts

    Task (WHAT the user wants - passed to execute()):
    - task.objective: Specific user request (e.g., "What is 5 + 3?")

    Tasks are created per execution and represent specific user requests.
    The agent's identity (role, objective, narrative) is separate from the task.
    """

    # Identity and Purpose (WHO the agent is)
    id: UUID4 = Field(default_factory=uuid.uuid4)
    name: str = Field(..., description="Human-readable name for the agent")
    role: str = Field(
        ...,
        description="Agent's role (e.g., 'Calculator', 'Assistant') - used as system prompt when prompt is None",
    )
    objective: str = Field(
        ...,
        description="Agent's general purpose (e.g., 'Perform calculations') - used as system prompt when prompt is None",
    )
    narrative: str | None = Field(
        default=None,
        description="Agent's description/personality (optional, for documentation)",
    )

    # Configuration
    config: AgentConfig = Field(default_factory=AgentConfig)

    # State and Metrics
    state: AgentState = Field(default=AgentState.INITIALIZING)
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)

    # Components
    memory: BaseMemory | None = Field(default=None)
    prompt: BasePrompt | None = Field(default=None)

    # Tooling
    # Can be BaseTool instances (function calling) or LLM-specific tools (e.g., OpenAITool for native tools)
    tools: List[Any] = Field(
        default_factory=list,
        description="List of tools: BaseTool instances (function calling) or LLM-specific tools (e.g., OpenAITool)",
    )

    llm: BaseLLM | None = Field(default=None)

    # Structured Output
    # Can be: ProviderStrategy, ToolStrategy, schema type, or None
    response_format: Any | None = Field(
        default=None,
        description="""Structured output configuration. Supports:
        - ProviderStrategy[T]: Uses provider-native structured output
        - ToolStrategy[T]: Uses tool calling for structured output
        - type[T]: Auto-selects best strategy based on model capabilities
        - None: No structured output (default)
        """,
    )

    # Private attributes
    _logger: logging.Logger = PrivateAttr()
    _start_time: float | None = PrivateAttr(default=None)
    _current_task: Dict[str, Any] | None = PrivateAttr(default=None)
    _execution_count: int = PrivateAttr(default=0)
    _retry_count: int = PrivateAttr(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_logger()
        self._validate_configuration()

    def _initialize_logger(self):
        """Initialize the agent's logger."""
        self._logger = logging.getLogger(f"agent.{self.name}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f"[%(levelname)s] {self.name}: %(message)s")
        handler.setFormatter(formatter)
        if not self._logger.handlers:
            self._logger.addHandler(handler)
        # Avoid duplicate logs when the app configures root logging (e.g., logging.basicConfig in examples).
        # We attach our own handler, so we should not also propagate to root.
        self._logger.propagate = False
        if self.config.verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            # Default to INFO for a small set of high-signal traces.
            self._logger.setLevel(logging.INFO)

    def _validate_configuration(self):
        """Validate the agent's configuration."""
        if (
            self.config.allow_code_execution
            and self.config.code_execution_mode == "unsafe"
        ):
            self._logger.warning(
                "Agent configured with unsafe code execution mode. "
                "This is not recommended for production use."
            )

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent components and resources."""
        pass

    @abstractmethod
    async def execute(self, task: Task | Dict[str, Any]) -> Any:
        """
        Execute a given task using the agent's capabilities.

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys
        """
        pass

    @abstractmethod
    async def plan(self, task: Task | Dict[str, Any]) -> Plan:
        """
        Create an execution plan for the given task.

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys

        Returns:
            Plan instance with steps
        """
        pass

    async def _execute_with_retry(self, task: Dict[str, Any]) -> Any:
        """Execute a task with retry mechanism."""
        self._execution_count = 0
        self._retry_count = 0
        self._start_time = datetime.now().timestamp()

        while self._execution_count < self.config.max_retries:
            try:
                self._execution_count += 1
                result = await self._execute_step(task)

                if self._validate_result(result):
                    return result

            except Exception as e:
                self._logger.error(f"Execution failed: {str(e)}")
                if self._retry_count >= self.config.max_retries:
                    raise RuntimeError(
                        f"Max retry limit ({self.config.max_retries}) reached"
                    )

                self._retry_count += 1
                self.state = AgentState.RETRYING
                self._logger.info(f"Retrying task (attempt {self._retry_count})")
                await asyncio.sleep(1)  # Basic backoff

        raise RuntimeError(
            f"Task execution failed after {self._execution_count} attempts"
        )

    async def _execute_step(self, task: Task | Dict[str, Any]) -> Any:
        """Execute a single step of the task."""
        if self._check_execution_timeout():
            raise TimeoutError("Maximum execution time exceeded")

        self.state = AgentState.EXECUTING
        start_time = datetime.now().timestamp()

        try:
            # Execute the actual task logic
            result = await self.execute(task)

            # Update metrics
            execution_time = datetime.now().timestamp() - start_time
            self._update_metrics(success=True, execution_time=execution_time)

            return result

        except Exception:
            self._update_metrics(success=False)
            raise

    def _check_execution_timeout(self) -> bool:
        """Check if execution time has exceeded the limit."""
        if not self._start_time or not self.config.max_execution_time:
            return False

        elapsed_time = datetime.now().timestamp() - self._start_time
        return elapsed_time > self.config.max_execution_time

    def _update_metrics(self, success: bool, execution_time: float = 0.0):
        """Update agent performance metrics."""
        if success:
            self.metrics.successful_executions += 1
            self.metrics.total_execution_time += execution_time
            self.metrics.average_response_time = (
                self.metrics.total_execution_time / self.metrics.successful_executions
            )
        else:
            self.metrics.failed_executions += 1
            self.metrics.error_count += 1

    def _validate_result(self, result: Any) -> bool:
        """Validate task execution results."""
        return result is not None

    # async def add_tool(self, tool: BaseTool) -> None:
    #     """Add a new tool to the agent's toolkit."""
    #     self.tools.append(tool)
    #     self._logger.info(f"Added tool: {tool.name}")

    async def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent's toolkit."""
        self.tools = [t for t in self.tools if t.name != tool_name]
        self._logger.info(f"Removed tool: {tool_name}")
