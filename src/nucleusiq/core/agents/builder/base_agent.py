# src/nucleusiq/agents/builder/base_agent.py
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from nucleusiq.agents.config.agent_config import AgentConfig, AgentMetrics, AgentState
from nucleusiq.agents.errors import AgentExecutionError, AgentTimeoutError
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

    Agent Identity (labels for logging and documentation):
    - name: Human-readable name (e.g., "TCS_Analyst")
    - role: Short label describing the agent's role (e.g., "Calculator").
            Used for logging, sub-agent identification, and documentation
            only — NOT sent to the LLM.
    - objective: Short label describing the agent's purpose
                 (e.g., "Perform calculations"). Used for logging and
                 documentation only — NOT sent to the LLM.

    Prompt (defines what the LLM sees):
    - prompt: A BasePrompt instance (required). Defines the system
              message, user template, and prompting technique.
              Use ``PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)``
              or any BasePrompt subclass.

    Task (WHAT the user wants - passed to execute()):
    - task.objective: Specific user request (e.g., "What is 5 + 3?")

    Tasks are created per execution and represent specific user requests.
    """

    # Identity labels (for logging and documentation — NOT sent to the LLM)
    id: UUID4 = Field(default_factory=uuid.uuid4)
    name: str = Field(..., description="Human-readable name for the agent")
    role: str = Field(
        default="Agent",
        description="Short label for logging and identification only — "
        "NOT sent to the LLM. Use prompt.system for LLM instructions.",
    )
    objective: str = Field(
        default="",
        description="Short label for documentation only — "
        "NOT sent to the LLM. Use prompt.system for LLM instructions.",
    )

    # Configuration
    config: AgentConfig = Field(default_factory=AgentConfig)

    # State and Metrics
    state: AgentState = Field(default=AgentState.INITIALIZING)
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)

    # Components
    memory: BaseMemory | None = Field(default=None)
    prompt: BasePrompt = Field(
        ...,
        description="Prompt defining the LLM's system message and behaviour. "
        "Required. Use PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)"
        ".configure(system='...') or any BasePrompt subclass.",
    )

    # Tooling
    # Can be BaseTool instances (function calling) or LLM-specific tools (e.g., OpenAITool for native tools)
    tools: list[Any] = Field(
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
    _current_task: dict[str, Any] | None = PrivateAttr(default=None)
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
        if self.config.effective_verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            obs = self.config.observability
            if obs is not None:
                self._logger.setLevel(
                    getattr(logging, obs.effective_log_level, logging.INFO)
                )
            else:
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
    async def execute(self, task: Task | dict[str, Any]) -> Any:
        """
        Execute a given task using the agent's capabilities.

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys
        """
        pass

    @abstractmethod
    async def plan(self, task: Task | dict[str, Any]) -> Plan:
        """
        Create an execution plan for the given task.

        Args:
            task: Task instance or dictionary with 'id' and 'objective' keys

        Returns:
            Plan instance with steps
        """
        pass

    async def _execute_with_retry(self, task: dict[str, Any]) -> Any:
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
                    raise AgentExecutionError(
                        f"Max retry limit ({self.config.max_retries}) reached",
                        original_error=e,
                    ) from e

                self._retry_count += 1
                self.state = AgentState.RETRYING
                self._logger.info(f"Retrying task (attempt {self._retry_count})")
                await asyncio.sleep(1)  # Basic backoff

        raise AgentExecutionError(
            f"Task execution failed after {self._execution_count} attempts",
        )

    async def _execute_step(self, task: Task | dict[str, Any]) -> Any:
        """Execute a single step of the task."""
        if self._check_execution_timeout():
            raise AgentTimeoutError(
                f"Execution exceeded {self.config.max_execution_time}s timeout",
            )

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
