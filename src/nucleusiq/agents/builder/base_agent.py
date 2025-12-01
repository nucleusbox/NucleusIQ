# src/nucleusiq/agents/builder/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, UUID4, validator, PrivateAttr
import uuid
from datetime import datetime
import logging
from enum import Enum
import asyncio

from nucleusiq.agents.config.agent_config import AgentConfig, AgentMetrics, AgentState
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.llms.base_llm import BaseLLM

class BaseAgent(ABC, BaseModel):
    """
    Abstract base class defining the core functionality for all agents in NucleusIQ.
    
    This class establishes the fundamental structure and capabilities that every
    agent must implement, ensuring consistency across the framework.
    """
    
    # Identity and Purpose
    id: UUID4 = Field(default_factory=uuid.uuid4)
    name: str = Field(..., description="Human-readable name for the agent")
    role: str = Field(..., description="Primary function or responsibility")
    objective: str = Field(..., description="Core goal or purpose")
    narrative: str = Field(..., description="Contextual background/personality")
    
    # Configuration
    config: AgentConfig = Field(default_factory=AgentConfig)
    
    # State and Metrics
    state: AgentState = Field(default=AgentState.INITIALIZING)
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    
    # Components
    memory: Optional[Any] = Field(default=None)
    prompt: Optional[BasePrompt] = Field(default=None)

    # Tooling
    tools: List[Any] = Field(
            default_factory=list,
            description="List of tools (BaseTool instances) available to this agent",
        )
    
    llm: Optional[BaseLLM] = Field(default=None)
    
    # Private attributes
    _logger: logging.Logger = PrivateAttr()
    _start_time: Optional[float] = PrivateAttr(default=None)
    _current_task: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _execution_count: int = PrivateAttr(default=0)
    _retry_count: int = PrivateAttr(default=0)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

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
        if self.config.verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)


    def _validate_configuration(self):
        """Validate the agent's configuration."""
        if self.config.allow_code_execution and self.config.code_execution_mode == "unsafe":
            self._logger.warning(
                "Agent configured with unsafe code execution mode. "
                "This is not recommended for production use."
            )

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent components and resources."""
        pass

    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Any:
        """Execute a given task using the agent's capabilities."""
        pass

    @abstractmethod
    async def plan(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create an execution plan for the given task."""
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
                
        raise RuntimeError(f"Task execution failed after {self._execution_count} attempts")

    async def _execute_step(self, task: Dict[str, Any]) -> Any:
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
            
        except Exception as e:
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
                self.metrics.total_execution_time / 
                self.metrics.successful_executions
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