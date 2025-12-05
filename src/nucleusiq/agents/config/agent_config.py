# src/nucleusiq/agents/config/agent_config.py
from typing import Literal
from pydantic import BaseModel, Field
from enum import Enum

class ExecutionMode(str, Enum):
    """Execution modes (Gearbox Strategy) for agent execution."""
    DIRECT = "direct"           # Gear 1: Fast, simple, no tools
    STANDARD = "standard"       # Gear 2: Tool-enabled, linear execution (default)
    AUTONOMOUS = "autonomous"   # Gear 3: Full reasoning loop with planning and self-correction

class AgentConfig(BaseModel):
    """Configuration settings for agent behavior."""
    max_execution_time: int = Field(
        default=3600,
        description="Maximum execution time in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    allow_code_execution: bool = Field(
        default=False,
        description="Enable code execution capabilities"
    )
    code_execution_mode: Literal["safe", "unsafe"] = Field(
        default="safe",
        description="Code execution safety mode"
    )
    respect_context_window: bool = Field(
        default=True,
        description="Maintain context within model's window"
    )
    verbose: bool = Field(
        default=False,
        description="Enable detailed logging"
    )
    use_planning: bool = Field(
        default=False,
        description="[DEPRECATED] Enable planning mode. Use execution_mode='autonomous' instead. If True, agent will call plan() before execute() to break down complex tasks into steps."
    )
    # Gearbox Strategy: Execution Modes
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.STANDARD,
        description="Execution mode (gear): DIRECT (fast, no tools), STANDARD (tool-enabled, linear), AUTONOMOUS (full reasoning loop with planning and self-correction)"
    )
    enable_memory: bool = Field(
        default=True,
        description="Enable memory for context (standard, autonomous modes). Memory stores conversation history and partial results."
    )
    require_quality_check: bool = Field(
        default=False,
        description="Require quality check before returning (autonomous mode only). Uses Critic component to review output."
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum iterations for iterative agents (autonomous mode, ReAct, etc.)"
    )

class AgentMetrics(BaseModel):
    """Tracks agent performance metrics."""
    tasks_completed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_response_time: float = 0.0
    retry_count: int = 0
    error_count: int = 0

class AgentState(str, Enum):
    """Defines the possible states of an agent."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    WAITING_FOR_HUMAN = "waiting_for_human"
    RETRYING = "retrying"
    BACKOFF = "backoff"
    ERROR = "error"
    COMPLETED = "completed"