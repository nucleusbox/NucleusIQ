# src/nucleusiq/agents/config/agent_config.py
from typing import Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum

from nucleusiq.llms.llm_params import LLMParams

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
    require_quality_check: bool = Field(
        default=False,
        description="Require quality check before returning (autonomous mode only). Uses Critic component to review output."
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum iterations for iterative agents (autonomous mode, ReAct, etc.)"
    )

    # LLM call budgets (model-agnostic knobs)
    llm_max_tokens: int = Field(
        default=2048,
        description="Token budget for normal LLM calls (STANDARD mode and general execution)."
    )
    planning_max_tokens: int = Field(
        default=4096,
        description="Token budget for plan generation calls (AUTONOMOUS planning)."
    )
    step_inference_max_tokens: int = Field(
        default=2048,
        description="Token budget for per-step tool-argument inference during plan execution."
    )

    # Timeout settings (in seconds)
    planning_timeout: int = Field(
        default=120,
        description="Timeout in seconds for plan generation. If exceeded, planning fails or retries."
    )
    step_timeout: int = Field(
        default=60,
        description="Timeout in seconds for each step execution. If exceeded, step fails."
    )
    llm_call_timeout: int = Field(
        default=90,
        description="Timeout in seconds for individual LLM API calls."
    )
    step_max_retries: int = Field(
        default=2,
        description="Maximum retries for a failed step before giving up (0 = no retries)."
    )

    # Autonomous mode
    critique_rounds: int = Field(
        default=3,
        description=(
            "[DEPRECATED] Use max_retries instead. Kept for backward compatibility. "
            "In the new architecture, max_retries controls validation retry cycles."
        ),
    )
    max_sub_agents: int = Field(
        default=5,
        description=(
            "Maximum parallel sub-agents for complex task decomposition "
            "in autonomous mode."
        ),
    )
    llm_review: bool = Field(
        default=False,
        description=(
            "Enable LLM-based review as validation Layer 3 (autonomous mode). "
            "Off by default — use plugin validators for reliable external checks."
        ),
    )

    # Type-safe LLM parameter overrides for this agent.
    # Accepts LLMParams (base) or any provider subclass (OpenAILLMParams, etc.)
    # These are merged into every llm.call() this agent makes, overriding
    # the LLM-level defaults set in BaseOpenAI.__init__().
    llm_params: Optional[LLMParams] = Field(
        default=None,
        description=(
            "Type-safe LLM call parameter overrides for this agent. "
            "Accepts LLMParams or any provider-specific subclass "
            "(e.g. OpenAILLMParams). Only non-None fields are merged."
        ),
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
