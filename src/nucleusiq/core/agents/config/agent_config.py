# src/nucleusiq/agents/config/agent_config.py
from enum import Enum
from typing import Literal

from nucleusiq.llms.llm_params import LLMParams
from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """Execution modes (Gearbox Strategy) for agent execution."""

    DIRECT = "direct"  # Gear 1: Fast, optional tools (max 5)
    STANDARD = "standard"  # Gear 2: Tool-enabled loop (max 30) — default
    AUTONOMOUS = "autonomous"  # Gear 3: Orchestration + Critic/Refiner (max 100)


class AgentConfig(BaseModel):
    """Configuration settings for agent behavior."""

    max_execution_time: int = Field(
        default=3600, description="Maximum execution time in seconds"
    )
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    allow_code_execution: bool = Field(
        default=False, description="Enable code execution capabilities"
    )
    code_execution_mode: Literal["safe", "unsafe"] = Field(
        default="safe", description="Code execution safety mode"
    )
    respect_context_window: bool = Field(
        default=True, description="Maintain context within model's window"
    )
    verbose: bool = Field(default=False, description="Enable detailed logging")
    # Gearbox Strategy: Execution Modes
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.STANDARD,
        description="Execution mode (gear): DIRECT (fast, optional tools), STANDARD (tool-enabled loop), AUTONOMOUS (orchestration + verification)",
    )
    require_quality_check: bool = Field(
        default=False,
        description="Require quality check before returning (autonomous mode only). Uses Critic component to review output.",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum iterations for iterative agents (autonomous mode, ReAct, etc.)",
    )

    # LLM call budgets (model-agnostic knobs)
    llm_max_tokens: int = Field(
        default=2048,
        description="Token budget for normal LLM calls.",
    )
    step_inference_max_tokens: int = Field(
        default=2048,
        description="Token budget for per-step tool-argument inference.",
    )

    # Timeout settings (in seconds)
    step_timeout: int = Field(
        default=60,
        description="Timeout in seconds for each step execution. If exceeded, step fails.",
    )
    llm_call_timeout: int = Field(
        default=90, description="Timeout in seconds for individual LLM API calls."
    )
    step_max_retries: int = Field(
        default=2,
        description="Maximum retries for a failed step before giving up (0 = no retries).",
    )

    # Tool limits
    max_tool_calls: int | None = Field(
        default=None,
        description=(
            "Maximum tool calls per execution. If None, uses mode defaults: "
            "DIRECT=5, STANDARD=30, AUTONOMOUS=100."
        ),
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
            "Off by default — use Critic component for independent verification."
        ),
    )

    # Type-safe LLM parameter overrides for this agent.
    # Accepts LLMParams (base) or any provider subclass (OpenAILLMParams, etc.)
    # These are merged into every llm.call() this agent makes, overriding
    # the LLM-level defaults set in BaseOpenAI.__init__().
    llm_params: LLMParams | None = Field(
        default=None,
        description=(
            "Type-safe LLM call parameter overrides for this agent. "
            "Accepts LLMParams or any provider-specific subclass "
            "(e.g. OpenAILLMParams). Only non-None fields are merged."
        ),
    )

    _MODE_TOOL_DEFAULTS: dict = {"direct": 5, "standard": 30, "autonomous": 100}

    def get_effective_max_tool_calls(self) -> int:
        """Return the effective tool call limit for the current execution mode.

        If ``max_tool_calls`` is explicitly set, that value is used.
        Otherwise the mode default is returned (DIRECT=5, STANDARD=30,
        AUTONOMOUS=100).
        """
        if self.max_tool_calls is not None:
            return self.max_tool_calls
        mode_val = (
            self.execution_mode.value
            if hasattr(self.execution_mode, "value")
            else str(self.execution_mode)
        )
        return self._MODE_TOOL_DEFAULTS.get(mode_val, 30)


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
