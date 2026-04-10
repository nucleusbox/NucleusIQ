"""ObservabilityConfig — unified observability settings.

Consolidates ``verbose`` and ``enable_tracing`` into a single,
forward-compatible configuration object.  Legacy fields on
``AgentConfig`` continue to work via backward-compatible resolution.

Usage::

    from nucleusiq.agents.config import ObservabilityConfig

    # Explicit (preferred)
    config = AgentConfig(
        observability=ObservabilityConfig(
            tracing=True,
            verbose=True,
            log_level="DEBUG",
        )
    )

    # Legacy (still works)
    config = AgentConfig(verbose=True, enable_tracing=True)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ObservabilityConfig(BaseModel):
    """Unified observability configuration.

    When set on ``AgentConfig.observability``, takes precedence over
    the legacy ``verbose`` and ``enable_tracing`` fields.
    """

    model_config = ConfigDict(frozen=True)

    tracing: bool = Field(
        default=False,
        description=(
            "Populate AgentResult with execution trace data "
            "(llm_calls, tool_calls, warnings, context_telemetry). "
            "Off by default for zero overhead."
        ),
    )

    verbose: bool = Field(
        default=False,
        description="Enable detailed DEBUG-level logging.",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description=(
            "Logger level for the agent. When verbose=True, this is "
            "forced to DEBUG regardless of this setting."
        ),
    )

    log_llm_calls: bool = Field(
        default=False,
        description="Log full LLM request/response payloads (may contain PII).",
    )

    log_tool_results: bool = Field(
        default=False,
        description="Log full tool results (may be large).",
    )

    @property
    def effective_log_level(self) -> str:
        """Resolve the effective log level (verbose overrides)."""
        if self.verbose:
            return "DEBUG"
        return self.log_level
