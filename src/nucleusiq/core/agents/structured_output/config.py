# src/nucleusiq/agents/structured_output/config.py
"""
Configuration for NucleusIQ Structured Output.

OutputSchema is the main configuration class that defines
how structured output should be obtained and validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from .types import OutputMode, SchemaType


@dataclass
class OutputSchema:
    """
    Configuration for structured output extraction.

    OutputSchema defines:
    - What schema to use (Pydantic, dataclass, TypedDict, JSON Schema)
    - How to obtain structured output (native, tool, prompt)
    - How to handle errors (retry, raise, fallback)

    Basic Usage:
        # Just the schema (mode=AUTO, defaults for everything)
        OutputSchema(Person)

    Advanced Usage:
        OutputSchema(
            schema=Person,
            mode=OutputMode.NATIVE,  # Force native structured output
            strict=True,             # Strict schema adherence
            retry_on_error=True,     # Retry on validation failure
            max_retries=2            # Max retry attempts
        )

    Attributes:
        schema: The target schema (Pydantic model, dataclass, TypedDict, or JSON Schema dict)
        mode: How to obtain structured output (AUTO, NATIVE, TOOL, PROMPT)
        strict: Enable strict schema adherence (provider-specific)
        retry_on_error: Whether to retry on validation errors
        max_retries: Maximum number of retry attempts
        error_handler: Custom error handler function
        include_schema_in_prompt: Whether to include schema description in prompt
    """

    schema: SchemaType
    """The target schema for output validation."""

    mode: OutputMode = OutputMode.AUTO
    """How to obtain structured output. AUTO lets NucleusIQ decide."""

    strict: bool = True
    """Enable strict schema adherence. Some providers enforce this."""

    retry_on_error: bool = True
    """Retry with error feedback if validation fails."""

    max_retries: int = 2
    """Maximum retry attempts on validation failure."""

    error_handler: Callable[[Exception], str] | None = None
    """Custom function to format error messages for retry."""

    include_schema_in_prompt: bool = True
    """Include schema description in the prompt for better results."""

    # Internal state
    _resolved_mode: OutputMode | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        # Validate mode is implemented (fail fast with helpful message)
        if self.mode != OutputMode.AUTO:
            OutputMode.validate_mode(self.mode)

    @property
    def schema_name(self) -> str:
        """Get human-readable name of the schema."""
        if isinstance(self.schema, dict):
            return self.schema.get("title", self.schema.get("name", "Schema"))
        if hasattr(self.schema, "__name__"):
            return self.schema.__name__
        return "Schema"

    def get_error_message(self, error: Exception) -> str:
        """
        Get error message to send to LLM for retry.

        Uses custom error_handler if provided, otherwise uses
        the error's format_for_retry() method.
        """
        if self.error_handler:
            return self.error_handler(error)

        # Check for custom format_for_retry method (e.g., StructuredOutputError)
        format_method = getattr(error, "format_for_retry", None)
        if callable(format_method):
            return str(format_method())

        return f"Error: {str(error)}\nPlease correct your response."

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if we should retry after an error."""
        if not self.retry_on_error:
            return False
        if attempt >= self.max_retries:
            return False
        # Check if error is marked as non-retryable (e.g., StructuredOutputError)
        if getattr(error, "retryable", True) is False:
            return False
        return True

    def for_provider(self, provider: str) -> Dict[str, Any]:
        """
        Get provider-specific configuration.

        Args:
            provider: Provider name (openai, anthropic, etc.)

        Returns:
            Provider-specific response_format configuration
        """
        from .parser import schema_to_json

        json_schema = schema_to_json(
            self.schema,
            strict=self.strict,
            for_provider=provider,
        )

        if provider == "openai":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema_name,
                    "strict": self.strict,
                    "schema": json_schema,
                },
            }

        # Generic format for other providers
        return {
            "type": "json",
            "schema": json_schema,
        }


def output_schema(
    schema: SchemaType,
    *,
    mode: OutputMode = OutputMode.AUTO,
    strict: bool = True,
    retry_on_error: bool = True,
    max_retries: int = 2,
) -> OutputSchema:
    """
    Convenience function to create OutputSchema.

    Example:
        from nucleusiq.agents.structured_output import output_schema, OutputMode

        agent = Agent(
            response_format=output_schema(Person, mode=OutputMode.NATIVE)
        )
    """
    return OutputSchema(
        schema=schema,
        mode=mode,
        strict=strict,
        retry_on_error=retry_on_error,
        max_retries=max_retries,
    )
