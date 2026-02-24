# src/nucleusiq/agents/structured_output/errors.py
"""
Error classes for NucleusIQ Structured Output.
"""

from __future__ import annotations

from typing import Any, Dict, List


class StructuredOutputError(Exception):
    """
    Base exception for all structured output errors.

    Attributes:
        message: Error description
        schema_name: Name of the schema that failed
        raw_output: The raw output that failed
        retryable: Whether this error can be retried
    """

    def __init__(
        self,
        message: str,
        *,
        schema_name: str | None = None,
        raw_output: Any = None,
        retryable: bool = True,
    ):
        super().__init__(message)
        self.schema_name = schema_name
        self.raw_output = raw_output
        self.retryable = retryable

    def format_for_retry(self) -> str:
        """Format error message to send back to LLM for retry."""
        return f"Error: {str(self)}\nPlease correct your response."


class SchemaValidationError(StructuredOutputError):
    """
    Raised when output doesn't match the expected schema.

    This occurs when:
    - Pydantic validation fails (missing fields, wrong types)
    - Dataclass instantiation fails
    - JSON doesn't match schema constraints

    Example:
        # Schema expects rating 1-5, LLM returns 10
        SchemaValidationError(
            "rating: Input should be less than or equal to 5",
            schema_name="ProductReview",
            field_errors=[{"field": "rating", "error": "..."}]
        )
    """

    def __init__(
        self,
        message: str,
        *,
        schema_name: str | None = None,
        raw_output: Any = None,
        field_errors: List[Dict[str, Any]] | None = None,
    ):
        super().__init__(
            message,
            schema_name=schema_name,
            raw_output=raw_output,
            retryable=True,
        )
        self.field_errors = field_errors or []

    def format_for_retry(self) -> str:
        """Format validation errors for LLM retry."""
        if self.field_errors:
            errors = "\n".join(
                f"  - {e.get('field', 'unknown')}: {e.get('error', 'invalid')}"
                for e in self.field_errors
            )
            return f"Validation failed for '{self.schema_name}':\n{errors}\nPlease fix and try again."
        return f"Validation failed: {str(self)}\nPlease fix and try again."


class SchemaParseError(StructuredOutputError):
    """
    Raised when output cannot be parsed as JSON.

    This occurs when:
    - LLM returns invalid JSON syntax
    - Response is not in expected format
    - JSON extraction from markdown fails
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: Any = None,
        parse_position: int | None = None,
    ):
        super().__init__(
            message,
            raw_output=raw_output,
            retryable=True,
        )
        self.parse_position = parse_position

    def format_for_retry(self) -> str:
        """Format parse error for LLM retry."""
        return (
            f"Failed to parse JSON: {str(self)}\n"
            "Please respond with valid JSON only, no additional text."
        )


class MultipleOutputError(StructuredOutputError):
    """
    Raised when LLM returns multiple outputs when one was expected.

    This can happen when:
    - Model calls multiple tools when using tool-based extraction
    - Union type schema and model returns multiple matches
    """

    def __init__(
        self,
        message: str,
        *,
        output_names: List[str] | None = None,
        outputs: List[Any] | None = None,
    ):
        super().__init__(
            message,
            retryable=True,
        )
        self.output_names = output_names or []
        self.outputs = outputs or []

    def format_for_retry(self) -> str:
        """Format for LLM retry."""
        names = ", ".join(self.output_names) if self.output_names else "multiple"
        return (
            f"Received {len(self.outputs)} outputs ({names}) but expected only one.\n"
            "Please provide a single structured response."
        )
