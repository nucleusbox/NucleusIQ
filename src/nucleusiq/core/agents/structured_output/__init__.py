# src/nucleusiq/agents/structured_output/__init__.py
"""
NucleusIQ Structured Output

Simple, intelligent structured output handling for agents.
Just pass your schema - NucleusIQ handles the rest.

Design Philosophy:
- Simple: Pass a Pydantic/dataclass/TypedDict, get validated output
- Intelligent: Framework auto-selects best method based on model
- Consistent: Same API regardless of underlying implementation

Basic Usage:
    from pydantic import BaseModel
    from nucleusiq.agents import Agent
    
    class Person(BaseModel):
        name: str
        age: int
    
    # Just pass the schema - that's it!
    agent = Agent(
        name="Extractor",
        response_format=Person
    )
    
    result = await agent.execute(task)
    person = result.output  # Person instance

Advanced Usage:
    from nucleusiq.agents.structured_output import OutputSchema, OutputMode
    
    # Explicit control over output handling
    agent = Agent(
        name="Extractor",
        response_format=OutputSchema(
            schema=Person,
            mode=OutputMode.NATIVE,  # Use provider's native structured output
            strict=True,             # Strict schema adherence
            retry_on_error=True      # Retry on validation failure
        )
    )
"""

from .types import OutputMode, SchemaType
from .config import OutputSchema
from .errors import (
    StructuredOutputError,
    SchemaValidationError,
    SchemaParseError,
    MultipleOutputError,
)
from .parser import (
    parse_schema,
    validate_output,
    schema_to_json,
)
from .resolver import resolve_output_config, get_provider_from_llm

__all__ = [
    # Main config
    "OutputSchema",
    "OutputMode",
    # Types
    "SchemaType",
    # Errors  
    "StructuredOutputError",
    "SchemaValidationError", 
    "SchemaParseError",
    "MultipleOutputError",
    # Utilities
    "parse_schema",
    "validate_output",
    "schema_to_json",
    "resolve_output_config",
    "get_provider_from_llm",
]

