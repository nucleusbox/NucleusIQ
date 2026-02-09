# src/nucleusiq/agents/structured_output/resolver.py
"""
Resolver for NucleusIQ Structured Output.

Handles automatic mode selection based on model capabilities.
"""

from __future__ import annotations

from typing import Any, Optional, Type, Union

from .types import OutputMode, SchemaType
from .config import OutputSchema


# Models known to support native structured output
NATIVE_SUPPORT = {
    "openai": ["gpt-4o", "gpt-4-turbo", "gpt-5", "o1", "o3"],
    "anthropic": ["claude-3", "claude-4"],
    "google": ["gemini"],
    "xai": ["grok"],
}


def supports_native_output(model_name: str, provider: Optional[str] = None) -> bool:
    """
    Check if a model supports native structured output.
    
    Args:
        model_name: Model name/identifier
        provider: Provider name (optional, for optimization)
        
    Returns:
        True if model supports native structured output
    """
    model_lower = model_name.lower()
    
    # Check all known providers
    for prov, prefixes in NATIVE_SUPPORT.items():
        if provider and prov != provider.lower():
            continue
        for prefix in prefixes:
            if model_lower.startswith(prefix):
                return True
    
    return False


def resolve_output_config(
    response_format: Any,
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[OutputSchema]:
    """
    Resolve response_format to an OutputSchema configuration.
    
    This is the main entry point for handling response_format.
    It converts various input types to a standardized OutputSchema.
    
    Args:
        response_format: User-provided response format, can be:
            - OutputSchema: Already configured, return as-is
            - Type (Pydantic, dataclass, TypedDict): Wrap in OutputSchema
            - Dict: JSON Schema, wrap in OutputSchema
            - None: No structured output
            
        model_name: Model name for capability detection
        provider: Provider name for optimization
        
    Returns:
        OutputSchema configuration or None
        
    Example:
        # From Agent._run_direct()
        config = resolve_output_config(
            self.response_format,
            model_name=self.llm.model_name,
            provider="openai"
        )
    """
    if response_format is None:
        return None
    
    # Already an OutputSchema
    if isinstance(response_format, OutputSchema):
        config = response_format
    
    # Schema type (Pydantic, dataclass, TypedDict, dict)
    elif isinstance(response_format, (type, dict)):
        config = OutputSchema(schema=response_format)
    
    else:
        raise ValueError(
            f"Invalid response_format type: {type(response_format)}. "
            "Expected OutputSchema, Pydantic model, dataclass, TypedDict, or dict."
        )
    
    # Resolve AUTO mode to concrete mode
    if config.mode == OutputMode.AUTO:
        config._resolved_mode = _auto_select_mode(
            model_name=model_name,
            provider=provider,
        )
    else:
        config._resolved_mode = config.mode
    
    return config


def _auto_select_mode(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
) -> OutputMode:
    """
    Auto-select the best output mode based on model capabilities.
    
    Priority:
    1. NATIVE if model supports it (most reliable)
    2. TOOL if model supports tool calling
    3. PROMPT as fallback (works with any model)
    """
    if model_name and supports_native_output(model_name, provider):
        return OutputMode.NATIVE
    
    # TODO: Check if model supports tool calling
    # For now, assume it does if we have a model name
    if model_name:
        return OutputMode.NATIVE  # Most modern models support this
    
    # Fallback to prompt-based extraction
    return OutputMode.PROMPT


def get_provider_from_llm(llm: Any) -> Optional[str]:
    """
    Detect provider from LLM instance.
    
    Args:
        llm: LLM instance
        
    Returns:
        Provider name or None
    """
    if llm is None:
        return None
    
    class_name = type(llm).__name__.lower()
    
    if "openai" in class_name:
        return "openai"
    if "anthropic" in class_name or "claude" in class_name:
        return "anthropic"
    if "google" in class_name or "gemini" in class_name:
        return "google"
    
    return None

