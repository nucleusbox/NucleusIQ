"""NucleusIQ error hierarchy — single import point for all framework errors.

Every custom exception inherits from :class:`NucleusIQError`.
Users can import any error from this one location::

    from nucleusiq.errors import NucleusIQError, ToolError, LLMError, PluginHalt

Subsystem ``errors.py`` modules remain in place for internal use.
Re-exports are lazy (via ``__getattr__``) to avoid circular imports.

Hierarchy::

    NucleusIQError
    ├── LLMError               — nucleusiq.llms.errors
    ├── ToolError              — nucleusiq.tools.errors
    ├── AgentError             — nucleusiq.agents.errors
    ├── NucleusMemoryError     — nucleusiq.memory.errors
    ├── PromptError            — nucleusiq.prompts.errors
    ├── StreamingError         — nucleusiq.streaming.errors
    ├── AttachmentError        — nucleusiq.agents.errors
    ├── PluginError            — nucleusiq.plugins.errors
    ├── StructuredOutputError  — nucleusiq.agents.structured_output.errors
    └── WorkspaceSecurityError — nucleusiq.tools.builtin.workspace
"""

from __future__ import annotations

from nucleusiq.errors.base import NucleusIQError

_LAZY_MAP: dict[str, str] = {
    # LLM
    "LLMError": "nucleusiq.llms.errors",
    "AuthenticationError": "nucleusiq.llms.errors",
    "PermissionDeniedError": "nucleusiq.llms.errors",
    "RateLimitError": "nucleusiq.llms.errors",
    "InvalidRequestError": "nucleusiq.llms.errors",
    "ModelNotFoundError": "nucleusiq.llms.errors",
    "ContentFilterError": "nucleusiq.llms.errors",
    "ProviderServerError": "nucleusiq.llms.errors",
    "ProviderConnectionError": "nucleusiq.llms.errors",
    "ContextLengthError": "nucleusiq.llms.errors",
    "ProviderError": "nucleusiq.llms.errors",
    # Tools
    "ToolError": "nucleusiq.tools.errors",
    "ToolExecutionError": "nucleusiq.tools.errors",
    "ToolTimeoutError": "nucleusiq.tools.errors",
    "ToolValidationError": "nucleusiq.tools.errors",
    "ToolPermissionError": "nucleusiq.tools.errors",
    "ToolNotFoundError": "nucleusiq.tools.errors",
    # Agent
    "AgentError": "nucleusiq.agents.errors",
    "AgentConfigError": "nucleusiq.agents.errors",
    "AgentExecutionError": "nucleusiq.agents.errors",
    "AgentTimeoutError": "nucleusiq.agents.errors",
    "AttachmentError": "nucleusiq.agents.errors",
    "AttachmentValidationError": "nucleusiq.agents.errors",
    "AttachmentProcessingError": "nucleusiq.agents.errors",
    "AttachmentUnsupportedError": "nucleusiq.agents.errors",
    # Memory
    "NucleusMemoryError": "nucleusiq.memory.errors",
    "MemoryWriteError": "nucleusiq.memory.errors",
    "MemoryReadError": "nucleusiq.memory.errors",
    "MemoryImportError": "nucleusiq.memory.errors",
    "MemoryCapacityError": "nucleusiq.memory.errors",
    # Prompts
    "PromptError": "nucleusiq.prompts.errors",
    "PromptTemplateError": "nucleusiq.prompts.errors",
    "PromptConfigError": "nucleusiq.prompts.errors",
    "PromptGenerationError": "nucleusiq.prompts.errors",
    # Streaming
    "StreamingError": "nucleusiq.streaming.errors",
    "StreamInterruptedError": "nucleusiq.streaming.errors",
    "StreamOrchestrationError": "nucleusiq.streaming.errors",
    # Plugins
    "PluginError": "nucleusiq.plugins.errors",
    "PluginExecutionError": "nucleusiq.plugins.errors",
    "PluginHalt": "nucleusiq.plugins.errors",
    # Structured Output
    "StructuredOutputError": "nucleusiq.agents.structured_output.errors",
    "SchemaValidationError": "nucleusiq.agents.structured_output.errors",
    "SchemaParseError": "nucleusiq.agents.structured_output.errors",
    "MultipleOutputError": "nucleusiq.agents.structured_output.errors",
}

__all__ = ["NucleusIQError", *_LAZY_MAP.keys()]


def __getattr__(name: str):
    """Lazy import: resolve subsystem errors on first access."""
    if name in _LAZY_MAP:
        import importlib

        module = importlib.import_module(_LAZY_MAP[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'nucleusiq.errors' has no attribute {name!r}")
