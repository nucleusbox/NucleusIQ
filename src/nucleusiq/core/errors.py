"""Root exception for the NucleusIQ framework.

Every custom exception in NucleusIQ inherits from :class:`NucleusIQError`.
This single root enables framework-wide ``except NucleusIQError`` catch-all
while preserving fine-grained subtypes in each subsystem.

Hierarchy overview (subsystem modules define the subtypes)::

    NucleusIQError
    ├── LLMError               — nucleusiq.llms.errors
    ├── ToolError              — nucleusiq.tools.errors
    ├── AgentError             — nucleusiq.agents.errors
    ├── MemoryError            — nucleusiq.memory.errors
    ├── PromptError            — nucleusiq.prompts.errors
    ├── StreamingError         — nucleusiq.streaming.errors
    ├── AttachmentError        — nucleusiq.agents.errors
    ├── PluginError            — nucleusiq.plugins.errors
    ├── StructuredOutputError  — nucleusiq.agents.structured_output.errors
    └── WorkspaceSecurityError — nucleusiq.tools.builtin.workspace

Usage::

    from nucleusiq.errors import NucleusIQError

    try:
        result = await agent.execute(task)
    except NucleusIQError as e:
        # Catches ANY framework error
        logger.error("Framework error: %s", e)
"""

from __future__ import annotations


class NucleusIQError(Exception):
    """Base exception for all NucleusIQ framework errors."""
