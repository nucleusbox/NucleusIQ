"""Prompt engineering error hierarchy.

Hierarchy::

    NucleusIQError
    └── PromptError
        ├── PromptTemplateError    — template rendering / variable substitution failed
        ├── PromptConfigError      — invalid prompt technique configuration
        └── PromptGenerationError  — auto-CoT or meta-prompt LLM call failed
"""

from __future__ import annotations

from nucleusiq.errors.base import NucleusIQError

__all__ = [
    "PromptError",
    "PromptTemplateError",
    "PromptConfigError",
    "PromptGenerationError",
]


class PromptError(NucleusIQError):
    """Base exception for prompt engineering errors.

    Attributes:
        technique: Prompt technique name (e.g. "chain_of_thought").
        template_name: Template identifier, if applicable.
    """

    def __init__(
        self,
        message: str = "",
        *,
        technique: str | None = None,
        template_name: str | None = None,
    ) -> None:
        self.technique = technique
        self.template_name = template_name
        super().__init__(message)


class PromptTemplateError(PromptError):
    """Template rendering or variable substitution failed."""


class PromptConfigError(PromptError):
    """Invalid prompt technique configuration."""


class PromptGenerationError(PromptError):
    """Auto-CoT or meta-prompt LLM call failed during prompt generation."""
