# src/nucleusiq/prompts/prompt_composer.py

from typing import Dict, Callable, Optional, List, Any
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt


class PromptComposer(BasePrompt):
    """
    A flexible prompt class for user-defined templates & placeholders.
    Supports variable/function mappings for advanced usage.
    """

    # Specific fields for PromptComposer
    examples: Optional[str] = Field(
        default=None,
        description="Few-shot or demonstration examples."
    )
    chain_of_thought: Optional[str] = Field(
        default=None,
        description="Chain-of-thought instructions."
    )
    user_query: Optional[str] = Field(
        default=None,
        description="User's final query."
    )

    variable_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Logical var name -> template placeholder mapping."
    )
    function_mappings: Dict[str, Callable[..., str]] = Field(
        default_factory=dict,
        description="Var -> function for dynamic generation."
    )

    @property
    def technique_name(self) -> str:
        return "prompt_composer"

    # Override default template, input_variables, and optional_variables
    template: str = Field(
        default="",
        description="User-defined template."
    )
    input_variables: List[str] = Field(
        default_factory=list,
        description="Required input variables."
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: ["system", "examples", "chain_of_thought", "user_query"],
        description="Optional variables for PromptComposer."
    )

    def configure(
        self,
        system: Optional[str] = None,
        examples: Optional[str] = None,
        chain_of_thought: Optional[str] = None,
        user_query: Optional[str] = None,
        template: Optional[str] = None
    ) -> "PromptComposer":
        """
        Configure multiple parameters at once.

        Args:
            system: System instructions or role text.
            examples: Few-shot or demonstration examples as a single string.
            chain_of_thought: Additional reasoning or CoT instructions.
            user_query: The user query or final request.
            template: Override the current or default template.

        Returns:
            Self: The updated prompt instance.
        """
        return super().configure(
            system=system,
            examples=examples,
            chain_of_thought=chain_of_thought,
            user_query=user_query,
            template=template
        )

    def _construct_prompt(self, **kwargs) -> str:
        """
        Construct final prompt by:
          1. Apply function mappings for dynamic transformations.
          2. Apply variable mappings for placeholder renaming.
          3. Format the final template with mapped variables.
        """
        # 1. Apply function mappings
        for key, func in self.function_mappings.items():
            if key in kwargs and callable(func):
                kwargs[key] = func(**kwargs)

        # 2. Apply variable mappings
        mapped_vars = {
            self.variable_mappings.get(k, k): v for k, v in kwargs.items()
        }

        # 3. Format the final template
        try:
            return self.template.format(**mapped_vars)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e.args[0]}")
