# src/nucleusiq/prompts/prompt_composer.py

from typing import Dict, Callable, Optional, List, Any
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt

class PromptComposer(BasePrompt):
    """
    A flexible prompt class for user-defined templates & placeholders.
    Supports variable/function mappings for advanced usage.
    """

    template: str = Field(
        default_factory=lambda: PromptComposer.default_template()
    )
    input_variables: List[str] = Field(
        default_factory=lambda: PromptComposer.default_input_variables()
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: PromptComposer.default_optional_variables()
    )

    # Potential placeholders
    system: Optional[str] = Field(default=None, description="System instructions.")
    examples: Optional[str] = Field(default=None, description="Few-shot or demonstration examples.")
    chain_of_thought: Optional[str] = Field(default=None, description="Chain-of-thought instructions.")
    user_query: Optional[str] = Field(default=None, description="User's final query.")

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

    @staticmethod
    def default_template() -> str:
        """Empty default template."""
        return ""

    @staticmethod
    def default_input_variables() -> List[str]:
        """No strictly required placeholders by default."""
        return []

    @staticmethod
    def default_optional_variables() -> List[str]:
        """Possible placeholders recognized by default."""
        return ["system", "examples", "chain_of_thought", "user_query"]

    def set_parameters(
        self,
        system: Optional[str] = None,
        examples: Optional[str] = None,
        chain_of_thought: Optional[str] = None,
        user_query: Optional[str] = None,
        template: Optional[str] = None
    ) -> "PromptComposer":
        """
        Sets multiple composer parameters in one go.

        Args:
            system: System instructions or role text.
            examples: Few-shot or demonstration examples as a single string.
            chain_of_thought: Additional reasoning or CoT instructions.
            user_query: The user query or final request.
            template: Override the current or default template.

        Returns:
            PromptComposer: This instance with updated fields.
        """
        if system is not None:
            self.system = system
        if examples is not None:
            self.examples = examples
        if chain_of_thought is not None:
            self.chain_of_thought = chain_of_thought
        if user_query is not None:
            self.user_query = user_query
        if template is not None:
            self.template = template
        return self

    def _construct_prompt(self, **kwargs) -> str:
        """
        Construct final prompt by:
          1) function_mappings -> dynamic transform
          2) variable_mappings -> rename placeholders
          3) self.template.format(...)
        """
        # 1) Apply function mappings
        for key, func in self.function_mappings.items():
            if key in kwargs and callable(func):
                kwargs[key] = func(**kwargs)

        # 2) Apply variable mappings
        mapped_vars = {
            self.variable_mappings.get(k, k): v for k, v in kwargs.items()
        }

        # 3) Format the final template
        try:
            return self.template.format(**mapped_vars)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e.args[0]}")
