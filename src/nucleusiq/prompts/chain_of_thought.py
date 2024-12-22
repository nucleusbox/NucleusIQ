# src/nucleusiq/prompts/chain_of_thought.py

from typing import Optional
from pydantic import Field, model_validator, ValidationInfo
from nucleusiq.prompts.base import BasePrompt


class ChainOfThoughtPrompt(BasePrompt):
    """
    Implements Chain-of-Thought Prompting.
    Encourages step-by-step reasoning in LLM responses.
    """

    # Specific fields for Chain-of-Thought Prompting
    use_cot: bool = Field(
        default=True,
        description="Whether to enforce Chain-of-Thought reasoning. Must always be True for ChainOfThoughtPrompt."
    )
    cot_instruction: str = Field(
        default="Let's think step by step.",
        description="The Chain-of-Thought instruction to append."
    )

    @property
    def technique_name(self) -> str:
        return "chain_of_thought"

    # Override default template, input_variables, and optional_variables
    template: str = Field(
        default="{system}\n\n{user}\n\n{cot_instruction}",
        description="Default template for Chain-of-Thought Prompting."
    )
    input_variables: list = Field(
        default_factory=lambda: ["system", "user"],
        description="Required input variables for Chain-of-Thought Prompting."
    )
    optional_variables: list = Field(
        default_factory=lambda: ["cot_instruction"],
        description="Optional variables for Chain-of-Thought Prompting."
    )

    def configure(
        self,
        system: Optional[str] = None,
        user: Optional[str] = None,
        use_cot: Optional[bool] = None,
        cot_instruction: Optional[str] = None
    ) -> "ChainOfThoughtPrompt":
        """
        Configure multiple parameters at once.

        Args:
            system (Optional[str]): System prompt.
            user (Optional[str]): User prompt.
            use_cot (Optional[bool]): Enable Chain-of-Thought. Must be True.
            cot_instruction (Optional[str]): CoT instruction.

        Returns:
            ChainOfThoughtPrompt: The updated prompt instance.

        Raises:
            ValueError: If use_cot is set to False.
        """
        # Handle use_cot: must be True
        if use_cot is not None:
            if not isinstance(use_cot, bool):
                raise ValueError(f"use_cot must be a boolean, got {type(use_cot)}")
            if not use_cot:
                raise ValueError("use_cot cannot be set to False for ChainOfThoughtPrompt.")
            self.use_cot = use_cot

        # Handle cot_instruction
        if cot_instruction is not None:
            if not isinstance(cot_instruction, str):
                raise ValueError(f"cot_instruction must be a string, got {type(cot_instruction)}")
            self.cot_instruction = cot_instruction

        # Configure other common fields using the base class's configure method
        super().configure(
            system=system,
            user=user
        )

        return self

    @model_validator(mode='after')
    def ensure_valid_fields(cls, model: 'ChainOfThoughtPrompt', info: ValidationInfo) -> 'ChainOfThoughtPrompt':
        """
        Ensures that:
        - use_cot is always True.
        - cot_instruction is a non-empty string when use_cot is True.

        Args:
            model (ChainOfThoughtPrompt): The model instance.
            info (ValidationInfo): Validation information.

        Returns:
            ChainOfThoughtPrompt: The validated and possibly modified model.

        Raises:
            ValueError: If use_cot is False or cot_instruction is invalid.
        """
        if not model.use_cot:
            raise ValueError("use_cot cannot be set to False for ChainOfThoughtPrompt.")

        if model.use_cot and (not isinstance(model.cot_instruction, str) or not model.cot_instruction.strip()):
            model.cot_instruction = "Let's think step by step."

        return model

    def _construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string, appending CoT instruction.
        """
        system_prompt = kwargs.get("system", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "")

        parts = []
        if system_prompt.strip():
            parts.append(system_prompt)
        if user_prompt.strip():
            parts.append(user_prompt)
        if cot_instruction.strip():
            parts.append(cot_instruction)

        return "\n\n".join(parts)
