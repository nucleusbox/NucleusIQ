# src/nucleusiq/prompts/chain_of_thought.py

from typing import Any, Dict

from nucleusiq.prompts.base import BasePrompt
from pydantic import Field


class ChainOfThoughtPrompt(BasePrompt):
    """
    Implements Chain-of-Thought Prompting.
    Encourages step-by-step reasoning in LLM responses.
    """

    # Specific fields for Chain-of-Thought Prompting
    use_cot: bool = Field(
        default=True,
        description="Whether to enforce Chain-of-Thought reasoning. Must always be True for ChainOfThoughtPrompt.",
    )
    cot_instruction: str = Field(
        default="Let's think step by step.",
        description="The Chain-of-Thought instruction to append if use_cot is True.",
    )

    @property
    def technique_name(self) -> str:
        return "chain_of_thought"

    # Override default template, input_variables, and optional_variables
    template: str = Field(
        default="{system}\n\n{user}\n\n{cot_instruction}",
        description="Default template for Chain-of-Thought Prompting.",
    )
    input_variables: list = Field(
        default_factory=lambda: ["system", "user"],
        description="Required input variables for Chain-of-Thought Prompting.",
    )
    optional_variables: list = Field(
        default_factory=lambda: ["cot_instruction", "use_cot"],
        description="Optional variables for Chain-of-Thought Prompting.",
    )

    #
    # Overriding configure to keep 'use_cot' always True
    #
    def configure(
        self,
        system: str | None = None,
        user: str | None = None,
        use_cot: bool | None = None,
        cot_instruction: str | None = None,
    ) -> "ChainOfThoughtPrompt":
        """
        Configure multiple parameters at once.

        Args:
            system: System prompt.
            user: User prompt.
            use_cot: Must be True for ChainOfThoughtPrompt (cannot be set to False).
            cot_instruction: The CoT instruction appended if use_cot is True.

        Raises:
            ValueError: If use_cot is explicitly set to False.
        """
        # Validate use_cot
        if use_cot is not None:
            if use_cot is False:
                raise ValueError(
                    "use_cot cannot be set to False for ChainOfThoughtPrompt."
                )
            self.use_cot = True  # Force True if user tries to pass True

        # If user provided a new cot_instruction
        if cot_instruction is not None:
            self.cot_instruction = cot_instruction

        # Configure the other fields with the base method
        super().configure(
            system=system,
            user=user,
        )
        return self

    #
    # Hook method to ensure final validation
    #
    def _pre_format_validation(self, combined_vars: Dict[str, Any]) -> None:
        """
        Subclass-specific validation:
          - use_cot must be True
          - cot_instruction must be non-empty if use_cot is True
        """
        # If the user tries to set it false or we got None, forcibly ensure it's True
        if not self.use_cot:
            raise ValueError(
                "ChainOfThoughtPrompt requires use_cot=True (cannot be False)."
            )

        # If CoT is true but cot_instruction is empty, default it
        c_instr = combined_vars.get("cot_instruction", "").strip()
        if not c_instr:
            # auto-fix or raise an error
            combined_vars["cot_instruction"] = "Let's think step by step."

    def _construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string, appending the CoT instruction if use_cot is True.
        """
        system_prompt = kwargs.get("system", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "")

        parts = []
        if system_prompt.strip():
            parts.append(system_prompt.strip())
        if user_prompt.strip():
            parts.append(user_prompt.strip())
        if cot_instruction.strip():  # If 'use_cot' is True
            parts.append(cot_instruction.strip())

        return "\n\n".join(parts)
