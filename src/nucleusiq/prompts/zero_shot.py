# src/nucleusiq/prompts/zero_shot.py

from typing import List, Optional
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt


class ZeroShotPrompt(BasePrompt):
    """
    Implements Zero-Shot Prompting.
    Can be enhanced with Chain-of-Thought (CoT) instructions.
    """

    # Specific fields for Zero-Shot Prompting
    use_cot: bool = Field(
        default=False,
        description="Whether to append a CoT instruction to the system prompt."
    )
    cot_instruction: str = Field(
        default="",
        description="The Chain-of-Thought instruction to append."
    )

    @property
    def technique_name(self) -> str:
        return "zero_shot"

    # Override default template, input_variables, and optional_variables
    template: str = Field(
        default="{system}\n\n{context}\n\n{user}\n\n{cot_instruction}",
        description="Default template for Zero-Shot Prompting."
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["system", "user"],
        description="Required input variables for Zero-Shot Prompting."
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: ["context", "use_cot", "cot_instruction"],
        description="Optional variables for Zero-Shot Prompting."
    )

    def configure(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None,
        use_cot: Optional[bool] = None,
        cot_instruction: Optional[str] = None
    ) -> "ZeroShotPrompt":
        """
        Configure multiple parameters at once.

        Args:
            system: System prompt.
            context: Additional context.
            user: User prompt.
            use_cot: Enable Chain-of-Thought.
            cot_instruction: CoT instruction.

        Returns:
            Self: The updated prompt instance.
        """
        return super().configure(
            system=system,
            context=context,
            user=user,
            use_cot=use_cot,
            cot_instruction=cot_instruction
        )

    def _construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string, appending CoT instruction if enabled.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "") if kwargs.get("use_cot", False) else ""

        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if context:
            parts.append(context)
        if user_prompt:
            parts.append(user_prompt)
        if cot_instruction:
            parts.append(cot_instruction)

        return "\n\n".join(parts)
