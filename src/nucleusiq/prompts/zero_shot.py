# src/nucleusiq/prompts/zero_shot.py

from typing import List, Optional, Dict
from nucleusiq.prompts.base import BasePrompt
from pydantic import Field


class ZeroShotPrompt(BasePrompt):
    """
    Implements Zero-Shot Prompting.
    Can be enhanced with Chain-of-Thought (CoT) instructions.
    """

    system: Optional[str] = Field(default=None, description="System prompt including instructions.")
    context: Optional[str] = Field(default=None, description="Additional context or background information.")
    user: Optional[str] = Field(default=None, description="User prompt or query.")

    use_cot: bool = Field(default=False, description="Whether to append a CoT instruction to the system prompt.")
    cot_instruction: str = Field(default="", description="The Chain-of-Thought instruction to append.")

    @property
    def technique_name(self) -> str:
        return "zero_shot"

    @staticmethod
    def default_template() -> str:
        return "{system}\n\n{context}\n\n{user}\n\n{cot_instruction}"

    @staticmethod
    def default_input_variables() -> List[str]:
        return ["system", "user"]

    @staticmethod
    def default_optional_variables() -> List[str]:
        return ["context", "use_cot", "cot_instruction"]

    # Provide default values for required fields using default_factory
    template: str = Field(default_factory=default_template)
    input_variables: List[str] = Field(default_factory=default_input_variables)
    optional_variables: List[str] = Field(default_factory=default_optional_variables)

    def set_parameters(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None,
        use_cot: Optional[bool] = None,
        cot_instruction: Optional[str] = None
    ) -> 'ZeroShotPrompt':
        """
        Sets the parameters for the ZeroShotPrompt.

        Args:
            system (Optional[str]): System prompt.
            context (Optional[str]): Additional context.
            user (Optional[str]): User prompt.
            use_cot (Optional[bool]): Enable Chain-of-Thought.
            cot_instruction (Optional[str]): CoT instruction.

        Returns:
            ZeroShotPrompt: The updated prompt instance.
        """
        if system is not None:
            self.system = system
        if context is not None:
            self.context = context
        if user is not None:
            self.user = user
        if use_cot is not None:
            self.use_cot = use_cot
        if cot_instruction is not None:
            self.cot_instruction = cot_instruction
        return self

    def _construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string, appending CoT instruction if enabled.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "") if kwargs.get("use_cot", False) else ""

        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        if context:
            prompt_parts.append(context)
        if user_prompt:
            prompt_parts.append(user_prompt)
        if cot_instruction:
            prompt_parts.append(cot_instruction)

        return "\n\n".join(prompt_parts)
