# src/nucleusiq/prompts/zero_shot.py

from typing import List, Optional
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

    @classmethod
    def default_template(cls) -> str:
        return "{system}\n\n{context}\n\n{user}\n\n{cot_instruction}"

    @classmethod
    def default_input_variables(cls) -> List[str]:
        return ["system", "user"]  # 'context' moved to optional

    @classmethod
    def default_optional_variables(cls) -> List[str]:
        return ["context", "use_cot", "cot_instruction"]

    def construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string, appending CoT instruction if enabled.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "") if self.use_cot else ""

        return f"{system_prompt}\n\n{context}\n\n{user_prompt}\n\n{cot_instruction}"
