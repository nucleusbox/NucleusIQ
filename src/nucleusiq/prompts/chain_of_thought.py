# src/nucleusiq/prompts/chain_of_thought.py

from typing import List, Optional
from nucleusiq.prompts.base import BasePrompt
from pydantic import Field

class ChainOfThoughtPrompt(BasePrompt):
    """
    Implements Chain-of-Thought Prompting as a separate component.
    Can be integrated within other prompt types.
    """

    system: Optional[str] = Field(default=None, description="System prompt including instructions.")
    context: Optional[str] = Field(default=None, description="Additional context or background information.")
    user: Optional[str] = Field(default=None, description="User prompt or query.")

    cot_instruction: str = Field(default="Let's think step by step.", description="The Chain-of-Thought instruction to append.")

    @property
    def technique_name(self) -> str:
        return "chain_of_thought"

    @classmethod
    def default_template(cls) -> str:
        return "{system}\n\n{context}\n\n{user}\n\n{cot_instruction}"

    @classmethod
    def default_input_variables(cls) -> List[str]:
        return ["system", "user"]  # 'context' and 'cot_instruction' moved to optional

    @classmethod
    def default_optional_variables(cls) -> List[str]:
        return ["context", "cot_instruction"]  # Added 'cot_instruction'

    def construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string by appending CoT instruction.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", self.cot_instruction)  # Use default if not provided

        return f"{system_prompt}\n\n{context}\n\n{user_prompt}\n\n{cot_instruction}"
