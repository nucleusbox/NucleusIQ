# src/nucleusiq/prompts/few_shot.py

from typing import List, Dict, Optional
from nucleusiq.prompts.base import BasePrompt
from pydantic import Field

class FewShotPrompt(BasePrompt):
    """
    Implements Few-Shot Prompting.
    Can incorporate Chain-of-Thought (CoT) reasoning within examples.
    """

    system: Optional[str] = Field(default=None, description="System prompt including instructions.")
    context: Optional[str] = Field(default=None, description="Additional context or background information.")
    user: Optional[str] = Field(default=None, description="User prompt or query.")

    examples: List[Dict[str, str]] = Field(default_factory=list, description="List of examples with input-output pairs.")
    example_separator: str = Field(default="\n\n", description="Separator between examples.")

    use_cot: bool = Field(default=False, description="Whether to append a CoT instruction to the system prompt.")
    cot_instruction: str = Field(default="", description="The Chain-of-Thought instruction to append.")

    @property
    def technique_name(self) -> str:
        return "few_shot"

    @classmethod
    def default_template(cls) -> str:
        return "{system}\n\n{context}\n\n{examples}\n\n{user}\n\n{cot_instruction}"

    @classmethod
    def default_input_variables(cls) -> List[str]:
        return ["system", "user", "examples"]  # Included 'examples'

    @classmethod
    def default_optional_variables(cls) -> List[str]:
        return ["context", "use_cot", "cot_instruction"]

    def add_example(self, input_text: str, output_text: str) -> 'FewShotPrompt':
        """
        Adds an example to the FewShotPrompt.
        """
        self.examples.append({"input": input_text, "output": output_text})
        return self

    def construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string with examples, appending CoT instruction if enabled.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "") if self.use_cot else ""

        if self.examples:
            formatted_examples = self.example_separator.join(
                [f"Input: {example['input']}\nOutput: {example['output']}" for example in self.examples]
            )
            if system_prompt:
                system_prompt = f"{formatted_examples}\n\n{system_prompt}"
            else:
                system_prompt = formatted_examples

        return f"{system_prompt}\n\n{context}\n\n{user_prompt}\n\n{cot_instruction}"
