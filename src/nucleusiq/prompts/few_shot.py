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

    @staticmethod
    def default_template() -> str:
        return "{system}\n\n{examples}\n\n{context}\n\n{user}\n\n{cot_instruction}"

    @staticmethod
    def default_input_variables() -> List[str]:
        return ["system", "user", "examples"]  # Included 'examples'

    @staticmethod
    def default_optional_variables() -> List[str]:
        return ["context", "use_cot", "cot_instruction"]

    # Provide default values for required fields using default_factory
    template: str = Field(default_factory=default_template)
    input_variables: List[str] = Field(default_factory=default_input_variables)
    optional_variables: List[str] = Field(default_factory=default_optional_variables)

    def add_example(self, input_text: str, output_text: str) -> 'FewShotPrompt':
        """
        Adds an example to the FewShotPrompt.

        Args:
            input_text (str): The input part of the example.
            output_text (str): The expected output for the input.

        Returns:
            FewShotPrompt: The updated prompt instance.
        """
        self.examples.append({"input": input_text, "output": output_text})
        return self

    def add_examples(self, examples: List[Dict[str, str]]) -> 'FewShotPrompt':
        """
        Adds multiple examples to the FewShotPrompt.

        Args:
            examples (List[Dict[str, str]]): A list of input-output pairs.

        Returns:
            FewShotPrompt: The updated prompt instance.
        """
        for example in examples:
            self.add_example(example['input'], example['output'])
        return self

    def set_parameters(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None,
        use_cot: Optional[bool] = None,
        cot_instruction: Optional[str] = None
    ) -> 'FewShotPrompt':
        """
        Sets the parameters for the FewShotPrompt.

        Args:
            system (Optional[str]): System prompt.
            context (Optional[str]): Additional context.
            user (Optional[str]): User prompt.
            use_cot (Optional[bool]): Enable Chain-of-Thought.
            cot_instruction (Optional[str]): CoT instruction.

        Returns:
            FewShotPrompt: The updated prompt instance.
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
        Constructs the prompt string with examples, appending CoT instruction if enabled.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "") if kwargs.get("use_cot", False) else ""

        if self.examples:
            formatted_examples = self.example_separator.join(
                [f"Input: {example['input']}\nOutput: {example['output']}" for example in self.examples]
            )
            if system_prompt:
                system_prompt = f"{formatted_examples}\n\n{system_prompt}"
            else:
                system_prompt = formatted_examples

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
