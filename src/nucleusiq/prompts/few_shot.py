# src/nucleusiq/prompts/few_shot.py

from typing import List, Dict, Optional
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt


class FewShotPrompt(BasePrompt):
    """
    Implements Few-Shot Prompting.
    Can incorporate Chain-of-Thought (CoT) reasoning within examples.
    """

    # Specific fields for Few-Shot Prompting
    examples: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of examples with input-output pairs."
    )
    example_separator: str = Field(
        default="\n\n",
        description="Separator between examples."
    )

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
        return "few_shot"

    # Override default template, input_variables, and optional_variables
    template: str = Field(
        default="{system}\n\n{examples}\n\n{context}\n\n{user}\n\n{cot_instruction}",
        description="Default template for Few-Shot Prompting."
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["system", "user", "examples"],
        description="Required input variables for Few-Shot Prompting."
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: ["context", "use_cot", "cot_instruction"],
        description="Optional variables for Few-Shot Prompting."
    )

    def add_example(self, input_text: str, output_text: str) -> "FewShotPrompt":
        """
        Adds a single example to the FewShotPrompt.

        Args:
            input_text (str): The input part of the example.
            output_text (str): The expected output for the input.

        Returns:
            FewShotPrompt: The updated prompt instance.
        """
        self.examples.append({"input": input_text, "output": output_text})
        return self

    def add_examples(self, examples: List[Dict[str, str]]) -> "FewShotPrompt":
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

    def configure(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None,
        use_cot: Optional[bool] = None,
        cot_instruction: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> "FewShotPrompt":
        """
        Configure multiple parameters at once, including adding examples.

        Args:
            system (Optional[str]): System prompt.
            context (Optional[str]): Additional context.
            user (Optional[str]): User prompt.
            use_cot (Optional[bool]): Enable Chain-of-Thought.
            cot_instruction (Optional[str]): CoT instruction.
            examples (Optional[List[Dict[str, str]]]): List of examples to add.

        Returns:
            FewShotPrompt: The updated prompt instance.
        """
        # Determine safe cot_instruction based on use_cot
        if use_cot is not None:
            if use_cot:
                cot_instruction_safe = cot_instruction if cot_instruction is not None else "Let's think step by step."
            else:
                cot_instruction_safe = ""
        else:
            # If use_cot is not being updated, keep existing logic
            cot_instruction_safe = self.cot_instruction
            if self.use_cot and not self.cot_instruction:
                cot_instruction_safe = "Let's think step by step."

        # Configure common fields using the base class's configure method
        super().configure(
            system=system,
            context=context,
            user=user,
            use_cot=use_cot,
            cot_instruction=cot_instruction_safe
        )

        # If examples are provided, add them using the existing add_examples method
        if examples:
            self.add_examples(examples)

        return self

    def _construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string with examples, appending CoT instruction if enabled.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        cot_instruction = kwargs.get("cot_instruction", "") if kwargs.get("use_cot", False) else ""

        # Format examples if they exist
        if self.examples:
            formatted_examples = self.example_separator.join(
                [f"Input: {example['input']}\nOutput: {example['output']}" for example in self.examples]
            )
            if system_prompt.strip():
                # Prepend examples to the system prompt
                system_prompt = f"{formatted_examples}\n\n{system_prompt}"
            else:
                system_prompt = formatted_examples

        # Assemble the final prompt parts, skipping empty sections
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
