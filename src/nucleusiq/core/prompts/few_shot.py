# src/nucleusiq/prompts/few_shot.py

from typing import Any, Dict, List

from nucleusiq.prompts.base import BasePrompt
from pydantic import Field


class FewShotPrompt(BasePrompt):
    """
    Implements Few-Shot Prompting.
    Can incorporate Chain-of-Thought (CoT) reasoning within examples.
    """

    # Fields unique to Few-Shot prompting
    examples: List[Dict[str, str]] = Field(
        default_factory=list, description="List of examples with input-output pairs."
    )
    example_separator: str = Field(
        default="\n\n", description="Separator between examples."
    )

    use_cot: bool = Field(
        default=False,
        description="Whether to append a CoT instruction to the system prompt.",
    )
    cot_instruction: str = Field(
        default="", description="The Chain-of-Thought instruction to append."
    )

    @property
    def technique_name(self) -> str:
        return "few_shot"

    # Override defaults for template, input_variables, optional_variables
    template: str = Field(
        default="{system}\n\n{examples}\n\n{context}\n\n{user}\n\n{cot_instruction}",
        description="Default template for Few-Shot Prompting.",
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["system", "user", "examples"],
        description="These three are mandatory by the time we format the prompt.",
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: ["context", "use_cot", "cot_instruction"],
        description="Additional optional fields for Few-Shot.",
    )

    # -----------------------------
    # Methods to add or manage examples
    # -----------------------------
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
        for ex in examples:
            self.add_example(ex["input"], ex["output"])
        return self

    # -----------------------------
    # Configure method
    # -----------------------------
    def configure(
        self,
        system: str | None = None,
        context: str | None = None,
        user: str | None = None,
        use_cot: bool | None = None,
        cot_instruction: str | None = None,
        examples: List[Dict[str, str]] | None = None,
    ) -> "FewShotPrompt":
        """
        Configure multiple parameters at once, including examples.

        Args:
            system (Optional[str]): System prompt.
            context (Optional[str]): Additional context.
            user (Optional[str]): User prompt.
            use_cot (Optional[bool]): Enable Chain-of-Thought reasoning.
            cot_instruction (Optional[str]): The CoT instruction appended if use_cot is True.
            examples (Optional[List[Dict[str, str]]]): If provided, add via `.add_examples()`.

        Returns:
            FewShotPrompt: The updated prompt instance.
        """
        # 1) Determine a safe CoT instruction based on use_cot
        if use_cot is not None:
            if use_cot:
                # If user didn't provide `cot_instruction`, default to "Let's think step by step."
                cot_instruction_safe = (
                    cot_instruction
                    if cot_instruction is not None
                    else "Let's think step by step."
                )
            else:
                # If user sets CoT to false, clear the instruction
                cot_instruction_safe = ""
        else:
            # If `use_cot` is not changed, keep existing logic
            cot_instruction_safe = self.cot_instruction
            if self.use_cot and not self.cot_instruction:
                cot_instruction_safe = "Let's think step by step."

        # 2) Call base .configure() with these fields
        super().configure(
            system=system,
            context=context,
            user=user,
            use_cot=use_cot if use_cot is not None else self.use_cot,
            cot_instruction=cot_instruction_safe,
        )

        # 3) If examples are passed in, add them to self.examples
        if examples:
            self.add_examples(examples)

        return self

    #
    # Overriding format_prompt() to enforce "examples" must not be empty
    #
    def _pre_format_validation(self, combined_vars: Dict[str, Any]) -> None:
        """
        Hook to enforce that 'examples' is non-empty before final prompt creation.
        """
        if not self.examples:  # or combined_vars.get('examples', []) is empty
            raise ValueError(
                "FewShotPrompt requires at least one example (examples list is empty)."
            )

    # -----------------------------
    # Final construction logic
    # -----------------------------
    def _construct_prompt(self, **kwargs) -> str:
        """
        Build the final string with the examples + system + user + CoT.
        By the time we get here, we know 'system', 'user', 'examples' are non-empty
        (unless user is intentionally empty, which is allowed but can lead to errors).
        """
        system_prompt = kwargs.get("system", "")
        context_prompt = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        use_cot_flag = kwargs.get("use_cot", False)
        # If using CoT, we might have a default or provided instruction
        c_instruction = (
            kwargs.get("cot_instruction", "").strip() if use_cot_flag else ""
        )

        # Format the examples
        if self.examples:
            # Join them
            formatted_examples = self.example_separator.join(
                [
                    f"Input: {ex['input']}\nOutput: {ex['output']}"
                    for ex in self.examples
                ]
            )
            # If there's a system prompt, we prepend examples to it
            if system_prompt:
                system_prompt = f"{system_prompt.strip()}\n\n{formatted_examples}"
            else:
                system_prompt = formatted_examples

        # Build the final parts
        parts = []
        if system_prompt:
            parts.append(system_prompt.strip())
        if context_prompt:
            parts.append(context_prompt.strip())
        if user_prompt:
            parts.append(user_prompt.strip())
        if use_cot_flag:
            parts.append(c_instruction)

        return "\n\n".join(parts)
