# src/nucleusiq/prompts/zero_shot.py

from typing import List, Optional
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt


class ZeroShotPrompt(BasePrompt):
    """
    Zero-Shot Prompting, can optionally use CoT.
    """

    # Extra fields specific to ZeroShotPrompt
    use_cot: bool = Field(
        default=False,
        description="If True, append CoT instruction"
    )
    cot_instruction: str = Field(
        default="",
        description="CoT instruction appended if use_cot is True."
    )

    @property
    def technique_name(self) -> str:
        return "zero_shot"

    # Default values for template & input/optional variables
    template: str = Field(
        default="{system}\n\n{context}\n\n{user}\n\n{cot_instruction}",
        description="Default zero-shot template."
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["system", "user"],
        description="system & user are mandatory once we finalize the prompt."
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: ["context", "use_cot", "cot_instruction"],
        description="Additional optional fields."
    )

    def _construct_prompt(self, **kwargs) -> str:
        """
        Actually build the final string from placeholders.
        By the time we reach here, 'system' and 'user' are guaranteed to be non-empty
        because format_prompt() checks them.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")
        # If use_cot is True, then append cot_instruction if not empty
        use_cot_flag = kwargs.get("use_cot", False)
        cot_instr = ""
        if use_cot_flag:
            # if user didn't provide cot_instruction, default is "Let's think step by step."
            c = kwargs.get("cot_instruction", "").strip()
            cot_instr = c if c else "Let's think step by step."

        parts = []
        if system_prompt:
            parts.append(system_prompt.strip())
        if context:
            parts.append(context.strip())
        if user_prompt:
            parts.append(user_prompt.strip())
        if use_cot_flag:
            parts.append(cot_instr)

        return "\n\n".join(parts)

    def configure(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None,
        use_cot: Optional[bool] = None,
        cot_instruction: Optional[str] = None,
    ) -> "ZeroShotPrompt":
        """
        Flexible method to set fields after initialization.
        """
        # If user sets use_cot to True but doesn't provide cot_instruction,
        # we'll handle that fallback in `_construct_prompt`.
        config_args = {}
        if system is not None:
            config_args['system'] = system
        if context is not None:
            config_args['context'] = context
        if user is not None:
            config_args['user'] = user
        if use_cot is not None:
            config_args['use_cot'] = use_cot
        if cot_instruction is not None:
            config_args['cot_instruction'] = cot_instruction

        return super().configure(**config_args)
