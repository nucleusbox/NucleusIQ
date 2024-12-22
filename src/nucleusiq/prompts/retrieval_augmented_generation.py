# src/nucleusiq/prompts/retrieval_augmented_generation.py

from typing import List, Optional
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt


class RetrievalAugmentedGenerationPrompt(BasePrompt):
    """
    Implements Retrieval-Augmented Generation Prompting.
    Incorporates external info from a knowledge base to enhance responses.
    """

    @property
    def technique_name(self) -> str:
        return "retrieval_augmented_generation"

    # Override default template, input_variables, and optional_variables
    template: str = Field(
        default="{system}\n\n{context}\n\n{user}",
        description="Default template for Retrieval-Augmented Generation Prompting."
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["system", "user"],
        description="Required input variables for Retrieval-Augmented Generation."
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: ["context"],
        description="Optional variables for Retrieval-Augmented Generation."
    )

    def configure(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None
    ) -> "RetrievalAugmentedGenerationPrompt":
        """
        Configure multiple parameters at once.

        Args:
            system: System instructions.
            context: Retrieved knowledge base info.
            user: User query.

        Returns:
            Self: The updated prompt instance.
        """
        return super().configure(
            system=system,
            context=context,
            user=user
        )

    def _construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt by appending system, context, and user prompts.
        """
        system_prompt = kwargs.get("system", "") or ""
        context = kwargs.get("context", "") or ""
        user_prompt = kwargs.get("user", "") or ""

        parts = []
        if system_prompt.strip():
            parts.append(system_prompt)
        if context.strip():
            parts.append(context)
        if user_prompt.strip():
            parts.append(user_prompt)

        return "\n\n".join(parts)
