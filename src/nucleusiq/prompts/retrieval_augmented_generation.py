# src/nucleusiq/prompts/retrieval_augmented_generation.py

from typing import List, Dict, Optional
from pydantic import Field
from nucleusiq.prompts.base import BasePrompt

class RetrievalAugmentedGenerationPrompt(BasePrompt):
    """
    Implements Retrieval-Augmented Generation Prompting.
    Incorporates external info from a knowledge base to enhance responses.
    """

    template: str = Field(
        default_factory=lambda: RetrievalAugmentedGenerationPrompt.default_template()
    )
    input_variables: List[str] = Field(
        default_factory=lambda: RetrievalAugmentedGenerationPrompt.default_input_variables()
    )
    optional_variables: List[str] = Field(
        default_factory=lambda: RetrievalAugmentedGenerationPrompt.default_optional_variables()
    )

    system: Optional[str] = Field(default=None, description="System prompt or instructions.")
    context: Optional[str] = Field(default=None, description="Knowledge base context.")
    user: Optional[str] = Field(default=None, description="User prompt or query.")

    @property
    def technique_name(self) -> str:
        return "retrieval_augmented_generation"

    @staticmethod
    def default_template() -> str:
        """Default template for RAG."""
        return "{system}\n\n{context}\n\n{user}"

    @staticmethod
    def default_input_variables() -> List[str]:
        """'system' and 'user' are required by default."""
        return ["system", "user"]

    @staticmethod
    def default_optional_variables() -> List[str]:
        """Context is optional (could be empty if no retrieval data)."""
        return ["context"]

    def set_parameters(
        self,
        system: Optional[str] = None,
        context: Optional[str] = None,
        user: Optional[str] = None
    ) -> "RetrievalAugmentedGenerationPrompt":
        """
        Sets multiple RAG parameters in one go.

        Args:
            system: System instructions or role.
            context: Retrieved knowledge base info.
            user: The user query.

        Returns:
            RetrievalAugmentedGenerationPrompt: This instance with updated fields.
        """
        if system is not None:
            self.system = system
        if context is not None:
            self.context = context
        if user is not None:
            self.user = user
        return self

    def _construct_prompt(self, **kwargs) -> str:
        system_prompt = kwargs.get("system", "")
        ctx = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")

        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if ctx:
            parts.append(ctx)
        if user_prompt:
            parts.append(user_prompt)

        return "\n\n".join(parts)
