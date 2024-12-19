# src/nucleusiq/prompts/retrieval_augmented_generation.py

from typing import List, Dict, Optional
from nucleusiq.prompts.base import BasePrompt
from pydantic import Field

class RetrievalAugmentedGenerationPrompt(BasePrompt):
    """
    Implements Retrieval Augmented Generation Prompting.
    """

    system: Optional[str] = Field(default=None, description="System prompt including instructions.")
    context: Optional[str] = Field(default=None, description="Context retrieved from the knowledge base.")
    user: Optional[str] = Field(default=None, description="User prompt or query.")

    @property
    def technique_name(self) -> str:
        return "retrieval_augmented_generation"

    @classmethod
    def default_template(cls) -> str:
        return "{system}\n\n{context}\n\n{user}"

    @classmethod
    def default_input_variables(cls) -> List[str]:
        return ["system", "user"]  # Removed 'context' from input_variables

    @classmethod
    def default_optional_variables(cls) -> List[str]:
        return ["context"]  # Added 'context' to optional_variables

    def construct_prompt(self, **kwargs) -> str:
        """
        Constructs the prompt string by combining system instructions, context, and user query.
        """
        system_prompt = kwargs.get("system", "")
        context = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")

        return f"{system_prompt}\n\n{context}\n\n{user_prompt}"
