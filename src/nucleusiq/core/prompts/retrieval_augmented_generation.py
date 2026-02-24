# src/nucleusiq/prompts/retrieval_augmented_generation.py

from typing import Any, Dict, List

from nucleusiq.prompts.base import BasePrompt
from pydantic import Field


class RetrievalAugmentedGenerationPrompt(BasePrompt):
    """
    Implements Retrieval-Augmented Generation Prompting (RAG).
    Incorporates external knowledge-base info to enhance responses.

    Requirements:
     - 'system' is mandatory & non-empty.
     - 'context' is also mandatory & non-empty.
     - 'user' is mandatory & non-empty.
    """

    @property
    def technique_name(self) -> str:
        return "retrieval_augmented_generation"

    # All three are truly required from the base class perspective:
    # The base class will check they are non-empty strings.
    template: str = Field(
        default="{system}\n\n{context}\n\n{user}",
        description="Default template for Retrieval-Augmented Generation.",
    )
    input_variables: List[str] = Field(
        default_factory=lambda: ["system", "context", "user"],
        description="All three fields are required & must be non-empty.",
    )
    # No optional variables here
    optional_variables: List[str] = Field(
        default_factory=list,
        description="No truly optional fields for RAG—context must be non-empty.",
    )

    def configure(
        self,
        system: str | None = None,
        context: str | None = None,
        user: str | None = None,
    ) -> "RetrievalAugmentedGenerationPrompt":
        """
        Configure multiple parameters at once.

        Args:
            system: Non-empty system instructions.
            context: Non-empty knowledge base info.
            user: Non-empty user query.

        Returns:
            Self: The updated prompt instance.
        """
        return super().configure(system=system, context=context, user=user)

    def _pre_format_validation(self, combined_vars: Dict[str, Any]) -> None:
        """
        Subclass hook to confirm 'context' is truly non-empty if provided.
        (But the base class is already set to treat 'context' as a required field
         in input_variables. Thus if it's empty, base will raise an error anyway.)

        We'll add a double-check to ensure the user didn't pass something invalid (like None).
        """
        if "context" not in combined_vars:
            raise ValueError(
                "Missing required field 'context' for RAG prompt. It must be provided and non-empty."
            )
        # If context is None or empty => the base class will actually catch it, but let's explicitly check
        val = combined_vars["context"]
        if val is None:
            raise ValueError(
                "RetrievalAugmentedGenerationPrompt requires 'context' not be None."
            )

        # If it's a string but empty or whitespace, the base class's final check will reject it,
        # but we can add a clearer error message if desired:
        # (optional) if not val.strip(): raise ValueError("RAG prompt requires 'context' be non-empty.")

    def _construct_prompt(self, **kwargs) -> str:
        """
        Construct final prompt by combining system, context, user in separate blocks.
        """
        system_prompt = kwargs.get("system", "")
        context_str = kwargs.get("context", "")
        user_prompt = kwargs.get("user", "")

        parts = []
        if system_prompt.strip():
            parts.append(system_prompt.strip())
        if context_str.strip():
            parts.append(context_str.strip())
        if user_prompt.strip():
            parts.append(user_prompt.strip())

        return "\n\n".join(parts)
