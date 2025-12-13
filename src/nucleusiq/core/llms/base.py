"""Base language models class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)
from functools import cache 

from pydantic import Field




@cache
def get_tokenizer() -> Any:
    """Get a GPT-2 tokenizer instance."""
    try:
        from tokenizers import GPT2TokenizerFast

    except ImportError:
        raise ImportError("tokenizers library is required for this model. Please install it with 'pip install tokenizers'.")
    return GPT2TokenizerFast.from_pretrained("gpt2")

def _get_token_ids_default_method(text: str) -> list[int]:
    """Encode the text into token IDs."""
    # get the cached tokenizer
    tokenizer = get_tokenizer()

    # tokenize the text using the GPT-2 tokenizer
    return tokenizer.encode(text)

class BaseLanguageModel(ABC):
    """
    Base class for all language models in NucleusIQ.

    This class provides a common interface and shared functionality for all language models.
    It is not intended to be instantiated directly.
    """
    metadata: Optional[dict[str, Any]] = Field(default=None, exclude=True)

    custom_get_token_ids: Optional[Callable[[str], list[int]]] = Field(
        default=None, exclude=True
    )


    def get_token_ids(self, text: str) -> list[int]:
        """Return the ordered ids of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of ids corresponding to the tokens in the text, in order they occur
                in the text.
        """
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        return _get_token_ids_default_method(text)
    
    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input fits in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        return len(self.get_token_ids(text))

    

