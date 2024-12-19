# src/nucleusiq/llms/base_llm.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseLLM(ABC):
    """
    Abstract base class for Language Model adapters.
    """

    @abstractmethod
    def create_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generates a completion based on the provided messages.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with 'role' and 'content'.
            max_tokens (int): Maximum number of tokens in the generated completion.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            frequency_penalty (float): Frequency penalty.
            presence_penalty (float): Presence penalty.
            stop (Optional[List[str]]): List of stop sequences.

        Returns:
            str: The generated completion text.
        """
        pass
