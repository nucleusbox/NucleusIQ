# src/nucleusiq/llms/mock_llm.py

import random
from typing import List, Dict, Optional
from .base_llm import BaseLLM


class MockLLM(BaseLLM):
    """
    Mock Language Model for testing purposes.
    Generates predefined reasoning steps.
    """

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
        Generates a mock completion by returning a random reasoning step.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with 'role' and 'content'.
            max_tokens (int): Maximum number of tokens in the generated completion.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            frequency_penalty (float): Frequency penalty.
            presence_penalty (float): Presence penalty.
            stop (Optional[List[str]]): List of stop sequences.

        Returns:
            str: The generated reasoning chain.
        """
        reasoning_steps = [
            "First, identify the key components involved.",
            "Next, analyze the relationships between these components.",
            "Then, apply the relevant formulas to compute the desired outcome.",
            "Finally, verify the results for accuracy."
        ]
        return random.choice(reasoning_steps)
