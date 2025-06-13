# File: src/nucleusiq/llms/base_llm.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from typing_extensions import override
from nucleusiq.llms.base import BaseLanguageModel

class BaseLLM(BaseLanguageModel, ABC):
    """
    Abstract base class for Language Model adapters.

    Subclasses must implement `call()`, which accepts:
      - model: the model name or identifier
      - messages: list of {'role': str, 'content': str} dicts
      - tools: optional list of function specs

    and returns an object with a `.choices` list, each having a `.message`
    attribute containing either `.content` or a `.function_call` dict.
    """

    @abstractmethod
    async def call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None
    ) -> Any:
        """
        Sends messages (and optional function specs) to the model and returns
        a response object with a `.choices` list. Each choice should have a
        `.message` attribute with:
          - `.content` (str) for normal completions,
          - or `.function_call` (dict) when the model decides to call a function.
        """
        raise NotImplementedError

