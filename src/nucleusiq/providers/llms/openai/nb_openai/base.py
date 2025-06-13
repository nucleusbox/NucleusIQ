from __future__ import annotations

import logging

import openai
import tiktoken
from nucleusiq.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class BaseOpenAI(BaseLLM):
    """
    Base class for OpenAI LLMs, providing common functionality.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.encoding = tiktoken.encoding_for_model(model_name)
        self._client = openai
        self._client.api_key = kwargs.get("api_key", None)
        self._client.api_base = kwargs.get("api_base", "https://api.openai.com/v1")             
        self._client.api_type = kwargs.get("api_type", "openai")
        self._client.api_version = kwargs.get("api_version", None)      
        self._client.organization = kwargs.get("organization", None)
        self._client.timeout = kwargs.get("timeout", 60)

    def get_client(self) -> openai.OpenAI:
        """
        Returns the OpenAI client instance.
        """
        return self._client
    def get_encoding(self) -> tiktoken.Encoding:
        """
        Returns the encoding instance for the model.
        """
        return self.encoding            
    def get_model_name(self) -> str:
        """ Returns the name of the model."""
        return self.model_name  
    
    @staticmethod
    def modelname_to_contextsize(modelname: str) -> int:
        """
        Returns the context size for the given model name.
        This is a placeholder implementation; actual logic may vary.
        """
        # Example mapping, should be replaced with actual model context sizes
        context_sizes = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
        }
        return context_sizes.get(modelname, 4096)