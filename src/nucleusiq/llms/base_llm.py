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

    def convert_tool_specs(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert BaseTool instances to LLM-specific tool format.
        
        This method should be overridden by LLM providers to convert
        generic BaseTool specs to their own format.
        
        Args:
            tools: List of BaseTool instances or tool specs
            
        Returns:
            List of tool specs in LLM-specific format
        """
        converted = []
        for tool in tools:
            if hasattr(tool, 'get_spec'):
                # BaseTool instance - get generic spec
                spec = tool.get_spec()
                # Convert to LLM-specific format
                converted.append(self._convert_tool_spec(spec))
            else:
                # Already a dict spec - assume it's in correct format
                converted.append(tool)
        return converted
    
    def _convert_tool_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a generic tool spec to LLM-specific format.
        
        Override this in LLM providers to implement conversion.
        Default implementation returns spec as-is.
        
        Args:
            spec: Generic tool spec from BaseTool.get_spec()
            
        Returns:
            LLM-specific tool spec
        """
        return spec

    @abstractmethod
    async def call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]], # change to pydentic model
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

