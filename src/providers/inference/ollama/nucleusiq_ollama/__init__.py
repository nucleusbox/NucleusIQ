"""NucleusIQ Ollama provider (local / remote via official ``ollama`` Python SDK)."""

from nucleusiq_ollama._shared.wire import ThinkLevel
from nucleusiq_ollama.llm_params import OllamaLLMParams
from nucleusiq_ollama.nb_ollama.base import BaseOllama
from nucleusiq_ollama.structured_output import build_ollama_format, parse_response
from nucleusiq_ollama.tools import NATIVE_TOOL_TYPES, to_ollama_function_tool

__version__ = "0.1.0a1"

__all__ = [
    "BaseOllama",
    "OllamaLLMParams",
    "ThinkLevel",
    "NATIVE_TOOL_TYPES",
    "build_ollama_format",
    "parse_response",
    "to_ollama_function_tool",
    "__version__",
]
