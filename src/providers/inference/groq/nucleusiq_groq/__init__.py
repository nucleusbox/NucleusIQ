"""NucleusIQ Groq provider (OpenAI-compatible Chat Completions on Groq)."""

from nucleusiq_groq.llm_params import GroqLLMParams
from nucleusiq_groq.nb_groq.base import BaseGroq
from nucleusiq_groq.structured_output import build_response_format, parse_response
from nucleusiq_groq.tools import NATIVE_TOOL_TYPES, to_openai_function_tool

__version__ = "0.1.0b1"

__all__ = [
    "BaseGroq",
    "GroqLLMParams",
    "NATIVE_TOOL_TYPES",
    "build_response_format",
    "parse_response",
    "to_openai_function_tool",
    "__version__",
]
