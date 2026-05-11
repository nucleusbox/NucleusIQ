"""Structured-output helpers for Ollama ``format``."""

from nucleusiq_ollama.structured_output.builder import build_ollama_format
from nucleusiq_ollama.structured_output.parser import parse_response

__all__ = ["build_ollama_format", "parse_response"]
