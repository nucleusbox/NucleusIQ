"""NucleusIQ Gemini Provider.

Google Gemini integration for NucleusIQ agents.

Install:
    pip install nucleusiq-gemini

Quick Start::

    from nucleusiq_gemini import BaseGemini, GeminiLLMParams, GeminiTool

    llm = BaseGemini(model_name="gemini-2.5-flash")
    response = await llm.call(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

__version__ = "0.2.5"

from nucleusiq_gemini.llm_params import (
    GeminiLLMParams,
    GeminiSafetySettings,
    GeminiThinkingConfig,
)
from nucleusiq_gemini.nb_gemini.base import BaseGemini
from nucleusiq_gemini.tools.gemini_tool import GeminiTool

__all__ = [
    "BaseGemini",
    "GeminiLLMParams",
    "GeminiSafetySettings",
    "GeminiThinkingConfig",
    "GeminiTool",
]
