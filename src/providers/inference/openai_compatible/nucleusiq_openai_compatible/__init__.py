"""Generic OpenAI-compatible inference provider for NucleusIQ.

One provider for any server that speaks the OpenAI Chat Completions wire
protocol — vLLM, SGLang, TGI, llama.cpp, LM Studio, Ollama's ``/v1`` shim,
NVIDIA NIM, and OpenAI-compatible clouds (OpenRouter, Together, Fireworks,
DeepInfra, Databricks, LiteLLM, Azure OpenAI v1).

Bring your own model, bring your own key::

    from nucleusiq_openai_compatible import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        base_url="http://gpu-node-1:8000/v1",
        model="gemma-4-27b-it",
        api_key="token-abc123",  # omit for an unauthenticated server
        context_window=32_768,  # recommended; probed otherwise
        engine="vllm",
    )

One instance describes exactly one model on exactly one endpoint, because
``BaseLLM.get_context_window()`` is per-instance and the framework sizes the
whole context budget from it.  Serving several models from one node? Build
several instances.
"""

from .auth import (
    AuthStrategy,
    BearerAuth,
    CredentialSource,
    HeaderAuth,
    NoAuth,
    build_auth,
)
from .capabilities import (
    DEFAULT_CONTEXT_WINDOW,
    ENGINE_PRESETS,
    EngineProfile,
    known_engines,
)
from .config import ResolvedConfig
from .llm_params import OpenAICompatibleLLMParams
from .nb_compat.base import OpenAICompatibleLLM
from .structured_output import DropPolicy, ErrorPolicy, PromptPolicy
from .tools import NATIVE_TOOL_TYPES
from .validation import ValidationReport

__version__ = "0.1.0"

BaseOpenAICompatible = OpenAICompatibleLLM
"""Alias matching the ``Base<Provider>`` naming used by sibling packages."""

__all__ = [
    "__version__",
    # ---- Public entry point ----
    "BaseOpenAICompatible",
    "OpenAICompatibleLLM",
    # ---- Auth strategies (BYOK) ----
    "AuthStrategy",
    "BearerAuth",
    "CredentialSource",
    "HeaderAuth",
    "NoAuth",
    "build_auth",
    # ---- Engine capabilities ----
    "DEFAULT_CONTEXT_WINDOW",
    "ENGINE_PRESETS",
    "EngineProfile",
    "known_engines",
    # ---- Configuration ----
    "OpenAICompatibleLLMParams",
    "ResolvedConfig",
    # ---- Structured-output policies ----
    "DropPolicy",
    "ErrorPolicy",
    "PromptPolicy",
    # ---- Tools & validation ----
    "NATIVE_TOOL_TYPES",
    "ValidationReport",
]
