# nucleusiq-ollama

[![PyPI version](https://img.shields.io/pypi/v/nucleusiq-ollama?color=brightgreen)](https://pypi.org/project/nucleusiq-ollama/)
[![PyPI downloads](https://img.shields.io/pypi/dm/nucleusiq-ollama?label=downloads%2Fmonth)](https://pypistats.org/packages/nucleusiq-ollama)
[![Python versions](https://img.shields.io/pypi/pyversions/nucleusiq-ollama)](https://pypi.org/project/nucleusiq-ollama/)

Stable provider for running **NucleusIQ** agents against [Ollama](https://docs.ollama.com/) (local or hosted).

## Install

**0.2.1** — **Development Status :: 5 - Production/Stable**. Requires **`nucleusiq>=0.7.12`** (pulled in automatically by `pip`).

```bash
pip install nucleusiq-ollama
```

Requires a running Ollama server unless you point `OLLAMA_HOST` at a remote instance. Optional: `OLLAMA_API_KEY` for authenticated endpoints.

## Usage

```python
from nucleusiq_ollama import BaseOllama, OllamaLLMParams

llm = BaseOllama(model_name="llama3.2", llm_params=OllamaLLMParams(think=True))
```

Runnable scripts (smoke, Agent DIRECT, streaming) live under **`examples/`** — see [`examples/README.md`](examples/README.md).

See the [Ollama provider guide](https://nucleusbox.github.io/nucleusiq-docs/python/nucleusiq/guides/ollama-provider/) for capability matrix, environment variables, and roadmap.

For Ollama's **`/v1` OpenAI shim** (or any other Chat Completions server) use [`nucleusiq-openai-compatible`](https://github.com/nucleusbox/NucleusIQ/tree/main/src/providers/inference/openai_compatible) instead — this package talks to the native Ollama API.

## Status

**0.2.1** — **Development Status :: 5 - Production/Stable**. First stable line plus declared `PROVIDER_NAME` / dependency completeness.

- **Chat + streaming + tools + structured outputs** (JSON schema / `format`) + `think` pass-through.
- **Vision (image messages)** in the wire layer — OpenAI-style multimodal content lists with `image_url` parts whose URL is a `data:image/*;base64,...` data URL are converted to Ollama's `message.images` shape; HTTP(S) URLs are skipped with a warning (pre-encode them as data URLs to send images).
- **Observability** — every Ollama LLM call now lands in `AgentResult.llm_calls` with `provider="ollama"` populated automatically by the core agent loop.
- Floor `nucleusiq>=0.7.12`. **98 unit tests, coverage 99.85%** (gate ≥ 95%).
- Embeddings + web search remain planned follow-ups for `0.2.x` / `0.3.x`.
