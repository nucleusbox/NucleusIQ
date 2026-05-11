# nucleusiq-ollama

Alpha provider for running **NucleusIQ** agents against [Ollama](https://docs.ollama.com/) (local or hosted).

## Install

**Alpha** on PyPI (**`0.1.0a1`**). Requires **`nucleusiq>=0.7.10`** (pulled in automatically by `pip`).

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

See [OLLAMA_PROVIDER.md](https://github.com/nucleusbox/NucleusIQ/blob/main/docs/design/OLLAMA_PROVIDER.md) for capability matrix, environment variables, and roadmap.

## Status

**0.1.0a1** — chat, streaming, tools, structured outputs (JSON schema / `format`), and `think` pass-through. Vision, embeddings, and web search are planned follow-ups.
