# Examples

Runnable scripts for `nucleusiq-openai-compatible`. They are **not** packaged
into the wheel or the sdist — clone the repo to run them.

## Setup

Point the examples at your server:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://gpu-node-1:8000/v1"
export OPENAI_COMPATIBLE_MODEL="gemma-4-27b-it"
export OPENAI_COMPATIBLE_API_KEY="..."   # only if your server requires one
```

A local vLLM server for these examples:

```bash
vllm serve google/gemma-4-27b-it \
  --served-model-name gemma-4-27b-it \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1
```

## No GPU? Use any OpenAI-compatible endpoint

This provider is not vLLM-specific — anything speaking `/v1/chat/completions`
works, so you do not need to host a model to exercise it end to end.

| Endpoint | `base_url` | `engine` | Notes |
|---|---|---|---|
| Ollama Cloud | `https://ollama.com/v1` | `ollama` | Hosted open-weight models, Bearer key, no local GPU. Pass `supports_json_schema=False` — the shim accepts `response_format` and ignores it. |
| `ollama serve` | `http://localhost:11434/v1` | `ollama` | CPU-friendly with a small model (`ollama pull qwen2.5:0.5b`). |
| LM Studio | `http://localhost:1234/v1` | `lmstudio` | Desktop app, runs on CPU. |
| `llama-server` | `http://localhost:8080/v1` | `llamacpp` | GGUF on CPU. |
| Groq / Together / Fireworks | provider URL | matching preset | Hosted, key required. |

Verify a deployment before trusting it:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://ollama.com/v1"
export OPENAI_COMPATIBLE_API_KEY="$OLLAMA_API_KEY"
export OPENAI_COMPATIBLE_MODEL="gemma4:31b"
export OPENAI_COMPATIBLE_ENGINE="ollama"

pytest tests/integration/test_live_endpoint.py -m integration --no-cov -v
```

That suite checks reachability, the resolved context window, completions,
streaming, tool calls, the tool round trip, structured output, agent
integration and credential handling against the real server. It is skipped
unless `OPENAI_COMPATIBLE_BASE_URL` is set, and `--no-cov` is needed because
the package-wide 95% gate would otherwise fail a run that only exercises the
request path.

Add `OPENAI_COMPATIBLE_REASONING_MODEL=gpt-oss:120b` to also check that a
thinking model's reasoning arrives separated from its answer.

Install the provider in editable mode:

```bash
pip install -e src/nucleusiq -e src/providers/inference/openai_compatible
```

## The examples

Run them in this order; each builds on the previous one.

| # | File | What it shows |
|---|------|---------------|
| 1 | [`01_validate_endpoint.py`](01_validate_endpoint.py) | Preflight a server before wiring an agent to it. Start here when something is misconfigured. |
| 2 | [`02_basic_completion.py`](02_basic_completion.py) | Direct provider use: a call, token usage, and streaming. No agent. |
| 3 | [`agents/03_agent_with_prompt.py`](agents/03_agent_with_prompt.py) | A `nucleusiq` `Agent` with a zero-shot prompt, and how the prompt becomes messages. |
| 4 | [`agents/04_agent_with_tools.py`](agents/04_agent_with_tools.py) | The tool loop: `@tool` functions, the call/execute/reply round trip. |
| 5 | [`agents/05_structured_output.py`](agents/05_structured_output.py) | Pydantic structured output, including alongside tools (the vLLM trap). |
| 6 | [`agents/06_thinking_model.py`](agents/06_thinking_model.py) | Reasoning models: `is_reasoning_model`, `chat_template_kwargs`, separated thinking. |
| 7 | [`07_bring_your_own_key.py`](07_bring_your_own_key.py) | BYOK: static keys, rotating callables, Azure's `api-key` header, per-tenant keys. |

Every example runs against a live server. `01` is the only one that is safe
to run against a server you are unsure about — it never sends a completion.
