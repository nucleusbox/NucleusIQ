# NucleusIQ + Ollama — runnable examples

Scripts call a real Ollama server (local **or** hosted). They load **`OLLAMA_API_KEY`**, **`OLLAMA_HOST`**, and optional **`OLLAMA_MODEL`** from the process environment or from a **`.env`** via optional **`python-dotenv`** (`load_dotenv()`), matching the OpenAI example pattern — set variables yourself or run from a directory where `python-dotenv` can discover the repo root `.env` (e.g. **`src/providers/inference/ollama`** or the **monorepo root**).

## Prerequisites

- **Ollama** running locally *or* a reachable **`OLLAMA_HOST`** (e.g. cloud).
- A pulled model, e.g. `ollama pull llama3.2` (names vary by setup).

From `src/providers/inference/ollama`:

```bash
uv sync --group dev
pip install python-dotenv   # optional; scripts still work if vars are exported
```

Optional **repo root** `.env`:

```env
# Required only for authenticated / hosted endpoints (SDK sends Bearer token).
OLLAMA_API_KEY=...
# Optional; default http://127.0.0.1:11434 when unset (official client default)
# OLLAMA_HOST=https://ollama.com
OLLAMA_MODEL=llama3.2
```

## Run

From `src/providers/inference/ollama` (with `nucleusiq` on `PYTHONPATH`, e.g. editable install):

```bash
uv run python examples/agents/00_ollama_smoke.py
uv run python examples/agents/01_ollama_direct.py
uv run python examples/agents/02_ollama_stream_live.py
uv run python examples/agents/03_ollama_capabilities_matrix.py
uv run python examples/agents/03_ollama_capabilities_matrix.py --only structured
```

Or after `pip install -e ../../../../nucleusiq -e .` from this directory:

```bash
python examples/agents/00_ollama_smoke.py
```

## Scripts

| File | What it does |
|------|----------------|
| `00_ollama_smoke.py` | Minimal `BaseOllama.call()` — fastest “can I reach the server?” check |
| `01_ollama_direct.py` | Full **`Agent`** in **DIRECT** mode |
| `02_ollama_stream_live.py` | **`BaseOllama.call_stream`** token printing |
| `03_ollama_capabilities_matrix.py` | **Chat, stream, structured output, thinking** × **DIRECT / STANDARD / AUTONOMOUS** (see `--only`) |
