# Installation

## End Users

Install the published packages from PyPI.

### With pip

```bash
# Core framework only
pip install nucleusiq

# Core + OpenAI cloud provider (most common)
pip install nucleusiq nucleusiq-openai

# Self-hosted / any OpenAI-compatible server (vLLM, SGLang, TGI, llama.cpp, …)
pip install nucleusiq nucleusiq-openai-compatible

# With optional clustering support
pip install "nucleusiq[clustering]"
```

### With uv

```bash
# Core + OpenAI provider
uv pip install nucleusiq nucleusiq-openai

# Or add to your project
uv add nucleusiq nucleusiq-openai
```

### Verify installation

```python
import nucleusiq
print(nucleusiq.__version__)  # 0.7.13
```

---

## Developers (Contributing)

Clone the monorepo and install in editable mode with dev dependencies.

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### With uv (recommended)

```bash
git clone https://github.com/nucleusbox/NucleusIQ.git
cd NucleusIQ

# Core package — install + dev deps
cd src/nucleusiq
uv venv && uv sync --all-groups
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Run core tests
uv run pytest tests/ -q

# OpenAI provider — install + dev deps (separate venv)
cd ../providers/llms/openai
uv venv && uv sync --all-groups
uv run pytest tests/ -q
```

### With pip

```bash
git clone https://github.com/nucleusbox/NucleusIQ.git
cd NucleusIQ

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Core package (editable)
pip install -e "src/nucleusiq[clustering]"
pip install pytest pytest-asyncio pytest-cov pytest-mock scikit-learn

# OpenAI provider (editable, links to local core)
pip install -e src/providers/llms/openai

# Run all tests
cd src/nucleusiq && python -m pytest tests/ -q
cd ../providers/llms/openai && python -m pytest tests/ -q
```

---

## Environment Variables

### OpenAI cloud

```bash
export OPENAI_API_KEY=sk-...
```

### OpenAI-compatible / self-hosted

```bash
export OPENAI_COMPATIBLE_BASE_URL=http://gpu-node-1:8000/v1
export OPENAI_COMPATIBLE_MODEL=gemma-4-27b-it
export OPENAI_COMPATIBLE_API_KEY=token-abc123   # omit if the server has no key
```

Or create a `.env` file in your project root. NucleusIQ automatically loads `.env` files from the project root.

---

## Package Architecture

NucleusIQ is a monorepo with independently installable packages:

```
nucleusiq                         # Core framework (agents, prompts, tools, memory, plugins)
  ├── nucleusiq-openai            # OpenAI cloud (Responses API + Chat Completions)
  ├── nucleusiq-gemini            # Google Gemini
  ├── nucleusiq-anthropic         # Anthropic Claude
  ├── nucleusiq-groq              # Groq (official groq SDK)
  ├── nucleusiq-ollama            # Ollama (official ollama SDK)
  ├── nucleusiq-openai-compatible # Any Chat Completions server (vLLM, SGLang, TGI, …)
  └── nucleusiq-mcp               # Model Context Protocol adapter
```

Most providers floor on `nucleusiq>=0.7.12`. `nucleusiq-openai-compatible` requires `nucleusiq>=0.7.13`. Install the core first, then add providers as needed.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'nucleusiq'`

Make sure the package is installed in your active virtual environment:

```bash
pip list | grep nucleusiq
# Should show: nucleusiq 0.7.13
```

### `ModuleNotFoundError: No module named 'nucleusiq_openai'`

Install the OpenAI provider separately:

```bash
pip install nucleusiq-openai
```

### Tests fail with import errors

For development, make sure both packages are installed in editable mode:

```bash
pip install -e src/nucleusiq -e src/providers/llms/openai
```
