# Installation

## End Users

Install the published packages from PyPI.

### With pip

```bash
# Core framework only
pip install nucleusiq

# Core + OpenAI provider (most common)
pip install nucleusiq nucleusiq-openai

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
print(nucleusiq.__version__)  # 0.1.0
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

### OpenAI Provider

```bash
# Required for nucleusiq-openai
export OPENAI_API_KEY=sk-...

# Or create a .env file in your project root
echo "OPENAI_API_KEY=sk-..." > .env
```

NucleusIQ automatically loads `.env` files from the project root.

---

## Package Architecture

NucleusIQ is a monorepo with independently installable packages:

```
nucleusiq                  # Core framework (agents, prompts, tools, memory, plugins)
  ├── nucleusiq-openai     # OpenAI provider (depends on nucleusiq)
  ├── nucleusiq-gemini     # Google Gemini (planned)
  ├── nucleusiq-ollama     # Ollama local LLMs (planned)
  ├── nucleusiq-groq       # Groq inference (planned)
  ├── nucleusiq-pinecone   # Pinecone vector DB (planned)
  └── nucleusiq-chroma     # ChromaDB vector DB (planned)
```

Each provider depends on `nucleusiq>=0.1.0` — install the core first, then add providers as needed.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'nucleusiq'`

Make sure the package is installed in your active virtual environment:

```bash
pip list | grep nucleusiq
# Should show: nucleusiq 0.1.0
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
