# NucleusIQ + Groq — runnable examples

Real **`Agent`** runs (no mocks) against the Groq API. Uses **`GROQ_API_KEY`** from your environment or from a **`.env`** file anywhere above `examples/` (typically the monorepo root).

Each script builds a **`Task(id=..., objective=...)`** and calls **`await agent.initialize()`** before **`execute()`**, matching the current NucleusIQ `Agent` API.

## Prerequisites

```bash
cd src/providers/inference/groq
uv sync --group dev
```

Optional: in the repo root `.env`:

```env
GROQ_API_KEY=gsk_...
# Optional; default is llama-3.3-70b-versatile
GROQ_MODEL=llama-3.3-70b-versatile
```

## Run

From `src/providers/inference/groq`:

```bash
uv run python examples/agents/01_groq_direct.py
uv run python examples/agents/02_groq_direct_with_tool.py
uv run python examples/agents/03_groq_standard_tools.py
uv run python examples/agents/04_groq_autonomous.py
uv run python examples/agents/05_groq_structured_output.py
uv run python examples/agents/06_groq_builtin_tools_status.py
```

## Scenarios

| Script | Gear | Tools |
|--------|------|--------|
| `01` | DIRECT | None — plain chat |
| `02` | DIRECT | Local `@tool` (single-hop tool path) |
| `03` | STANDARD | Local tools — multi-step tool loop |
| `04` | AUTONOMOUS | Local tools + critic/refiner path |
| `05` | DIRECT | Structured output (Pydantic / provider JSON schema) |
| `06` | — | **Built-in / hosted Groq tools** — status only (Phase B; see design doc) |

**Built-in tools** (Groq-hosted web search, agentic compound models, etc.) are **not** wired in `nucleusiq-groq` Phase A; the framework uses **local function calling** through Chat Completions. See `docs/design/GROQ_PROVIDER.md`. Scripts `01` through `05` cover real end-to-end runs.
