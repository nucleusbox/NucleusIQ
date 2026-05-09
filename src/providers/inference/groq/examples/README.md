# NucleusIQ + Groq — runnable examples

Real **`Agent`** runs (no mocks) against the Groq API for scripts **`01`–`05`**. **`07`** streams tokens via **`BaseGroq.call_stream`** only. Uses **`GROQ_API_KEY`** from your environment or from a **`.env`** file anywhere above `examples/` (typically the monorepo root).

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
uv run python examples/agents/07_groq_stream_live.py
```

## Scenarios

| Script | Gear | Notes |
|--------|------|--------|
| `01` | DIRECT | None — plain chat |
| `02` | DIRECT | Local `@tool` (single-hop tool path) |
| `03` | STANDARD | Local tools — multi-step tool loop |
| `04` | AUTONOMOUS | Local tools + critic/refiner path |
| `05` | DIRECT | Structured output (Pydantic / provider JSON schema) |
| `06` | — | **Hosted built-in tools** — prints `GROQ_*_HOSTED_TOOL_IDS` vs local tools (no API call); Phase B wiring |
| `07` | — | **Streaming** — `BaseGroq.call_stream` live smoke |

**Built-in tools** (Groq-hosted web search, compound models, MCP, etc.) are **not** wired in Phase A / public beta; use **local function calling**. See `docs/design/GROQ_PROVIDER.md`. Scripts `01` through `05` and `07` exercise real API calls.
