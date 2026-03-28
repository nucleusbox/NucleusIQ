# nucleusiq-gemini

Google Gemini provider for the [NucleusIQ](https://github.com/nucleusbox/NucleusIQ) AI agent framework.

## Installation

```bash
pip install nucleusiq nucleusiq-gemini
```

Set your API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Or pass it directly:

```python
from nucleusiq_gemini import BaseGemini
llm = BaseGemini(api_key="your-key")
```

## Quick Start

### Direct LLM Call

```python
import asyncio
from nucleusiq_gemini import BaseGemini

async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")
    response = await llm.call(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello, Gemini!"}],
        max_output_tokens=256,
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### With NucleusIQ Agent

```python
import asyncio
from nucleusiq.agents.agent import Agent
from nucleusiq.agents.config import AgentConfig
from nucleusiq_gemini import BaseGemini, GeminiLLMParams

async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")
    config = AgentConfig(
        llm_params=GeminiLLMParams(temperature=0.5, max_output_tokens=1024),
    )
    agent = Agent(
        llm=llm, config=config,
        name="my-agent", model="gemini-2.5-flash",
        instructions="You are a helpful assistant.",
    )
    result = await agent.execute("Explain quantum computing simply.")
    print(result.content)

asyncio.run(main())
```

## Gearbox Strategy (3 Execution Modes)

NucleusIQ's Gearbox Strategy lets you pick the right power level for each task:

### Gear 1: DIRECT — Fast, No Tools

Single LLM call. Best for Q&A, classification, summarization.

```python
from nucleusiq.agents.config import AgentConfig, ExecutionMode

config = AgentConfig(execution_mode=ExecutionMode.DIRECT)
agent = Agent(llm=llm, config=config, model="gemini-2.5-flash", ...)
result = await agent.execute("What is the capital of France?")
```

### Gear 2: STANDARD — Tool-Enabled Loop (Default)

Iterative tool calling (up to 30 calls). Best for tasks needing external data.

```python
from nucleusiq.tools.decorators import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return "22°C, Sunny"

config = AgentConfig(execution_mode=ExecutionMode.STANDARD)
agent = Agent(llm=llm, config=config, tools=[get_weather], ...)
result = await agent.execute("What's the weather in Paris?")
```

### Gear 3: AUTONOMOUS — Orchestration + Verification

Task decomposition, Critic verification, Refiner loop. Best for complex multi-step tasks.

```python
config = AgentConfig(
    execution_mode=ExecutionMode.AUTONOMOUS,
    require_quality_check=True,
    max_iterations=5,
)
agent = Agent(llm=llm, config=config, tools=[...], ...)
result = await agent.execute("Compare Python and Rust for AI applications.")
```

## Native Server-Side Tools

Gemini provides built-in tools that execute on Google's servers:

```python
from nucleusiq_gemini import GeminiTool

# Google Search — ground responses with search results
search = GeminiTool.google_search()

# Code Execution — run Python in a secure sandbox
code = GeminiTool.code_execution()

# URL Context — fetch and understand web pages
url_ctx = GeminiTool.url_context()

# Google Maps — location-aware grounding
maps = GeminiTool.google_maps()

# Use with an agent
agent = Agent(llm=llm, config=config, tools=[search, code], ...)
```

Or use directly with the LLM:

```python
result = await llm.call(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "What are the latest AI news?"}],
    tools=[GeminiTool.google_search()],
    max_output_tokens=1024,
)
```

## Framework-Level Tools

Use any NucleusIQ built-in tool or create your own with the `@tool` decorator:

```python
from nucleusiq.tools import FileReadTool, FileSearchTool, DirectoryListTool
from nucleusiq.tools.decorators import tool

# Built-in file tools
file_reader = FileReadTool(workspace_root="/path/to/project")
file_search = FileSearchTool(workspace_root="/path/to/project")

# Custom tools via @tool decorator
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = Agent(
    llm=llm, config=config,
    tools=[file_reader, file_search, calculate],
    ...
)
```

## Plugins

NucleusIQ plugins work identically with Gemini:

```python
from nucleusiq.plugins.builtin import ContextWindowPlugin

plugin = ContextWindowPlugin(max_messages=50, max_tokens=8000)
agent = Agent(llm=llm, config=config, plugins=[plugin], ...)
```

## Memory Strategies

All memory strategies work with Gemini:

```python
from nucleusiq.memory import BufferMemory, SummaryMemory, SummaryWindowMemory

# Simple buffer (last N messages)
memory = BufferMemory(max_messages=20)

# Summary-based (compresses old messages)
memory = SummaryMemory(llm=llm, model="gemini-2.5-flash", max_messages=5)

# Hybrid (summary + recent window)
memory = SummaryWindowMemory(llm=llm, model="gemini-2.5-flash", window_size=10)

agent = Agent(llm=llm, config=config, memory=memory, ...)
```

## Structured Output

Pass a Pydantic model to get typed responses:

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

result = await llm.call(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Review the movie Inception"}],
    response_format=MovieReview,
    max_output_tokens=512,
)
# result is a MovieReview instance
print(result.title, result.rating)
```

## Streaming

```python
async for event in llm.call_stream(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Write a poem"}],
    max_output_tokens=512,
):
    if event.type.value == "token":
        print(event.delta, end="", flush=True)
    elif event.type.value == "complete":
        print(f"\n\nTotal tokens: {event.metadata.get('usage', {})}")
```

## Gemini-Specific Parameters

Use `GeminiLLMParams` for Gemini-specific settings:

```python
from nucleusiq_gemini import GeminiLLMParams, GeminiThinkingConfig, GeminiSafetySettings

params = GeminiLLMParams(
    temperature=0.7,
    max_output_tokens=2048,
    top_k=40,
    top_p=0.95,
    thinking_config=GeminiThinkingConfig(thinking_budget=2048),
    safety_settings=[
        GeminiSafetySettings(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        )
    ],
)
config = AgentConfig(llm_params=params)
```

## Cost Estimation

Track costs using the built-in `CostTracker`:

```python
from nucleusiq.agents.components.pricing import CostTracker

# After agent execution
usage = agent.last_usage
tracker = CostTracker()
cost = tracker.estimate(usage, model="gemini-2.5-flash")

print(f"Total cost: ${cost.total_cost:.6f}")
print(f"Prompt: ${cost.prompt_cost:.6f}")
print(f"Completion: ${cost.completion_cost:.6f}")
```

Built-in pricing tables include:

| Model | Prompt (per 1K) | Completion (per 1K) |
|-------|-----------------|---------------------|
| gemini-2.5-pro | $0.00125 | $0.01 |
| gemini-2.5-flash | $0.000075 | $0.0003 |
| gemini-2.0-flash | $0.0001 | $0.0004 |
| gemini-1.5-pro | $0.00125 | $0.005 |
| gemini-1.5-flash | $0.000075 | $0.0003 |

## Error Handling & Retry

The Gemini provider maps SDK errors to NucleusIQ's framework-level exception hierarchy, so you can catch provider-agnostic errors regardless of which LLM you use:

```python
from nucleusiq.llms.errors import RateLimitError, AuthenticationError, LLMError

try:
    result = await agent.execute("Hello")
except RateLimitError as e:
    print(f"Rate limited by {e.provider}, status {e.status_code}")
except AuthenticationError:
    print("Invalid API key")
except LLMError as e:
    print(f"LLM error from {e.provider}: {e}")
```

Built-in retry with exponential backoff:

- **Rate limits (429)** → `RateLimitError` after 3 retries with backoff
- **Server errors (5xx)** → `ProviderServerError` after 3 retries with backoff
- **Connection errors** → `ProviderConnectionError` after 3 retries with backoff
- **Auth errors (401)** → `AuthenticationError` immediately (no retry)
- **Permission denied (403)** → `PermissionDeniedError` immediately (no retry)
- **Bad requests (400)** → `InvalidRequestError` immediately
- **Model not found (404)** → `ModelNotFoundError` immediately

The `google-genai` SDK also has built-in tenacity retry for HTTP-level transient failures, providing a second layer of resilience.

## Provider Portability

Switch between providers with zero code changes to your agent logic:

```python
# Gemini
from nucleusiq_gemini import BaseGemini
llm = BaseGemini(model_name="gemini-2.5-flash")

# OpenAI (same agent code works)
from nucleusiq_openai import BaseOpenAI
llm = BaseOpenAI(model_name="gpt-4o")

# Same agent, different provider
agent = Agent(llm=llm, config=config, ...)
```

## Supported Models

| Model | Context | Max Output | Thinking |
|-------|---------|------------|----------|
| gemini-2.5-pro | 1M | 65,536 | Yes |
| gemini-2.5-flash | 1M | 65,536 | Yes |
| gemini-2.0-flash | 1M | 8,192 | No |
| gemini-1.5-pro | 2M | 8,192 | No |
| gemini-1.5-flash | 1M | 8,192 | No |

## Examples

| Example | Description |
|---------|-------------|
| `01_quickstart.py` | Basic LLM call |
| `02_tools.py` | Function calling with custom tools |
| `03_streaming.py` | Streaming responses |
| `04_provider_portability.py` | Switch between Gemini and OpenAI |
| `05_native_tools.py` | Google Search, Code Execution, URL Context, Maps |
| `06_cost_estimation.py` | Cost tracking with CostTracker |
| `agents/01_gemini_agent.py` | Basic agent |
| `agents/02_gearbox_direct.py` | Gear 1: Direct mode |
| `agents/03_gearbox_standard.py` | Gear 2: Standard mode with tools |
| `agents/04_gearbox_autonomous.py` | Gear 3: Autonomous mode |

## Architecture

The Gemini provider follows **SOLID** design principles:

```
nucleusiq_gemini/
├── __init__.py              # Public API exports
├── llm_params.py            # GeminiLLMParams (Pydantic)
├── _shared/
│   ├── model_config.py      # Model capabilities + metadata
│   ├── models.py            # GenerationConfig, ThinkingConfig (Pydantic)
│   ├── response_models.py   # GeminiLLMResponse, AssistantMessage (Pydantic)
│   └── retry.py             # Retry with exponential backoff
├── nb_gemini/
│   ├── base.py              # BaseGemini — orchestrator (BaseLLM contract)
│   ├── client.py            # GeminiClient — SDK communication
│   ├── response_normalizer.py  # Raw SDK → normalized response
│   └── stream_adapters.py   # Streaming chunks → StreamEvent
├── structured_output/
│   ├── builder.py           # Pydantic → Gemini JSON schema
│   └── parser.py            # JSON response → Pydantic instances
└── tools/
    ├── gemini_tool.py        # Native tool factory (Search, Code, URL, Maps)
    └── tool_converter.py     # BaseTool spec → Gemini function declarations
```

**Key principles:**
- **SRP**: Each file has one responsibility
- **OCP**: New tools added via factory methods, not modification
- **DIP**: `BaseGemini` depends on `BaseLLM` abstraction
- **No god classes**: `BaseGemini` delegates to specialized collaborators
- **Pydantic models**: Every request, response, and intermediate is typed

## v0.7.0+ Roadmap (Deferred)

These features are planned for future releases:

- **Batch API** — 50% cost reduction for async processing via `client.batches`
- **Deep Research Agent** — Via `client.interactions` API (preview)
- **File Search** — Server-side document search via `file_search_stores`
- **Computer Use** — Automated UI interaction (preview)
- **Live API** — Real-time voice/audio via WebSocket

## License

MIT — same as NucleusIQ core.
