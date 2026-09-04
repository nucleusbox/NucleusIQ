# nucleusiq-openai-compatible

Run NucleusIQ agents against **any server that speaks the OpenAI Chat Completions protocol** — your own vLLM or SGLang deployment, llama.cpp, LM Studio, TGI, NVIDIA NIM, or an OpenAI-compatible cloud (OpenRouter, Together, Fireworks, DeepInfra, Databricks, LiteLLM, Azure OpenAI).

**Bring your own model. Bring your own key.** Nothing is inferred from a model name, and an unauthenticated local server is a first-class case, not a workaround.

```bash
pip install nucleusiq-openai-compatible
pip install 'nucleusiq-openai-compatible[tokenizer]'   # optional: exact token counts
```

---

## Quick start

```python
from nucleusiq_openai_compatible import OpenAICompatibleLLM

llm = OpenAICompatibleLLM(
    base_url="http://gpu-node-1:8000/v1",
    model="gemma-4-27b-it",       # your --served-model-name
    api_key="token-abc123",       # omit entirely if the server has no --api-key
    context_window=32_768,        # recommended: must match --max-model-len
    engine="vllm",
)
```

That is the whole configuration. Everything else has a safe default.

> **Pass `context_window` if you know it.** It must match the server's `--max-model-len`. Without it the provider reads `/v1/models` for `max_model_len`, and if that fails it falls back to a deliberately conservative **8192** with a warning. Over-reporting a window is the dangerous direction — the context engine would skip compaction and your server would reject the request.

### Dropping it into an agent

The LLM is a plain dependency; nothing else about your agent changes.

```python
import asyncio
from nucleusiq.agents import Agent
from nucleusiq.tools import CalculatorTool

agent = Agent(
    name="analyst",
    role="Data analyst",
    llm=llm,
    tools=[CalculatorTool()],
)

result = asyncio.run(agent.execute("What is 17% of 4,830?"))
print(result.output)
```

Tools, execution modes, memory, plugins, structured output and the context engine all behave exactly as they do on OpenAI or Anthropic. The context engine sizes its budget from `llm.get_context_window()`, so a 32K self-hosted model gets correct region-aware budgeting and progressive compaction rather than assumptions borrowed from a 128K cloud model.

### Check your deployment before running an agent

```python
report = await llm.validate()
print(report.render())
```

```
OpenAI-compatible endpoint validation: FAILED
  reachable          : True
  model found        : False
  context window     : 32768 (source: probe)
  served models      : gemma-4-27b-it, qwen3-32b
  error             : Model 'gemma-4' is not served by http://gpu-node-1:8000/v1.
                      This server serves: gemma-4-27b-it, qwen3-32b.
```

`validate()` is explicit and never called implicitly — a constructor should not perform I/O. Its value is that a wrong model name becomes a list of what the server *does* serve, instead of a bare `404` surfacing halfway through an agent run.

---

## One instance, one model

**An `OpenAICompatibleLLM` describes exactly one model on exactly one endpoint.** This is enforced: passing a different `model=` to `call()` raises.

The reason is structural. NucleusIQ reads `BaseLLM.get_context_window()` **once**, with no model argument, to size the entire context budget for a run. A per-call model switch would leave that budget sized for the wrong window — a silent context-overflow bug in the exact subsystem this framework exists to get right.

Serving several models from one node? Build several instances. They are cheap and share nothing but a URL.

```python
models = {
    "gemma":  OpenAICompatibleLLM(base_url=URL, model="gemma-4-27b-it", context_window=32_768, engine="vllm"),
    "qwen":   OpenAICompatibleLLM(base_url=URL, model="qwen3-32b",      context_window=40_960, engine="vllm"),
}
agent = Agent(name="a", role="r", llm=models["gemma"])
```

---

## Bring your own key

`api_key` covers almost every case, and accepts a **string or a callable**:

```python
api_key="token-abc123"                        # static
api_key=None                                  # unauthenticated local server (default)
api_key=lambda: vault.get("tenant-a-key")     # resolved per request
api_key=fetch_token_async                     # async callable also accepted
```

Callables are resolved **per request**, so key rotation and per-tenant credentials work with no agent rebuild. One instance can safely serve many tenants: credentials are applied through the SDK's `with_options()`, which returns a shallow copy sharing the connection pool and never mutates shared state.

For a server that wants the credential somewhere other than `Authorization: Bearer`, use a strategy:

| Strategy | Sends | Use for |
|---|---|---|
| `NoAuth()` *(default)* | nothing | local vLLM, SGLang, llama.cpp, LM Studio, Ollama |
| `BearerAuth(v)` | `Authorization: Bearer v` | vLLM `--api-key`, and every hosted OpenAI-compatible cloud |
| `HeaderAuth(name, v)` | `name: v` | Azure (`api-key`), gateways (`X-API-Key`) |

```python
from nucleusiq_openai_compatible import HeaderAuth

llm = OpenAICompatibleLLM(
    base_url="https://my-resource.openai.azure.com/openai/v1",
    model="my-deployment",         # deployment name, not the base model
    engine="azure",
    auth=HeaderAuth("api-key", os.environ["AZURE_OPENAI_API_KEY"]),
)
```

`api_key="..."` is exactly sugar for `BearerAuth("...")`; passing both `api_key` and `auth` is rejected as ambiguous. Anything more exotic — mTLS, SigV4, a corporate proxy — goes through `http_client=` (bring your own configured `httpx.AsyncClient`) rather than a new strategy class.

**Secrets never leak.** Credentials are absent from `repr()`, logs, telemetry and mapped exception messages. Error text from the server is scrubbed before it is wrapped, because gateways routinely echo request headers into error bodies.

```python
>>> repr(llm)
"OpenAICompatibleLLM(base_url='http://gpu-node-1:8000/v1', model='gemma-4-27b-it', engine='vllm', context_window=32768, auth=<redacted>)"
```

---

## Thinking / reasoning models

Most current open-weight models can think before answering, and getting this wrong is quiet rather than loud. Two things to set:

```python
llm = OpenAICompatibleLLM(
    base_url=URL,
    model="qwen3-32b",
    engine="vllm",
    context_window=40_960,
    is_reasoning_model=True,                            # 1. budget
    chat_template_kwargs={"enable_thinking": True},     # 2. switch
)
```

**1. `is_reasoning_model=True` is load-bearing, not cosmetic.** Thinking and visible output share one completion budget, and the framework widens the budget for internal calls (Critic, Refiner, Decomposer) when this is set. Left `False` on a thinking model, those calls get truncated mid-thought and come back unusable. It is never inferred from the model name — a served name is arbitrary on a self-hosted deployment. If a response arrives with separated thinking while this is `False`, you get a one-time warning telling you to set it.

**2. The thinking switch differs per model family**, and the default differs too:

| Family | Default | Toggle |
|---|---|---|
| Qwen3 series | **on** | `{"enable_thinking": False}` to disable |
| **Gemma 4** | **off** | `{"enable_thinking": True}` to enable |
| IBM Granite 3.2 | off | `{"thinking": True}` |
| DeepSeek-V3.1 | off | `{"thinking": True}` |

`chat_template_kwargs` is set once on the constructor and forwarded on every request, because the chat template is applied server-side per request. You can also set it server-wide with vLLM's `--default-chat-template-kwargs`.

Your server needs `--reasoning-parser` for thinking to arrive in its own field:

```bash
vllm serve google/gemma-4-27b-it \
  --served-model-name gemma-4-27b-it \
  --max-model-len 32768 \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Without a reasoning parser, thinking text arrives inline in `content` — and `validate()` warns you about that combination.

Reading it back:

```python
response = await llm.call(messages=[...])
response.content            # the answer
response.reasoning          # the thinking, separated
response.reasoning_tokens   # thinking tokens, when the server reports them
```

The provider reads **both** `reasoning` and the older `reasoning_content` field name, since vLLM renamed it and both are still in the wild depending on server version.

On engines that support it (vLLM, SGLang, OpenRouter, LiteLLM, Azure), `reasoning_effort` is forwarded rather than stripped — vLLM maps it onto the template's thinking switch. On engines that don't, it is dropped silently rather than triggering a `400`.

### When thinking arrives but the answer doesn't

There is a known vLLM failure mode where the chat template and the `--reasoning-parser` disagree, and the **entire answer comes back as reasoning with `content: null`** ([vLLM #53284](https://github.com/vllm-project/vllm/issues/53284)). The provider detects this and tells you what to do instead of reporting an empty completion:

```python
if response.reasoning_only:
    ...   # a warning is already logged with the remediation
```

The fix is nearly always to pass the thinking toggle explicitly rather than relying on the template default.

---

## Tool calling

Tools work unchanged — pass them to `Agent(tools=[...])`. Every tool is a **local function tool**: a self-hosted inference server has no server-side tool runtime, so `NATIVE_TOOL_TYPES` is empty and every `ToolCallRecord` is `executed_by="local"`.

Your vLLM or SGLang server must be started for it:

```bash
--enable-auto-tool-choice --tool-call-parser hermes
```

Parser choice depends on the model family — `hermes`, `llama3_json`, `mistral`, `granite`, `qwen3_coder`, `deepseek_v3`, `glm45`. If your server was started without those flags, pass `supports_tools=False` and the provider will warn and drop tools rather than let the server reject the request.

Fragmented tool-call deltas in streams are merged automatically: servers send `id` and `name` in the first fragment and `arguments` across many, and the accumulator reassembles whole calls.

Note that **not every thinking model supports tools** — DeepSeek-R1 does not, while QwQ-32B and Qwen3 do. Check your model's parser row in the vLLM reasoning docs.

### Structured output plus tools — the one real trap

vLLM applies constrained decoding for `response_format`. Send **both** `tools` and `response_format` with `tool_choice="auto"` and the grammar forces JSON, so the model **never emits tool calls** — you get `tool_calls: []` and a JSON body ([vLLM #39929](https://github.com/vllm-project/vllm/issues/39929)). OpenAI's cloud does not behave this way. For an agent framework this is severe: enabling structured output would silently disable the tool loop.

The provider will not let that happen. `structured_output_with_tools` controls how it is resolved:

| Mode | Behavior |
|---|---|
| `"prompt"` *(default)* | Omit `response_format`, inject the schema into the system message, validate the reply. **Tools keep working and you still get structured output.** |
| `"drop"` | Omit `response_format` and warn. |
| `"error"` | Raise before the HTTP call. |

With no tools present, native `response_format={"type": "json_schema"}` is used normally — or `json_object` plus a prompt-injected schema on servers without schema support.

---

## Engine presets

`engine=` sets capability defaults. Every field stays overridable, because what a server can actually do depends on the flags it was started with. A typo lists the valid names.

| Preset | Auth | Tools | JSON schema | Thinking |
|---|---|---|---|---|
| `vllm` | none / Bearer | needs `--tool-call-parser` | yes | needs `--reasoning-parser` |
| `sglang` | none / Bearer | needs `--tool-call-parser` | yes | yes |
| `tgi` | Bearer | yes | partial | no |
| `llamacpp` | none / Bearer | build-dependent | `json_object` | build-dependent |
| `lmstudio` | none | yes | yes | yes |
| `ollama` | none / Bearer | yes | yes | yes |
| `nim` | Bearer | yes | yes | yes |
| `openrouter` | Bearer | yes | model-dependent | yes |
| `together`, `fireworks`, `deepinfra` | Bearer | yes | model-dependent | varies |
| `databricks`, `litellm` | Bearer | yes | yes | varies |
| `azure` | `api-key` or Bearer | yes | yes | yes |
| `generic` | none | conservative | off | no |

Overriding when your server differs from the preset:

```python
llm = OpenAICompatibleLLM(
    base_url=URL, model="my-model", engine="vllm", context_window=8_192,
    supports_tools=False,             # started without --enable-auto-tool-choice
    supports_json_schema=False,
    strict_capabilities=True,         # raise instead of warn on unsupported combos
)
```

---

## Sampling and engine-specific knobs

```python
from nucleusiq_openai_compatible import OpenAICompatibleLLMParams

llm = OpenAICompatibleLLM(
    base_url=URL, model="gemma-4-27b-it", engine="vllm", context_window=32_768,
    llm_params=OpenAICompatibleLLMParams(
        temperature=0.2,
        seed=42,
        parallel_tool_calls=True,
        extra_body={                         # engine-specific escape hatch
            "top_k": 40,
            "repetition_penalty": 1.05,
        },
    ),
)
```

`extra_body` is the escape hatch for anything outside the OpenAI schema — `top_k`, `min_p`, `repetition_penalty`, `guided_json`, `guided_regex`, `guided_choice`, `chat_template_kwargs`. It is forwarded only on engines that tolerate a non-standard body.

OpenAI-cloud-only parameters (`service_tier`, `store`, `prompt_cache_key`, `logit_bias`, `logprobs`, `modalities`, `audio`, …) are **stripped before the request**, logged at debug. Strict servers answer `400 unknown parameter` rather than ignoring them, and these routinely arrive from a shared `AgentConfig` written for OpenAI.

---

## Token counting

By default tokens are estimated at ~4 characters per token, matching the framework default. That is approximate, and worst on code and non-English text — exactly where a context budget matters. For exact counts, declare the model's own tokenizer:

```python
llm = OpenAICompatibleLLM(
    base_url=URL, model="gemma-4-27b-it", context_window=32_768,
    tokenizer="google/gemma-4-27b-it",     # needs the [tokenizer] extra
)
```

`tiktoken` is deliberately **not** used: it is an OpenAI tokenizer family and would produce systematically wrong counts for Llama, Qwen, Gemma or Mistral. Which method is in use is recorded on every call record, so telemetry can tell measured counts from estimated ones.

---

## Errors

Provider errors map onto the standard `nucleusiq.llms.errors` hierarchy, so error handling is identical across providers. Self-hosted specifics:

- **Connection errors are expected** — GPU nodes restart and models take minutes to load. These retry with exponential backoff, and a "model is still loading" 503 is treated as retryable.
- **Context overflow is recognized from vLLM and TGI phrasing**, not just OpenAI's, so it surfaces as `ContextLengthError` and the context engine can compact rather than fail the run. If you see this unexpectedly, your `context_window` is probably larger than the server's `--max-model-len`.
- **`401` is never retried** — it is a caller problem, and the message names the auth strategy without ever printing the value.

```python
from nucleusiq.llms.errors import ContextLengthError, ProviderConnectionError

try:
    result = await agent.execute(task)
except ProviderConnectionError:
    ...   # node down or base_url wrong
except ContextLengthError:
    ...   # window larger than the server's real limit
```

---

## Introspection

```python
caps = llm.capabilities            # frozen, read-only
caps.context_window                # 32768
caps.context_window_source         # "explicit" | "probe" | "engine" | "default"
caps.supports_tools                # True
caps.is_reasoning_model            # True
caps.token_count_method            # "tokenizer" | "heuristic"
```

`context_window_source` answers the most common self-hosted question: *where did that number come from?*

---

## Which provider should I use?

| Situation | Package |
|---|---|
| OpenAI cloud (Responses API, hosted tools, `reasoning_effort`) | `nucleusiq-openai` |
| Anthropic, Gemini, Groq | `nucleusiq-anthropic`, `nucleusiq-gemini`, `nucleusiq-groq` |
| Local Ollama, native API | `nucleusiq-ollama` |
| **Your own vLLM / SGLang / TGI / llama.cpp / LM Studio** | **this package** |
| **An OpenAI-compatible cloud with no first-party NucleusIQ provider** | **this package** |

This package deliberately does not depend on `nucleusiq-openai`: it carries no Responses API code and no `tiktoken`, and the two evolve independently.

---

## Not yet supported

Planned for `0.2.x`: `/v1/embeddings`, vision content parts, and engine auto-detection from `/v1/models`. Guided decoding as a first-class structured-output mode (`guided_json`) is under consideration.

## License

MIT — see the repository root.
