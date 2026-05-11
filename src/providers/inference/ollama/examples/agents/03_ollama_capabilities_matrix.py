"""Exercise Ollama capabilities across all Agent execution modes.

Covers:

- **Chat** — plain completion (no ``response_format``)
- **Streaming** — ``Agent.execute_stream`` (token / thinking events)
- **Structured output** — ``response_format`` as a Pydantic model (native JSON schema)
- **Thinking** — ``OllamaLLMParams(think=True)`` (model must support Ollama ``think``)

Runs each capability under **DIRECT**, **STANDARD**, and **AUTONOMOUS**.

Prerequisites
-------------

From ``src/providers/inference/ollama``::

    uv sync --group dev

Repo-root ``.env`` (local or cloud), e.g.::

    OLLAMA_API_KEY=...           # for cloud
    OLLAMA_HOST=https://ollama.com
    OLLAMA_MODEL=gpt-oss:120b    # or any model supporting format/thinking on your host

Usage::

    uv run python examples/agents/03_ollama_capabilities_matrix.py
    uv run python examples/agents/03_ollama_capabilities_matrix.py --only chat
    uv run python examples/agents/03_ollama_capabilities_matrix.py --only stream
    uv run python examples/agents/03_ollama_capabilities_matrix.py --only structured
    uv run python examples/agents/03_ollama_capabilities_matrix.py --only thinking

Notes
-----

- **AUTONOMOUS** runs Decomposer + (for some tasks) Critic/Refiner — more LLM calls and latency.
- **AUTONOMOUS + structured** may finish with plain text after verification; PRIMARY output is still shown.
- **Thinking** in stream mode emits ``StreamEventType.THINKING`` when the model returns reasoning deltas.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from typing import Any

# Load environment variables (optional) — same pattern as OpenAI examples
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nucleusiq.agents import Agent  # noqa: E402
from nucleusiq.agents.config import AgentConfig, ExecutionMode  # noqa: E402
from nucleusiq.agents.task import Task  # noqa: E402
from nucleusiq.prompts.zero_shot import ZeroShotPrompt  # noqa: E402
from nucleusiq.streaming.events import StreamEventType  # noqa: E402
from nucleusiq_ollama import BaseOllama, OllamaLLMParams  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

logging.basicConfig(level=logging.WARNING)


class CapitalFact(BaseModel):
    """Example schema for structured-output demos."""

    country: str = Field(description="Country name in English")
    capital: str = Field(description="Capital city name in English")


def _model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2")


def _print_banner(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def _describe_output(raw: Any, *, max_chars: int = 1200) -> str:
    if raw is None:
        return "(no output)"
    if isinstance(raw, BaseModel):
        return raw.model_dump_json(indent=2)
    if isinstance(raw, dict) and "output" in raw and raw.get("mode") == "native":
        inner = raw["output"]
        if isinstance(inner, BaseModel):
            return inner.model_dump_json(indent=2)
        return str(inner)[:max_chars]
    text = str(raw).strip()
    if len(text) > max_chars:
        return f"{text[:max_chars]}… ({len(text)} chars total)"
    return text


def _build_agent(
    *,
    mode: ExecutionMode,
    name_suffix: str,
    system: str,
    llm_params: OllamaLLMParams | None,
    response_format: type[BaseModel] | None,
) -> Agent:
    llm = BaseOllama(model_name=_model(), async_mode=True)
    return Agent(
        name=f"ollama-{name_suffix}-{mode.value}",
        prompt=ZeroShotPrompt().configure(system=system),
        llm=llm,
        tools=[],
        response_format=response_format,
        config=AgentConfig(
            execution_mode=mode,
            llm_params=llm_params,
            llm_max_output_tokens=512,
            step_inference_max_tokens=384,
            max_iterations=8,
        ),
    )


async def _run_chat_for_mode(mode: ExecutionMode) -> None:
    system = (
        "You are a concise assistant. Answer clearly in one or two short sentences."
    )
    params = OllamaLLMParams(temperature=0.25, max_output_tokens=256)
    agent = _build_agent(
        mode=mode,
        name_suffix="chat",
        system=system,
        llm_params=params,
        response_format=None,
    )
    await agent.initialize()
    tid = uuid.uuid4().hex[:10]
    result = await agent.execute(
        Task(
            id=f"ollama-chat-{mode.value}-{tid}",
            objective=(
                "What is the chemical symbol for gold? Reply in one short sentence."
            ),
        )
    )
    print(f"[{mode.value}] status={result.status.value}", flush=True)
    print(_describe_output(result.output), flush=True)


async def _run_stream_for_mode(mode: ExecutionMode, *, think: bool) -> None:
    label = "stream+think" if think else "stream"
    system = (
        "You are a concise assistant. Keep answers short."
        if not think
        else "Think step by step briefly, then give a very short final answer."
    )
    params = OllamaLLMParams(
        temperature=0.3,
        max_output_tokens=320,
        think=True if think else None,
    )
    agent = _build_agent(
        mode=mode,
        name_suffix=label,
        system=system,
        llm_params=params,
        response_format=None,
    )
    await agent.initialize()
    tid = uuid.uuid4().hex[:10]
    task = Task(
        id=f"ollama-{label}-{mode.value}-{tid}",
        objective=(
            "Name exactly one gas giant in our solar system. "
            "Final answer: one proper noun only."
            if think
            else "Say hello in ten words or fewer."
        ),
    )
    print(f"[{mode.value}] streaming…", flush=True)
    token_buf: list[str] = []
    think_buf: list[str] = []
    async for ev in agent.execute_stream(task):
        if ev.type == StreamEventType.TOKEN and ev.token:
            token_buf.append(ev.token)
            print(ev.token, end="", flush=True)
        elif ev.type == StreamEventType.THINKING and ev.message:
            think_buf.append(ev.message)
            print(f"\n[thinking] {ev.message}", end="", flush=True)
        elif ev.type == StreamEventType.ERROR:
            print(f"\n[stream error] {ev.message}", flush=True)
            return
        elif ev.type == StreamEventType.COMPLETE:
            print(flush=True)
    joined = "".join(token_buf).strip()
    print(
        f"\n--- tokens(chars)={len(joined)} thinking_chunks={len(think_buf)}",
        flush=True,
    )


async def _run_structured_for_mode(mode: ExecutionMode) -> None:
    system = "Reply only as structured data matching the schema; use factual geography."
    params = OllamaLLMParams(temperature=0.1, max_output_tokens=256)
    agent = _build_agent(
        mode=mode,
        name_suffix="structured",
        system=system,
        llm_params=params,
        response_format=CapitalFact,
    )
    await agent.initialize()
    tid = uuid.uuid4().hex[:10]
    result = await agent.execute(
        Task(
            id=f"ollama-struct-{mode.value}-{tid}",
            objective="Fill in country=Japan and its capital city.",
        )
    )
    print(f"[{mode.value}] status={result.status.value}", flush=True)
    print(_describe_output(result.output), flush=True)


async def _run_thinking_nonstream_for_mode(mode: ExecutionMode) -> None:
    """Single-shot call with ``think=True`` (reasoning may appear in message metadata)."""
    system = "Reason briefly, then answer in at most one sentence."
    params = OllamaLLMParams(think=True, temperature=0.2, max_output_tokens=384)
    agent = _build_agent(
        mode=mode,
        name_suffix="thinking",
        system=system,
        llm_params=params,
        response_format=None,
    )
    await agent.initialize()
    tid = uuid.uuid4().hex[:10]
    result = await agent.execute(
        Task(
            id=f"ollama-think-{mode.value}-{tid}",
            objective=(
                "Is 17 a prime number? Answer Yes or No first, then one short reason."
            ),
        )
    )
    print(f"[{mode.value}] status={result.status.value}", flush=True)
    print(_describe_output(result.output), flush=True)


async def _section_chat() -> None:
    _print_banner("Capability: CHAT (non-streaming)")
    for mode in ExecutionMode:
        print(f"\n--- Mode: {mode.value.upper()} ---", flush=True)
        await _run_chat_for_mode(mode)


async def _section_stream() -> None:
    _print_banner("Capability: STREAMING (execute_stream)")
    for mode in ExecutionMode:
        print(f"\n--- Mode: {mode.value.upper()} ---", flush=True)
        await _run_stream_for_mode(mode, think=False)


async def _section_structured() -> None:
    _print_banner("Capability: STRUCTURED OUTPUT (Pydantic response_format)")
    for mode in ExecutionMode:
        print(f"\n--- Mode: {mode.value.upper()} ---", flush=True)
        await _run_structured_for_mode(mode)


async def _section_thinking() -> None:
    _print_banner("Capability: THINKING (Ollama think=True)")
    print(
        "Non-stream first (execute), then stream (thinking deltas if supported).\n",
        flush=True,
    )
    for mode in ExecutionMode:
        print(f"\n--- Mode: {mode.value.upper()} / execute() ---", flush=True)
        await _run_thinking_nonstream_for_mode(mode)
        print(f"\n--- Mode: {mode.value.upper()} / execute_stream() ---", flush=True)
        await _run_stream_for_mode(mode, think=True)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ollama capability demos across Agent execution modes.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "chat", "stream", "structured", "thinking"),
        default="all",
        help="Run a single capability section (default: all).",
    )
    args = parser.parse_args()

    print(
        f"model={_model()!r} host={os.getenv('OLLAMA_HOST') or '(SDK default)'}",
        flush=True,
    )

    if args.only in ("all", "chat"):
        await _section_chat()
    if args.only in ("all", "stream"):
        await _section_stream()
    if args.only in ("all", "structured"):
        await _section_structured()
    if args.only in ("all", "thinking"):
        await _section_thinking()

    print("\nDone.\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
