"""Live streaming via ``BaseOllama.call_stream``.

Run from ``src/providers/inference/ollama``::

    uv run python examples/agents/02_ollama_stream_live.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Load environment variables (optional) — same pattern as OpenAI examples
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nucleusiq.streaming.events import StreamEventType  # noqa: E402
from nucleusiq_ollama import BaseOllama  # noqa: E402


def _model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2")


async def main() -> None:
    host = os.getenv("OLLAMA_HOST")
    print(f"model={_model()!r} host={host or '(default)'} streaming", flush=True)
    llm = BaseOllama(model_name=_model(), async_mode=True)
    text_parts: list[str] = []
    async for ev in llm.call_stream(
        model=_model(),
        messages=[
            {"role": "user", "content": "Say hello in fewer than ten words."},
        ],
        max_output_tokens=128,
        temperature=0.3,
    ):
        if ev.type == StreamEventType.TOKEN and ev.token:
            text_parts.append(ev.token)
            print(ev.token, end="", flush=True)
        elif ev.type == StreamEventType.THINKING and ev.message:
            print(f"\n[thinking] {ev.message}", end="", flush=True)
        elif ev.type == StreamEventType.ERROR:
            print(f"\n[stream error] {ev.message}", flush=True)
            return
    print(f"\n\n--- complete prefix: {''.join(text_parts)[:200]!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
