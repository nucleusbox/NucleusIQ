"""Live streaming chat via ``BaseGroq.call_stream`` (no Agent).

Run from ``src/providers/inference/groq``::

    uv run python examples/agents/07_groq_stream_live.py

Requires ``GROQ_API_KEY`` (e.g. repo-root ``.env``). Uses the same 429/retry
policy as non-streaming chat for the **stream open** call.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from util_env import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from nucleusiq.streaming.events import StreamEventType  # noqa: E402
from nucleusiq_groq import BaseGroq  # noqa: E402


def _model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


async def main() -> None:
    llm = BaseGroq(model_name=_model(), async_mode=True)
    print(f"model={_model()} (streaming)", flush=True)
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
        elif ev.type == StreamEventType.ERROR:
            print(f"\n[stream error] {ev.message}", flush=True)
            return
    print(f"\n\n--- complete: {''.join(text_parts)[:200]!r}...", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
