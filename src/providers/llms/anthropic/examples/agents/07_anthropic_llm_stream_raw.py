"""Low-level streaming: ``BaseAnthropic.call_stream`` (no Agent).

Same pattern as Groq ``07_groq_stream_live.py`` — exercises provider streaming only.

Run from ``src/providers/llms/anthropic``::

    uv run python examples/agents/07_anthropic_llm_stream_raw.py

Requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from util_env import load_repo_dotenv  # noqa: E402

load_repo_dotenv()

from nucleusiq.streaming.events import StreamEventType  # noqa: E402
from nucleusiq_anthropic import BaseAnthropic  # noqa: E402


def _model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")


async def main() -> None:
    llm = BaseAnthropic(model_name=_model(), async_mode=True)
    print(f"model={_model()} (streaming, raw LLM)\n", flush=True)
    parts: list[str] = []
    async for ev in llm.call_stream(
        model=_model(),
        messages=[{"role": "user", "content": "Say hello in fewer than ten words."}],
        max_output_tokens=128,
        temperature=0.3,
    ):
        if ev.type == StreamEventType.TOKEN and ev.token:
            parts.append(ev.token)
            print(ev.token, end="", flush=True)
        elif ev.type == StreamEventType.ERROR:
            print(f"\n[stream error] {ev.message}", flush=True)
            return
    print(f"\n\n--- joined: {''.join(parts)!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
