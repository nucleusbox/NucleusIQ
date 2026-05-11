"""Minimal smoke: one ``BaseOllama.call()`` (no Agent).

Run from ``src/providers/inference/ollama``::

    uv run python examples/agents/00_ollama_smoke.py

Uses ``OLLAMA_API_KEY`` / ``OLLAMA_HOST`` from the environment or repo-root ``.env``.
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

from nucleusiq_ollama import BaseOllama  # noqa: E402


def _model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2")


async def main() -> None:
    host = os.getenv("OLLAMA_HOST")
    print(
        f"model={_model()!r} host={host or 'unset (SDK default, usually 127.0.0.1:11434)'}",
        flush=True,
    )
    llm = BaseOllama(model_name=_model(), async_mode=True)
    try:
        resp = await llm.call(
            model=_model(),
            messages=[
                {"role": "user", "content": "Reply with exactly one word: OK"},
            ],
            max_output_tokens=32,
            temperature=0.2,
        )
    except Exception as exc:
        print(f"CALL FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1) from exc

    if not getattr(resp, "choices", None):
        print("Empty choices in response.", flush=True)
        raise SystemExit(1)

    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    print(f"assistant: {text!r}", flush=True)
    print("Smoke test OK.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
