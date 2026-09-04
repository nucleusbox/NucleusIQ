"""Preflight an OpenAI-compatible endpoint before wiring an agent to it.

``validate()`` performs the only network check this provider ever does
implicitly: ``GET /v1/models``.  It never raises for a reachability or
model-name problem — it reports, so you can print guidance instead of
catching an exception.

Run:
    python examples/01_validate_endpoint.py
"""

from __future__ import annotations

import asyncio
import os

from nucleusiq_openai_compatible import OpenAICompatibleLLM

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gemma-4-27b-it")


async def main() -> int:
    llm = OpenAICompatibleLLM(
        base_url=BASE_URL,
        model=MODEL,
        engine="vllm",
        # Deliberately omitted so the probe has something to discover.
        # In production, pass context_window= to skip the round-trip.
    )

    print(f"Validating {BASE_URL} (model={MODEL})\n")
    report = await llm.validate()
    print(report.render())

    if not report.ok:
        print("\nFix the errors above before building an agent on this endpoint.")
        return 1

    print(f"\nResolved context window: {llm.get_context_window():,} tokens")
    print(f"Tools supported        : {llm.capabilities.supports_tools}")
    print(f"Reasoning supported    : {llm.capabilities.supports_reasoning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
