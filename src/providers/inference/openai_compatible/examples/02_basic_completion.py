"""Direct provider use: one call, then the same request streamed.

No agent here — this is the provider on its own, which is the clearest way to
see what actually goes over the wire.

Run:
    python examples/02_basic_completion.py
"""

from __future__ import annotations

import asyncio
import os

from nucleusiq_openai_compatible import OpenAICompatibleLLM

BASE_URL = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
MODEL = os.getenv("OPENAI_COMPATIBLE_MODEL", "gemma-4-27b-it")


def build_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        base_url=BASE_URL,
        model=MODEL,
        engine="vllm",
        # Declaring the window skips the /v1/models probe and guarantees the
        # context engine budgets against the server's real --max-model-len.
        context_window=32_768,
    )


async def basic_call(llm: OpenAICompatibleLLM) -> None:
    print("=== Single call ===")
    response = await llm.call(
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Name three uses for a paperclip."},
        ],
        max_output_tokens=200,
        temperature=0.3,
    )

    print(response.content)
    print(f"\nfinish_reason : {response.finish_reason}")
    if response.usage_reported:
        print(
            f"tokens        : {response.prompt_tokens} in / "
            f"{response.completion_tokens} out"
        )
    else:
        # llama.cpp and some gateways omit usage entirely.
        print("tokens        : not reported by this server")


async def streaming_call(llm: OpenAICompatibleLLM) -> None:
    print("\n=== Streamed ===")
    async for event in llm.call_stream(
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        max_output_tokens=100,
    ):
        if event.type == "token":
            print(event.token, end="", flush=True)
        elif event.type == "complete":
            print()

    # A stream is single-use, so the accumulated result is kept here.
    outcome = llm.last_stream
    if outcome and outcome.response and outcome.response.usage_reported:
        print(f"tokens        : {outcome.response.total_tokens} total")


async def main() -> None:
    llm = build_llm()
    print(f"{llm!r}\n")  # credentials are redacted in repr
    await basic_call(llm)
    await streaming_call(llm)


if __name__ == "__main__":
    asyncio.run(main())
