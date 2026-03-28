"""Provider portability: Same code, different providers.

Demonstrates NucleusIQ's provider-agnostic design. The same Agent
code works with both OpenAI and Gemini — just swap the LLM instance.

Prerequisites:
    pip install nucleusiq nucleusiq-openai nucleusiq-gemini
    export OPENAI_API_KEY="your-openai-key"
    export GEMINI_API_KEY="your-gemini-key"
"""

import asyncio
import os


async def run_with_provider(llm, provider_name: str):
    """Run the same conversation with any BaseLLM provider."""
    print(f"\n{'=' * 50}")
    print(f"  Provider: {provider_name}")
    print(f"{'=' * 50}")

    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. Reply in one sentence.",
        },
        {"role": "user", "content": "What is the capital of Japan?"},
    ]

    response = await llm.call(
        model=llm.model_name,
        messages=messages,
        max_output_tokens=100,
        temperature=0.3,
    )

    content = response.choices[0].message.content
    print(f"Response: {content}")

    if response.usage:
        print(f"Tokens:   {response.usage.total_tokens}")


async def main():
    providers = []

    if os.getenv("GEMINI_API_KEY"):
        from nucleusiq_gemini import BaseGemini

        providers.append(
            (BaseGemini(model_name="gemini-2.5-flash"), "Gemini 2.5 Flash")
        )

    if os.getenv("OPENAI_API_KEY"):
        from nucleusiq_openai import BaseOpenAI

        providers.append((BaseOpenAI(model_name="gpt-4o-mini"), "OpenAI GPT-4o Mini"))

    if not providers:
        print("Set GEMINI_API_KEY and/or OPENAI_API_KEY to run this example.")
        return

    for llm, name in providers:
        await run_with_provider(llm, name)

    print(f"\n{'=' * 50}")
    print(f"  Same code, {len(providers)} provider(s) — zero changes.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
