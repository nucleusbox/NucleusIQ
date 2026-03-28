"""Quickstart: Basic Gemini usage with NucleusIQ.

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio

from nucleusiq_gemini import BaseGemini


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")

    response = await llm.call(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the three laws of robotics?"},
        ],
        max_output_tokens=512,
        temperature=0.7,
    )

    print("=== Response ===")
    print(response.choices[0].message.content)

    if response.usage:
        print("\n=== Usage ===")
        print(f"Prompt tokens:     {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens:      {response.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
