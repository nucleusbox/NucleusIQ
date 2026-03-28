"""Streaming with Gemini.

Demonstrates real-time token streaming via call_stream().

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio

from nucleusiq_gemini import BaseGemini


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")

    print("=== Streaming Response ===")
    print()

    async for event in llm.call_stream(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a haiku about programming."},
        ],
        max_output_tokens=256,
        temperature=0.9,
    ):
        if event.type == "thinking":
            print(f"[thinking] {event.message}")
        elif event.type == "token":
            print(event.token, end="", flush=True)
        elif event.type == "complete":
            print()
            print("\n=== Stream Complete ===")
            print(f"Full content: {event.content}")
            if event.metadata:
                if "usage" in event.metadata:
                    usage = event.metadata["usage"]
                    print(f"Tokens used: {usage.get('total_tokens', 'N/A')}")
                if "tool_calls" in event.metadata:
                    print(f"Tool calls: {len(event.metadata['tool_calls'])}")
        elif event.type == "error":
            print(f"\n[ERROR] {event.message}")


if __name__ == "__main__":
    asyncio.run(main())
