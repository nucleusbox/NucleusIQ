"""Using Gemini's native server-side tools.

Demonstrates Google Search, Code Execution, URL Context, and Google Maps
tools that execute on Google's servers — no local tool execution needed.

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio

from nucleusiq_gemini import BaseGemini, GeminiTool


async def demo_google_search():
    """Ground a response with Google Search."""
    llm = BaseGemini(model_name="gemini-2.5-flash")
    search = GeminiTool.google_search()

    result = await llm.call(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "What are the latest developments in quantum computing?",
            }
        ],
        tools=[search],
        max_output_tokens=1024,
    )
    print("=== Google Search Grounding ===")
    if result.choices:
        print(result.choices[0].message.content or "(no content)")
    print()


async def demo_code_execution():
    """Let Gemini execute Python code server-side."""
    llm = BaseGemini(model_name="gemini-2.5-flash")
    code = GeminiTool.code_execution()

    result = await llm.call(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "Calculate the first 20 Fibonacci numbers and show them.",
            }
        ],
        tools=[code],
        max_output_tokens=1024,
    )
    print("=== Code Execution ===")
    if result.choices:
        print(result.choices[0].message.content or "(no content)")
    print()


async def demo_url_context():
    """Fetch and understand a web page."""
    llm = BaseGemini(model_name="gemini-2.5-flash")
    url_ctx = GeminiTool.url_context()

    result = await llm.call(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "Summarize the key features listed on https://ai.google.dev/gemini-api/docs/",
            }
        ],
        tools=[url_ctx],
        max_output_tokens=1024,
    )
    print("=== URL Context ===")
    if result.choices:
        print(result.choices[0].message.content or "(no content)")
    print()


async def demo_google_maps():
    """Ground a response with Google Maps data."""
    llm = BaseGemini(model_name="gemini-2.5-flash")
    maps = GeminiTool.google_maps()

    result = await llm.call(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "Find the best-rated Italian restaurants near Times Square, New York.",
            }
        ],
        tools=[maps],
        max_output_tokens=1024,
    )
    print("=== Google Maps ===")
    if result.choices:
        print(result.choices[0].message.content or "(no content)")
    print()


async def main():
    await demo_google_search()
    await demo_code_execution()
    await demo_url_context()
    await demo_google_maps()


if __name__ == "__main__":
    asyncio.run(main())
