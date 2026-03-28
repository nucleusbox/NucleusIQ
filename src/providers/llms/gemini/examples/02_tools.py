"""Function calling (tools) with Gemini.

Demonstrates:
- Custom function tools via convert_tool_specs
- Native Gemini tools (Google Search, Code Execution)
- Tool call response handling

Prerequisites:
    pip install nucleusiq nucleusiq-gemini
    export GEMINI_API_KEY="your-key-here"
"""

import asyncio
import json

from nucleusiq_gemini import BaseGemini, GeminiTool


async def main():
    llm = BaseGemini(model_name="gemini-2.5-flash")

    # --- Custom function tools ---
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        }
    ]

    response = await llm.call(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"Tool call: {tc.function.name}")
            print(f"Arguments: {tc.function.arguments}")
            print(f"Call ID:   {tc.id}")

        # Simulate tool result and send back
        tool_result = json.dumps({"temperature": 22, "conditions": "partly cloudy"})
        followup = await llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "What's the weather in Tokyo?"},
                msg.to_dict(),
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_calls[0].id,
                    "name": "get_weather",
                    "content": tool_result,
                },
            ],
        )
        print(f"\nFinal response: {followup.choices[0].message.content}")
    else:
        print(f"Direct response: {msg.content}")

    # --- Native Gemini tools ---
    print("\n=== Native Google Search ===")
    search = GeminiTool.google_search()
    search_spec = llm.convert_tool_specs([search])
    print(f"Search tool spec: {json.dumps(search_spec, indent=2)}")

    code = GeminiTool.code_execution()
    print(f"Code execution tool: {code.name}")


if __name__ == "__main__":
    asyncio.run(main())
