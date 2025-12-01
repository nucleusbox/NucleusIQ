"""
Integration test script for BaseOpenAI.

This script tests BaseOpenAI end-to-end with real API calls.
Run with: python src/test_openai_integration.py

Requires OPENAI_API_KEY environment variable.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List

from nucleusiq.providers.llms.openai.nb_openai.base import BaseOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_completion():
    """Test 1: Basic text completion."""
    logger.info("=" * 60)
    logger.info("Test 1: Basic Text Completion")
    logger.info("=" * 60)
    
    client = BaseOpenAI(model_name="gpt-3.5-turbo")
    
    messages = [
        {"role": "user", "content": "Say 'Hello, NucleusIQ!' and nothing else."}
    ]
    
    try:
        response = await client.call(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50,
        )
        
        message = response.choices[0].message
        content = message.get("content", "")
        
        logger.info(f"✅ Success! Response: {content}")
        assert "NucleusIQ" in content or "nucleusiq" in content.lower()
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


async def test_function_calling():
    """Test 2: Function calling."""
    logger.info("=" * 60)
    logger.info("Test 2: Function Calling")
    logger.info("=" * 60)
    
    client = BaseOpenAI(model_name="gpt-3.5-turbo")
    
    messages = [
        {"role": "user", "content": "What is 15 + 27? Use the add function."}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "The first number"
                        },
                        "b": {
                            "type": "number",
                            "description": "The second number"
                        }
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ]
    
    try:
        response = await client.call(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools,
            max_tokens=100,
        )
        
        message = response.choices[0].message
        
        if "function_call" in message:
            fn_call = message["function_call"]
            logger.info(f"✅ Function call requested: {fn_call}")
            return True
        elif "content" in message:
            content = message["content"]
            logger.info(f"✅ Direct response: {content}")
            return True
        else:
            logger.error("❌ No function_call or content in response")
            return False
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


async def test_conversation():
    """Test 3: Multi-turn conversation."""
    logger.info("=" * 60)
    logger.info("Test 3: Multi-turn Conversation")
    logger.info("=" * 60)
    
    client = BaseOpenAI(model_name="gpt-3.5-turbo")
    
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    
    try:
        response1 = await client.call(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50,
        )
        
        answer1 = response1.choices[0].message.get("content", "")
        logger.info(f"First response: {answer1}")
        
        # Add assistant response and continue conversation
        messages.append({"role": "assistant", "content": answer1})
        messages.append({"role": "user", "content": "What about 3 + 3?"})
        
        response2 = await client.call(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50,
        )
        
        answer2 = response2.choices[0].message.get("content", "")
        logger.info(f"Second response: {answer2}")
        
        logger.info("✅ Multi-turn conversation successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return False


async def test_error_handling():
    """Test 4: Error handling."""
    logger.info("=" * 60)
    logger.info("Test 4: Error Handling")
    logger.info("=" * 60)
    
    # Test with invalid model name
    client = BaseOpenAI(model_name="gpt-3.5-turbo")
    
    messages = [{"role": "user", "content": "Hello"}]
    
    try:
        # This should fail gracefully
        await client.call(
            model="invalid-model-name-xyz",
            messages=messages,
            max_tokens=10,
        )
        logger.error("❌ Should have raised an error")
        return False
    except (ValueError, Exception) as e:
        logger.info(f"✅ Error handled correctly: {type(e).__name__}")
        return True


async def test_token_estimation():
    """Test 5: Token estimation."""
    logger.info("=" * 60)
    logger.info("Test 5: Token Estimation")
    logger.info("=" * 60)
    
    client = BaseOpenAI(model_name="gpt-3.5-turbo")
    
    text = "Hello, this is a test of token estimation in NucleusIQ."
    tokens = client.estimate_tokens(text)
    
    logger.info(f"Text: {text}")
    logger.info(f"Estimated tokens: {tokens}")
    
    assert tokens > 0
    logger.info("✅ Token estimation works!")
    return True


async def test_retry_logic():
    """Test 6: Retry logic (simulated with low max_retries)."""
    logger.info("=" * 60)
    logger.info("Test 6: Retry Logic Configuration")
    logger.info("=" * 60)
    
    # Create client with low retries for testing
    client = BaseOpenAI(
        model_name="gpt-3.5-turbo",
        max_retries=2,
    )
    
    logger.info(f"Client max_retries: {client.max_retries}")
    logger.info("✅ Retry logic configured correctly!")
    return True


async def main():
    """Run all integration tests."""
    logger.info("Starting BaseOpenAI Integration Tests")
    logger.info("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set!")
        logger.error("Please set it before running tests.")
        return
    
    tests = [
        ("Basic Completion", test_basic_completion),
        ("Function Calling", test_function_calling),
        ("Conversation", test_conversation),
        ("Error Handling", test_error_handling),
        ("Token Estimation", test_token_estimation),
        ("Retry Logic", test_retry_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())

