"""
Tests for BaseOpenAI integration.

These tests verify that BaseOpenAI works end-to-end with the OpenAI API.
Requires OPENAI_API_KEY environment variable to be set.
"""

import os
import pytest
import asyncio
from typing import Dict, Any, List

from nucleusiq.providers.llms.openai.nb_openai.base import BaseOpenAI


@pytest.fixture
def openai_client():
    """Create a BaseOpenAI client for testing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set, skipping OpenAI tests")
    return BaseOpenAI(
        model_name="gpt-3.5-turbo",
        api_key=api_key,
        max_retries=2,  # Lower retries for faster tests
    )


@pytest.mark.asyncio
async def test_basic_completion(openai_client):
    """Test basic text completion."""
    messages = [
        {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
    ]
    
    response = await openai_client.call(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=50,
    )
    
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    
    message = response.choices[0].message
    assert isinstance(message, dict)
    assert "content" in message
    assert message["content"] is not None
    assert "Hello" in message["content"] or "hello" in message["content"]


@pytest.mark.asyncio
async def test_function_calling(openai_client):
    """Test function calling capability."""
    messages = [
        {"role": "user", "content": "What is 5 + 3? Use the add function."}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    
    response = await openai_client.call(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        max_tokens=100,
    )
    
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    
    message = response.choices[0].message
    assert isinstance(message, dict)
    
    # Should have either function_call or content
    assert "function_call" in message or "content" in message


@pytest.mark.asyncio
async def test_error_handling_invalid_api_key():
    """Test that invalid API key raises appropriate error."""
    client = BaseOpenAI(
        model_name="gpt-3.5-turbo",
        api_key="invalid-key-12345",
        max_retries=1,
    )
    
    messages = [{"role": "user", "content": "Hello"}]
    
    with pytest.raises((ValueError, Exception)):
        await client.call(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=10,
        )


@pytest.mark.asyncio
async def test_error_handling_invalid_model(openai_client):
    """Test that invalid model name raises appropriate error."""
    messages = [{"role": "user", "content": "Hello"}]
    
    with pytest.raises((ValueError, Exception)):
        await openai_client.call(
            model="invalid-model-name-12345",
            messages=messages,
            max_tokens=10,
        )


@pytest.mark.asyncio
async def test_estimate_tokens(openai_client):
    """Test token estimation."""
    text = "Hello, world! This is a test."
    tokens = openai_client.estimate_tokens(text)
    
    assert isinstance(tokens, int)
    assert tokens > 0
    assert tokens < 100  # Should be reasonable for this text


@pytest.mark.asyncio
async def test_temperature_override(openai_client):
    """Test that temperature can be overridden in call."""
    messages = [{"role": "user", "content": "Say 'test'"}]
    
    # Call with custom temperature
    response = await openai_client.call(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0,  # Deterministic
        max_tokens=10,
    )
    
    assert response is not None
    assert hasattr(response, "choices")


@pytest.mark.asyncio
async def test_max_tokens(openai_client):
    """Test max_tokens parameter."""
    messages = [{"role": "user", "content": "Count from 1 to 20"}]
    
    response = await openai_client.call(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=5,  # Very short response
    )
    
    assert response is not None
    message = response.choices[0].message
    assert "content" in message
    # Response should be short due to max_tokens limit


@pytest.mark.asyncio
async def test_multiple_messages(openai_client):
    """Test conversation with multiple messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    
    response = await openai_client.call(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=50,
    )
    
    assert response is not None
    message = response.choices[0].message
    assert "content" in message
    # Should mention 4 or answer the question
    content = message["content"].lower()
    assert "4" in content or "four" in content


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

