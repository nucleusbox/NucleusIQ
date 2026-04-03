"""Tests for nucleusiq_gemini.nb_gemini.base — BaseGemini main class.

Uses mock fixtures from conftest.py to test without actual API access.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.unit._mock_factories import (
    make_candidate,
    make_function_call_part,
    make_response,
    make_stream_chunks,
)

# ====================================================================== #
# Construction tests                                                       #
# ====================================================================== #


class TestBaseGeminiConstruction:
    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_default_init(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        assert llm.model_name == "gemini-2.5-flash"
        assert llm.temperature == 0.7
        assert llm.top_k is None

    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_custom_init(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(
            model_name="gemini-2.5-pro",
            api_key="test-key",
            temperature=0.3,
            top_k=40,
        )
        assert llm.model_name == "gemini-2.5-pro"
        assert llm.temperature == 0.3
        assert llm.top_k == 40

    def test_missing_api_key_raises(self):
        from nucleusiq.llms.errors import AuthenticationError
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="GEMINI_API_KEY"):
                BaseGemini(api_key=None)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"})
    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_api_key_from_env(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini()
        assert llm.api_key == "env-key"


# ====================================================================== #
# Tool conversion tests                                                    #
# ====================================================================== #


class TestToolConversion:
    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_function_tool_conversion(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        spec = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": False,
            },
        }
        result = llm._convert_tool_spec(spec)
        assert result["name"] == "get_weather"
        assert "additionalProperties" not in result["parameters"]

    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_native_tool_passthrough(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        spec = {"type": "google_search", "google_search": {}}
        result = llm._convert_tool_spec(spec)
        assert result == spec


# ====================================================================== #
# Tool config tests                                                        #
# ====================================================================== #


class TestBuildToolConfig:
    def test_auto(self):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        result = BaseGemini._build_tool_config("auto")
        assert result["function_calling_config"]["mode"] == "AUTO"

    def test_none_str(self):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        result = BaseGemini._build_tool_config("none")
        assert result["function_calling_config"]["mode"] == "NONE"

    def test_required(self):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        result = BaseGemini._build_tool_config("required")
        assert result["function_calling_config"]["mode"] == "ANY"

    def test_none_value(self):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        result = BaseGemini._build_tool_config(None)
        assert result is None

    def test_specific_function(self):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        result = BaseGemini._build_tool_config(
            {
                "type": "function",
                "function": {"name": "get_weather"},
            }
        )
        cfg = result["function_calling_config"]
        assert cfg["mode"] == "ANY"
        assert "get_weather" in cfg["allowed_function_names"]


# ====================================================================== #
# call() tests                                                             #
# ====================================================================== #


class TestBaseGeminiCall:
    @pytest.mark.asyncio
    async def test_simple_call(self, mock_gemini_client):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        with patch(
            "nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None
        ):
            llm = BaseGemini(api_key="test-key")
            llm._client = mock_gemini_client

        result = await llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result.choices[0].message.content == "Hello!"
        mock_gemini_client.generate_content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_with_system_message(self, mock_gemini_client):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        with patch(
            "nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None
        ):
            llm = BaseGemini(api_key="test-key")
            llm._client = mock_gemini_client

        await llm.call(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
        )
        call_kwargs = mock_gemini_client.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.get("system_instruction") == "Be helpful"

    @pytest.mark.asyncio
    async def test_call_with_tools(self, mock_gemini_client):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        with patch(
            "nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None
        ):
            llm = BaseGemini(api_key="test-key")
            llm._client = mock_gemini_client

        tools = [{"name": "fn", "description": "d", "parameters": {}}]
        await llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        call_kwargs = mock_gemini_client.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert "tools" in config

    @pytest.mark.asyncio
    async def test_call_with_tool_choice(self, mock_gemini_client):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        with patch(
            "nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None
        ):
            llm = BaseGemini(api_key="test-key")
            llm._client = mock_gemini_client

        tools = [{"name": "fn", "description": "d", "parameters": {}}]
        await llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
            tool_choice="auto",
        )
        call_kwargs = mock_gemini_client.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert "tool_config" in config

    @pytest.mark.asyncio
    async def test_call_returns_tool_calls(self):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        tool_response = make_response(
            candidates=[
                make_candidate(
                    [
                        make_function_call_part(
                            "get_weather", {"location": "SF"}, "call_1"
                        )
                    ]
                )
            ]
        )
        mock_client = MagicMock()
        mock_client.generate_content = AsyncMock(return_value=tool_response)

        with patch(
            "nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None
        ):
            llm = BaseGemini(api_key="test-key")
            llm._client = mock_client

        result = await llm.call(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Weather?"}],
        )
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"


# ====================================================================== #
# call_stream() tests                                                      #
# ====================================================================== #


class TestBaseGeminiCallStream:
    @pytest.mark.asyncio
    async def test_basic_stream(self):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        chunks = make_stream_chunks(["Hello", " world"])
        mock_client = MagicMock()
        mock_client.generate_content_stream = AsyncMock(return_value=chunks)

        with patch(
            "nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None
        ):
            llm = BaseGemini(api_key="test-key")
            llm._client = mock_client

        events = []
        async for event in llm.call_stream(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
        ):
            events.append(event)

        token_events = [e for e in events if e.type == "token"]
        complete_events = [e for e in events if e.type == "complete"]
        assert len(token_events) == 2
        assert len(complete_events) == 1


# ====================================================================== #
# estimate_tokens tests                                                    #
# ====================================================================== #


class TestEstimateTokens:
    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_estimate_tokens(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        tokens = llm.estimate_tokens("Hello, world!")
        assert isinstance(tokens, int)
        assert tokens > 0

    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_estimate_tokens_empty(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        tokens = llm.estimate_tokens("")
        assert tokens >= 1


# ====================================================================== #
# BaseLLM contract tests                                                   #
# ====================================================================== #


class TestBaseLLMContract:
    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_is_base_llm(self, mock_init):
        from nucleusiq.llms.base_llm import BaseLLM
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        assert isinstance(llm, BaseLLM)

    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_has_native_tool_types(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        assert isinstance(llm.NATIVE_TOOL_TYPES, frozenset)
        assert "google_search" in llm.NATIVE_TOOL_TYPES

    @patch("nucleusiq_gemini.nb_gemini.client.GeminiClient.__init__", return_value=None)
    def test_has_supported_extensions(self, mock_init):
        from nucleusiq_gemini.nb_gemini.base import BaseGemini

        llm = BaseGemini(api_key="test-key")
        assert ".pdf" in llm.SUPPORTED_FILE_EXTENSIONS
        assert ".py" in llm.SUPPORTED_FILE_EXTENSIONS
