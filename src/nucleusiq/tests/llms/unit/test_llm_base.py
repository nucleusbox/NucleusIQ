"""Tests for llms/base.py, llms/base_llm.py, and llms/mock_llm.py."""

import json
from unittest.mock import MagicMock

import pytest
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.llms.mock_llm import MockLLM

# ═══════════════════════════════════════════════════════════════════════════════
# BaseLLM
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseLLM:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLLM()

    def test_convert_tool_specs_with_get_spec(self):
        class ConcreteLLM(BaseLLM):
            async def call(self, **kwargs):
                pass

        llm = ConcreteLLM()
        tool = MagicMock()
        tool.get_spec.return_value = {"name": "tool1", "parameters": {}}
        result = llm.convert_tool_specs([tool])
        assert len(result) == 1
        assert result[0]["name"] == "tool1"

    def test_convert_tool_specs_dict_passthrough(self):
        class ConcreteLLM(BaseLLM):
            async def call(self, **kwargs):
                pass

        llm = ConcreteLLM()
        raw = {"name": "raw_tool", "parameters": {}}
        result = llm.convert_tool_specs([raw])
        assert result[0] is raw

    def test_convert_tool_spec_default(self):
        class ConcreteLLM(BaseLLM):
            async def call(self, **kwargs):
                pass

        llm = ConcreteLLM()
        spec = {"name": "test"}
        assert llm._convert_tool_spec(spec) is spec


# ═══════════════════════════════════════════════════════════════════════════════
# MockLLM
# ═══════════════════════════════════════════════════════════════════════════════


class TestMockLLM:
    def test_init(self):
        llm = MockLLM()
        assert llm.model_name == "mock-model"
        assert llm._call_count == 0

    @pytest.mark.asyncio
    async def test_first_call_with_tools_openai_format(self):
        llm = MockLLM()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                    },
                },
            }
        ]
        response = await llm.call(
            model="mock", messages=[{"role": "user", "content": "5 + 3"}], tools=tools
        )
        assert response.choices[0].message.function_call is not None
        assert response.choices[0].message.tool_calls is not None
        fn = response.choices[0].message.function_call
        assert fn["name"] == "add"
        args = json.loads(fn["arguments"])
        assert args["a"] == 5

    @pytest.mark.asyncio
    async def test_first_call_with_tools_generic_format(self):
        llm = MockLLM()
        tools = [
            {
                "name": "multiply",
                "parameters": {
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                },
            }
        ]
        response = await llm.call(
            model="mock", messages=[{"role": "user", "content": "2 and 3"}], tools=tools
        )
        fn = response.choices[0].message.function_call
        assert fn["name"] == "multiply"

    @pytest.mark.asyncio
    async def test_second_call_with_function_result(self):
        llm = MockLLM()
        llm._call_count = 1
        response = await llm.call(
            model="mock",
            messages=[
                {"role": "user", "content": "sum"},
                {"role": "function", "content": "8"},
            ],
        )
        assert "8" in response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_second_call_echo(self):
        llm = MockLLM()
        llm._call_count = 1
        response = await llm.call(
            model="mock",
            messages=[{"role": "user", "content": "hello world"}],
        )
        assert "hello world" in response.choices[0].message.content

    def test_create_completion_sync(self):
        llm = MockLLM()
        result = llm.create_completion(messages=[{"role": "user", "content": "test"}])
        assert isinstance(result, str)

    def test_convert_tool_specs(self):
        llm = MockLLM()
        tool = MagicMock()
        tool.get_spec.return_value = {"name": "t", "parameters": {}}
        result = llm.convert_tool_specs([tool])
        assert result[0]["name"] == "t"

    def test_convert_tool_specs_dict(self):
        llm = MockLLM()
        raw = {"name": "raw"}
        result = llm.convert_tool_specs([raw])
        assert result[0]["name"] == "raw"

    @pytest.mark.asyncio
    async def test_tool_result_message(self):
        llm = MockLLM()
        llm._call_count = 1
        response = await llm.call(
            model="mock",
            messages=[
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "tool_result"},
            ],
        )
        assert "tool_result" in response.choices[0].message.content
