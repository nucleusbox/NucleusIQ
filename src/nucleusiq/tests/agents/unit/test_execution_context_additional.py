"""Additional coverage for ExecutionContext protocol module."""

from __future__ import annotations

import importlib

from nucleusiq.agents.config import AgentConfig, AgentState


class _Ctx:
    llm = None
    tools = []
    memory = None
    prompt = None
    config = AgentConfig()
    role = "assistant"
    objective = "help"
    state = AgentState.INITIALIZING
    response_format = None

    @property
    def _logger(self):
        return None

    @property
    def _executor(self):
        return None

    @property
    def _current_llm_overrides(self):
        return {}

    def _resolve_response_format(self):
        return None

    def _get_structured_output_kwargs(self, output_config):
        return {}

    def _wrap_structured_output_result(self, response, output_config):
        return response


def test_execution_context_module_import_and_runtime_check():
    module = importlib.import_module("nucleusiq.agents.execution_context")
    ExecutionContext = module.ExecutionContext
    assert isinstance(_Ctx(), ExecutionContext)
