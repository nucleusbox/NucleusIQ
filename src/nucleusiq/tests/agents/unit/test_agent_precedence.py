"""
Tests for Agent prompt usage in message construction.

``prompt`` is the only source for system/user preamble messages.
``role`` and ``objective`` on the Agent are labels (defaults) and are not
passed to ``MessageBuilder`` or used for LLM message text.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import logging

import pytest
from nucleusiq.agents import Agent
from nucleusiq.agents.config import AgentConfig, AgentState
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.task import Task
from nucleusiq.llms.mock_llm import MockLLM

from nucleusiq.tests.conftest import make_test_prompt


class TestAgentPromptPrecedence:
    """Test that LLM messages come from ``prompt.system`` / ``prompt.user`` only."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockLLM(model_name="test-model")

    @pytest.fixture
    def agent_with_prompt(self, mock_llm):
        """Agent with prompt and distinct role/objective labels."""
        prompt = make_test_prompt(
            system="You are a helpful calculator assistant.",
            user="Answer questions accurately.",
        )
        return Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            prompt=prompt,
            llm=mock_llm,
            config=AgentConfig(verbose=True),
        )

    @pytest.mark.asyncio
    async def test_prompt_takes_precedence_in_messages(self, agent_with_prompt):
        """System message text comes from ``prompt.system`` only, not role/objective labels."""
        task = Task.from_dict({"id": "task1", "objective": "What is 5 + 3?"})

        messages = MessageBuilder.build(
            task,
            prompt=agent_with_prompt.prompt,
        )

        assert len(messages) >= 1
        system_msg = next((m for m in messages if m.role == "system"), None)
        assert system_msg is not None
        assert system_msg.content == "You are a helpful calculator assistant."
        assert "Calculator" not in (system_msg.content or "")
        assert "Perform calculations" not in (system_msg.content or "")

    @pytest.mark.asyncio
    async def test_prompt_overrides_role_objective_in_system_message(
        self, agent_with_prompt
    ):
        """``prompt.system`` defines the system message; role/objective labels are not injected."""
        task = Task.from_dict({"id": "task1", "objective": "What is 5 + 3?"})

        assert agent_with_prompt.role == "Calculator"
        assert agent_with_prompt.objective == "Perform calculations"

        messages = MessageBuilder.build(
            task,
            prompt=agent_with_prompt.prompt,
        )

        system_msg = next((m for m in messages if m.role == "system"), None)
        assert system_msg is not None
        assert system_msg.content == "You are a helpful calculator assistant."
        assert "Calculator" not in (system_msg.content or "")
        assert "Perform calculations" not in (system_msg.content or "")

    @pytest.mark.asyncio
    async def test_no_warning_when_no_role_objective(self, mock_llm, caplog):
        """No override warning when role/objective are empty strings."""
        prompt = make_test_prompt(system="You are a helpful assistant.")
        agent = Agent(
            name="TestAgent",
            role="",
            objective="",
            prompt=prompt,
            llm=mock_llm,
            config=AgentConfig(verbose=True),
        )

        task = Task.from_dict({"id": "task1", "objective": "Test"})

        with caplog.at_level(logging.INFO):
            messages = MessageBuilder.build(
                task,
                prompt=agent.prompt,
            )

        assert len(messages) >= 1
        assert not any(
            "overriding" in record.message.lower() for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_prompt_user_included_in_messages(self, agent_with_prompt):
        """``prompt.user`` is included as a user message when set."""
        task = Task.from_dict({"id": "task1", "objective": "What is 5 + 3?"})

        messages = MessageBuilder.build(
            task,
            prompt=agent_with_prompt.prompt,
        )

        user_messages = [m for m in messages if m.role == "user"]

        assert len(user_messages) >= 1
        assert any(
            "Answer questions accurately" in (m.content or "") for m in user_messages
        )

    @pytest.mark.asyncio
    async def test_prompt_without_system_falls_back_to_role(self, mock_llm):
        """When ``prompt.system`` is unset, no role/objective fallback into system text."""
        from nucleusiq.prompts.base import BasePrompt

        class MockPromptWithoutSystem(BasePrompt):
            @property
            def technique_name(self) -> str:
                return "mock"

            def _construct_prompt(self, **kwargs) -> str:
                return kwargs.get("user", "")

            def format_prompt(self, **kwargs) -> str:
                return self._construct_prompt(**kwargs)

        prompt = MockPromptWithoutSystem()
        prompt.user = "Answer questions."
        prompt.system = None

        agent = Agent(
            name="TestAgent",
            role="Calculator",
            objective="Perform calculations",
            prompt=prompt,
            llm=mock_llm,
            config=AgentConfig(verbose=True),
        )

        task = Task.from_dict({"id": "task1", "objective": "What is 5 + 3?"})
        messages = MessageBuilder.build(
            task,
            prompt=agent.prompt,
        )

        system_msg = next((m for m in messages if m.role == "system"), None)
        assert system_msg is None or "Calculator" not in (system_msg.content or "")

    @pytest.mark.asyncio
    async def test_execution_with_prompt_precedence(self, agent_with_prompt):
        """Full execution flow uses the agent's prompt for messages."""
        task = Task.from_dict({"id": "task1", "objective": "What is 5 + 3?"})

        result = await agent_with_prompt.execute(task)

        assert result is not None
        assert agent_with_prompt.state == AgentState.COMPLETED
