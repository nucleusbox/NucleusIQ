# tests/test_auto_chain_of_thought_prompt.py

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.llms.mock_llm import MockLLM  # or however you import your mock
# If you have a fixture named mock_llm, you can also do `def test_auto_cot_creation_success(mock_llm): ...`

class TestAutoChainOfThoughtPrompt:

    @pytest.fixture
    def mock_llm(self):
        """Simple fixture returning a mock LLM that returns a constant chain text."""
        return MockLLM()

    def test_auto_cot_creation_success(self, mock_llm):
        """
        Provide valid task, questions, llm, etc. => success
        """
        task = "Provide detailed reasoning for these math problems."
        questions = [
            "Calculate the product of prime numbers < 10.",
            "Determine if 29 is prime."
        ]
        system_prompt = "You are an assistant specialized in math."
        user_prompt = "Here are your problems:"

        auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                llm=mock_llm,
                num_clusters=2,
                max_questions_per_cluster=1,
                instruction="Let's think step by step.",
                system=system_prompt,
                user=user_prompt
            )
        )
        # Then we call format_prompt or _construct_prompt
        prompt_text = auto_cot.format_prompt(task=task, questions=questions)

        # The mock LLM returns "This is a mock reasoning chain." for each question
        # So final prompt should have system_prompt, no context, chain for 2 Qs, user_prompt, cot
        assert "You are an assistant specialized in math." in prompt_text
        assert "Calculate the product of prime numbers < 10." in prompt_text
        assert "Determine if 29 is prime." in prompt_text
        assert "Here are your problems:" in prompt_text
        assert "Let's think step by step." in prompt_text

    def test_auto_cot_missing_llm(self):
        """
        Omit llm => should raise an error that llm is not set.
        """
        auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                num_clusters=2,
                max_questions_per_cluster=1,
                instruction="Let's think step by step.",
                # llm not provided
                system="System stuff.",
                user="User stuff."
            )
        )
        with pytest.raises(ValueError) as exc_info:
            # We pass valid task & questions but no llm => error in pre_format_validation
            auto_cot.format_prompt(task="some task", questions=["Q1","Q2"])
        assert "AutoChainOfThoughtPrompt requires 'llm' to be set" in str(exc_info.value)

    def test_auto_cot_negative_num_clusters(self, mock_llm):
        """
        num_clusters < 1 => pydantic validation error
        """
        with pytest.raises(ValueError) as exc_info:
            PromptFactory.create_prompt(
                PromptTechnique.AUTO_CHAIN_OF_THOUGHT
            ).configure(
                llm=mock_llm,
                num_clusters=-1,
                max_questions_per_cluster=1
            ).format_prompt(task="T", questions=["Q"])
        assert "num_clusters must be >= 1" in str(exc_info.value)

    def test_auto_cot_zero_max_questions_per_cluster(self, mock_llm):
        """
        max_questions_per_cluster < 1 => error
        """
        with pytest.raises(ValueError) as exc_info:
            PromptFactory.create_prompt(
                PromptTechnique.AUTO_CHAIN_OF_THOUGHT
            ).configure(
                llm=mock_llm,
                num_clusters=2,
                max_questions_per_cluster=0
            ).format_prompt(task="Task", questions=["Q"])
        assert "max_questions_per_cluster must be >= 1" in str(exc_info.value)

    def test_auto_cot_missing_task_or_questions(self, mock_llm):
        """
        'task' & 'questions' are required => if missing => the base class or pre_format check => error
        """
        auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                llm=mock_llm,
                num_clusters=2,
                max_questions_per_cluster=1
            )
        )
        # Omit 'task' and 'questions' => base class sees them missing => error
        with pytest.raises(ValueError) as exc_info:
            auto_cot.format_prompt()
        assert "Missing required field 'task' or it's empty." in str(exc_info.value)

    def test_auto_cot_empty_questions_list(self, mock_llm):
        """
        Provide an empty list for 'questions' => pre_format_validation raises error
        """
        auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                llm=mock_llm,
                num_clusters=2,
                max_questions_per_cluster=1,
                system="System info.",
                user="User info."
            )
        )
        with pytest.raises(ValueError) as exc_info:
            auto_cot.format_prompt(task="A valid task", questions=[])  # empty => not allowed
        assert "non-empty list of 'questions'" in str(exc_info.value)

    def test_auto_cot_empty_task_string(self, mock_llm):
        """
        Provide an empty string for 'task' => also triggers error in _pre_format_validation.
        """
        auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                llm=mock_llm,
                num_clusters=2,
                max_questions_per_cluster=1,
                system="System info.",
                user="User info."
            )
        )
        with pytest.raises(ValueError) as exc_info:
            auto_cot.format_prompt(task="   ", questions=["Q1"])
        assert "Missing required field 'task' or it's empty" in str(exc_info.value)

    def test_auto_cot_no_system_and_context(self, mock_llm):
        """
        Omitting 'system' and 'context' is allowed => but 'task', 'questions', 'llm' are present => it builds a prompt anyway.
        """
        auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                llm=mock_llm,
                num_clusters=2,
                max_questions_per_cluster=1,
                instruction="Step by step."
            )
        )
        # Provide a minimal task & questions => should succeed
        prompt_text = auto_cot.format_prompt(
            task="Solve the following questions:",
            questions=["Q1", "Q2"]
        )
        # Check it includes the final chain for 2 Qs, no system, no context => leading blank lines
        # The mock LLM returns "This is a mock reasoning chain." => verify
        assert "Solve the following questions" in prompt_text
        assert "Q1" in prompt_text
        assert "Q2" in prompt_text
        assert "Step by step." in prompt_text

    def test_auto_cot_extra_field_error(self, mock_llm):
        """
        Provide an unrecognized field => base class should raise an error.
        """
        with pytest.raises(ValueError) as exc_info:
            auto_cot = (
                PromptFactory
                .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
                .configure(
                    llm=mock_llm,
                    num_clusters=2,
                    max_questions_per_cluster=1,
                    unknown_field="Should fail"
                )
            )
            auto_cot.format_prompt(task="T", questions=["Q"])
        assert "Field 'unknown_field' is not recognized" in str(exc_info.value)
    
    def test_auto_cot_includes_task_in_prompt(self, mock_llm):
        """
        Verifies the 'task' string appears in the final constructed prompt.
        """
        task = "Solve the following geometry questions carefully."
        questions = ["What is the area of a circle with radius 5?","Q1", "Q2"]
        auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                llm=mock_llm,
                # We'll omit system, context, user for brevity
                # but supply an instruction
                num_clusters=2,
                max_questions_per_cluster=1,
                instruction="Please provide step-by-step reasoning."
            )
        )
        prompt_text = auto_cot.format_prompt(
            task=task,
            questions=questions
        )
        # Check that 'task' is indeed in the output
        assert task in prompt_text, "The 'task' text should appear in the final prompt."
        # Also check the generated example
        assert "What is the area of a circle with radius 5?\nA:" in prompt_text
        # And the fallback instruction
        assert "Please provide step-by-step reasoning." in prompt_text

