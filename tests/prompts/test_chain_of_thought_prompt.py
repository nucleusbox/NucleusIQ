# tests/test_chain_of_thought_prompt.py

import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique

class TestChainOfThoughtPrompt:
    def test_chain_of_thought_minimal_success(self):
        """
        Provide system + user only, rely on defaults:
        - use_cot defaults True
        - cot_instruction defaults to "Let's think step by step."
        """
        cot_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
            .configure(
                system="You are an analytical assistant.",
                user="Is the following statement correct?"
                # no cot_instruction => default
            )
        )
        prompt_text = cot_prompt.format_prompt()
        expected_prompt = (
            "You are an analytical assistant.\n\n"
            "Is the following statement correct?\n\n"
            "Let's think step by step."
        )
        assert prompt_text.strip() == expected_prompt.strip()

    def test_chain_of_thought_with_custom_instruction(self):
        """
        Provide a custom CoT instruction, keep use_cot=True => final prompt uses that instruction.
        """
        custom_instr = "Please provide a thorough reasoning process."
        cot_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
            .configure(
                system="You are an AI reasoner.",
                user="What is the capital of France?",
                cot_instruction=custom_instr
            )
        )
        prompt_text = cot_prompt.format_prompt()
        expected_prompt = (
            "You are an AI reasoner.\n\n"
            "What is the capital of France?\n\n"
            f"{custom_instr}"
        )
        assert prompt_text.strip() == expected_prompt.strip()

    def test_chain_of_thought_cannot_set_use_cot_false(self):
        """
        If user tries to set use_cot=False => we raise a ValueError in configure().
        """
        with pytest.raises(ValueError) as exc:
            (
                PromptFactory
                .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
                .configure(
                    system="System text.",
                    user="User text.",
                    use_cot=False,  # This is disallowed
                )
            )
        assert "cannot be set to False" in str(exc.value)

    def test_chain_of_thought_missing_system(self):
        """
        'system' is in input_variables => if missing or empty => raises ValueError at format_prompt time.
        """
        cot_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
            .configure(
                system="",  # empty
                user="Should I invest in stocks?",
                cot_instruction="Analyze carefully."
            )
        )
        with pytest.raises(ValueError) as exc_info:
            cot_prompt.format_prompt()
        assert "Missing required field 'system' or it's empty." in str(exc_info.value)

    def test_chain_of_thought_missing_user(self):
        """
        'user' is also in input_variables => if missing or empty => ValueError at format.
        """
        cot_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
            .configure(
                system="You are a logic engine.",
                user=None,  # or blank
                cot_instruction="Think step by step."
            )
        )
        with pytest.raises(ValueError) as exc_info:
            cot_prompt.format_prompt()
        assert "Missing required field 'user' or it's empty." in str(exc_info.value)

    def test_chain_of_thought_no_cot_instruction_in_format(self):
        """
        If user doesn't set cot_instruction, we fallback to default "Let's think step by step."
        at pre-validation if it's empty.
        """
        cot_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
            .configure(
                system="System instructions.",
                user="What is 2+2?"
            )
        )
        # remove any existing instructions
        cot_prompt.cot_instruction = ""
        prompt_text = cot_prompt.format_prompt()
        # The _pre_format_validation sets it to "Let's think step by step."
        assert "Let's think step by step." in prompt_text

    def test_chain_of_thought_empty_cot_instruction_override(self):
        """
        Even if user sets cot_instruction = '' in configure, we default to "Let's think step by step."
        during _pre_format_validation.
        """
        cot_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
            .configure(
                system="You are a reasoner.",
                user="How many planets are in the Solar System?",
                cot_instruction=""
            )
        )
        prompt_text = cot_prompt.format_prompt()
        assert "Let's think step by step." in prompt_text
        # The rest of the prompt
        assert "You are a reasoner." in prompt_text

    def test_chain_of_thought_all_fields_working(self):
        """
        Provide system, user, and a non-empty custom CoT. 
        Valid scenario => no errors, CoT is appended.
        """
        cot_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.CHAIN_OF_THOUGHT)
            .configure(
                system="System: You are an advanced reasoning model.",
                user="What is the next prime after 10?",
                cot_instruction="Deduce carefully step by step."
            )
        )
        prompt_text = cot_prompt.format_prompt()
        # Check final structure
        assert "You are an advanced reasoning model." in prompt_text
        assert "What is the next prime after 10?" in prompt_text
        assert "Deduce carefully step by step." in prompt_text
