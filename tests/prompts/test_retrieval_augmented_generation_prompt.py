# tests/test_retrieval_augmented_generation_prompt.py

import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique

class TestRetrievalAugmentedGenerationPrompt:
    def test_rag_creation_success(self):
        """
        Provide system, non-empty context, user => success
        """
        rag_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
            .configure(
                system="You are an assistant with knowledge base access.",
                context="France is a country in Western Europe known for its rich history and culture.",
                user="What is the capital of France?"
            )
        )
        prompt_text = rag_prompt.format_prompt()
        expected_prompt = (
            "You are an assistant with knowledge base access.\n\n"
            "France is a country in Western Europe known for its rich history and culture.\n\n"
            "What is the capital of France?"
        )
        assert prompt_text.strip() == expected_prompt.strip()

    def test_rag_missing_context_field_entirely(self):
        """
        Omit context => raise ValueError about missing 'context'
        """
        rag_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
            .configure(
                system="System instructions.",
                # context is omitted
                user="Ask about France."
            )
        )
        with pytest.raises(ValueError) as exc_info:
            rag_prompt.format_prompt()
        assert "Missing required field 'context'" in str(exc_info.value)

    def test_rag_context_is_none(self):
        """
        Provide context=None => also disallowed => error
        """
        rag_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
            .configure(
                system="KB assistant.",
                context=None,
                user="What about capital?"
            )
        )
        with pytest.raises(ValueError) as exc_info:
            rag_prompt.format_prompt()
        assert "Missing required field 'context' or it's empty." in str(exc_info.value)

    def test_rag_with_empty_context_fails(self):
        """
        Provide an empty string for context => now we want it to fail, 
        because context is mandatory & non-empty.
        """
        rag_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
            .configure(
                system="System info.",
                context="",
                user="What is the capital of France?"
            )
        )
        with pytest.raises(ValueError) as exc_info:
            rag_prompt.format_prompt()
        # The base class error or our custom message could appear
        assert "Missing required field 'context' or it's empty." in str(exc_info.value)

    def test_rag_with_extra_fields(self):
        """
        Provide an unrecognized field => base .configure => raise error
        """
        with pytest.raises(TypeError) as exc:
            rag_prompt = (
                PromptFactory
                .create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
                .configure(
                    system="You are an assistant.",
                    context="Some knowledge snippet.",
                    user="Ask a question.",
                    extra_field="Should cause error."
                )
            )
            rag_prompt.format_prompt()
        assert "'extra_field'" in str(exc.value)

    def test_rag_empty_system_or_user_fields(self):
        """
        If system or user is empty => base class fails them as well
        """
        rag_prompt = (
            PromptFactory
            .create_prompt(PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION)
            .configure(
                system="",    # not allowed
                context="Some context.",
                user=""
            )
        )
        with pytest.raises(ValueError) as exc_info:
            rag_prompt.format_prompt()
        err_msg = str(exc_info.value)
        assert "Missing required field 'system' or it's empty." in err_msg or \
               "Missing required field 'user' or it's empty." in err_msg
