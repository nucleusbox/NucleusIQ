# src/examples/prompts/auto_chain_of_thought_examples.py

import os
import sys

# Add src directory to path so we can import nucleusiq
_src_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, _src_dir)

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.auto_chain_of_thought import AutoChainOfThoughtPrompt
from nucleusiq.llms.mock_llm import MockLLM

def auto_cot_example():
    auto_cot: AutoChainOfThoughtPrompt= PromptFactory.create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
    # Set multiple parameters in one go
    auto_cot.configure(
        system="You are an intelligent assistant.",
        context="",
        user="Describe the process of photosynthesis.",
        instruction="Let's reason step by step.",
        num_clusters=2,
        max_questions_per_cluster=1,
        cot_instruction="(Additional CoT text at the end.)"
    )
    # Provide an LLM
    auto_cot.llm = MockLLM()
    # Format the prompt
    prompt_text = auto_cot.format_prompt(
        task="Explain how photosynthesis works in detail.",
        questions=[
            "What is the main chemical equation for photosynthesis?",
            "How does sunlight factor in photosynthesis?",
             "What is the main sdjhdjs equation for photosynthesis?",
            "How does skdjk factor in photosynthesis?"
        ]
    )
    print(prompt_text)

def auto_cot_without_system_and_context():
    auto_cot = (
            PromptFactory
            .create_prompt(PromptTechnique.AUTO_CHAIN_OF_THOUGHT)
            .configure(
                llm=MockLLM(),
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
    print(prompt_text)
    # Check it includes the final chain for 2 Qs, no system, no context => leading blank lines
    # The mock LLM returns "This is a mock reasoning chain." => verify
    assert "Solve the following questions" in prompt_text
    assert "Q1" in prompt_text
    assert "Q2" in prompt_text
    assert "Step by step." in prompt_text

if __name__ == "__main__":
    auto_cot_example()
    auto_cot_without_system_and_context()
