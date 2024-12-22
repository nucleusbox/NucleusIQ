# examples/zero_shot_example.py

from nucleusiq.prompts.factory import PromptFactory, PromptTechnique
from nucleusiq.prompts.zero_shot import ZeroShotPrompt


def zero_shot_example():
    # Create a ZeroShotPrompt instance using the factory
    zero_shot: ZeroShotPrompt = PromptFactory.create_prompt(PromptTechnique.ZERO_SHOT)

    # Configure the prompt
    zero_shot.configure(
        system="You are a knowledgeable assistant.",
        user="Explain the significance of the Turing Test.",
        use_cot=True,
        cot_instruction="Please provide a detailed reasoning process."
    )

    # Format the prompt
    final_prompt = zero_shot.format_prompt()
    print("Zero-Shot Prompt:\n")
    print(final_prompt)


if __name__ == "__main__":
    zero_shot_example()
