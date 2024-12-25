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
        # use_cot=False,
        # cot_instruction="Please provide a detailed reasoning process."
    )

    # Format the prompt
    final_prompt = zero_shot.format_prompt()
    print("Zero-Shot Prompt:\n")
    print(final_prompt)

def test_extra_variable_error():
    zero_shot = PromptFactory.create_prompt(
                technique=PromptTechnique.ZERO_SHOT
            ).configure(
                system="You are a helpful assistant.",
                user="Translate the following English text to French: 'Hello, how are you?'",
                extra_field="This should be ignored."
            )
    print(zero_shot.format_prompt())

def test_cot_toggel():
    zero_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.ZERO_SHOT
        ).configure(
            system="System prompt.",
            user="User prompt.",
            use_cot=False,
            cot_instruction="Custom CoT instruction."
        )
    prompt_text = zero_shot.format_prompt()
        # expected_prompt = (
        #     "System prompt.\n\n"
        #     "User prompt."
        #     # 'cot_instruction' should not be included because use_cot=False
        # )
    print(prompt_text)

if __name__ == "__main__":
    # zero_shot_example()
    # test_extra_variable_error()
    test_cot_toggel()
