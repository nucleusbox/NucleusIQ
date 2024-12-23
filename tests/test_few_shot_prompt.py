# tests/test_few_shot_prompt.py

import pytest
from nucleusiq.prompts.factory import PromptFactory, PromptTechnique


class TestFewShotPrompt:
    def test_few_shot_creation_success(self):
        """
        Scenario: We provide examples but no system. user is set.
                  We expect the final prompt to include examples + user.
        """
        examples = [
            {"input": "Translate 'Good morning' to Spanish.", "output": "Buenos días."},
            {"input": "Translate 'Thank you' to German.", "output": "Danke."},
        ]
        few_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="You are helpful assistent",
            user="Translate 'Good night' to Italian.",
            examples=examples
        )
        prompt_text = few_shot.format_prompt()
        expected_prompt = (
            "You are helpful assistent\n\n"
            "Input: Translate 'Good morning' to Spanish.\n"
            "Output: Buenos días.\n\n"
            "Input: Translate 'Thank you' to German.\n"
            "Output: Danke.\n\n"
            "Translate 'Good night' to Italian."
        )
        assert prompt_text.strip() == expected_prompt.strip()

    def test_few_shot_missing_examples(self):
        """
        'examples' is in input_variables, so it's required to be non-empty at format time.
        We provide system and user but skip examples => expect a ValueError.
        """
        with pytest.raises(ValueError) as exc_info:
            few_shot = (
                PromptFactory
                .create_prompt(technique=PromptTechnique.FEW_SHOT)
                .configure(
                    system="",
                    user="Translate 'Good night' to Italian."
                    # examples not provided => empty by default
                )
            )
            few_shot.format_prompt()
        # The error should mention 'examples' is missing or empty
        assert "Missing required field 'system' or it's empty." in str(exc_info.value)

    def test_few_shot_with_extra_fields(self):
        """
        Pass an unrecognized field to configure => expect ValueError.
        """
        examples = [
            {"input": "Translate 'Good morning' to Spanish.", "output": "Buenos días."},
            {"input": "Translate 'Thank you' to German.", "output": "Danke."},
        ]
        with pytest.raises(TypeError) as exc_info:
            few_shot = PromptFactory.create_prompt(
                technique=PromptTechnique.FEW_SHOT
            ).configure(
                system="Extra system instruction.",
                user="Translate 'Good night' to Italian.",
                extra_field="This should be ignored.",  # not recognized => error
                examples=examples
            )
            few_shot.format_prompt()
        assert "'extra_field'" in str(exc_info.value)

    def test_few_shot_with_empty_examples(self):
        """
        Provide an empty list of examples => we have 'examples' key, but it's empty.
        That is also a missing/empty required field => ValueError on format.
        """
        few_shot = (
            PromptFactory
            .create_prompt(technique=PromptTechnique.FEW_SHOT)
            .configure(
                system="You are Helpfull assistence",
                user="Translate 'Good night' to Italian.",
                # examples=[]
            )
        )
        with pytest.raises(ValueError) as exc_info:
            few_shot.format_prompt()
        assert "FewShotPrompt requires at least one example" in str(exc_info.value)

    def test_few_shot_minimal_with_system_user_and_examples(self):
        """
        Provide all required fields (system, user, examples) => success.
        """
        examples = [
            {"input": "Hello", "output": "Hola"}
        ]
        few_shot = (
            PromptFactory
            .create_prompt(technique=PromptTechnique.FEW_SHOT)
            .configure(
                system="You are a translation engine.",
                user="Translate words into Spanish.",
                examples=examples
            )
        )
        prompt_text = few_shot.format_prompt()
        # The final prompt should embed the examples with system & user.
        assert "You are a translation engine." in prompt_text
        assert "Input: Hello" in prompt_text
        assert "Output: Hola" in prompt_text
        assert "Translate words into Spanish." in prompt_text

    def test_few_shot_add_example_methods(self):
        """
        Test using .add_example() / .add_examples() after creation.
        """
        few_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        )
        # No examples yet => will fail if we call format_prompt() now
        # Configure system & user
        few_shot.configure(
            system="System instructions here.",
            user="Translate all to French.",
        )
        # Add examples
        few_shot.add_example("Good morning", "Bonjour")
        few_shot.add_examples([
            {"input": "Thank you", "output": "Merci"},
            {"input": "Goodbye", "output": "Au revoir"},
        ])
        # Now format => success
        prompt_text = few_shot.format_prompt()
        assert "System instructions here." in prompt_text
        assert "Input: Good morning" in prompt_text
        assert "Output: Bonjour" in prompt_text
        assert "Thank you" in prompt_text
        assert "Merci" in prompt_text

    def test_few_shot_missing_user(self):
        """
        'user' is a required field => if omitted or empty, raises ValueError on format_prompt.
        """
        examples = [
            {"input": "Hi", "output": "Hola"}
        ]
        few_shot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="You are a translator.",
            examples=examples
            # user not set
        )
        with pytest.raises(ValueError) as exc_info:
            few_shot.format_prompt()
        assert "Missing required field 'user' or it's empty." in str(exc_info.value)


class TestFewShotPromptWithCoT:
    def test_few_shot_cot_creation_success(self):
        """
        Provide use_cot=True, a custom cot_instruction,
        plus examples & user => final prompt includes CoT text at the end.
        """
        examples = [
            {
                "input": "Translate 'Good morning' to Spanish.",
                "output": "Buenos días."
            },
            {
                "input": "The odd numbers: 1,3,5 => sum is 9",
                "output": "True"
            },
        ]
        few_shot_cot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="System instructions.",
            user="Check these odd numbers sum.",
            use_cot=True,
            cot_instruction="Let's reason it step by step.",
            examples=examples
        )
        prompt_text = few_shot_cot.format_prompt()
        assert "Let's reason it step by step." in prompt_text
        # The final line should be "Check these odd numbers sum.\n\nLet's reason it step by step."

    def test_few_shot_cot_default_instruction(self):
        """
        If use_cot=True but no cot_instruction => default 'Let's think step by step.'
        """
        examples = [{"input": "Cat", "output": "Gato"}]
        few_shot_cot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="You are a translator with reasoning.",
            user="Translate to Spanish.",
            use_cot=True,
            examples=examples
        )
        prompt_text = few_shot_cot.format_prompt()
        assert "Let's think step by step." in prompt_text

    def test_few_shot_cot_disable_instruction(self):
        """
        If user sets use_cot=False => cot_instruction should not appear at all.
        """
        examples = [{"input": "Morning", "output": "Mañana"}]
        few_shot_cot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="System info.",
            user="Translate to Spanish.",
            use_cot=False,
            cot_instruction="Custom CoT (ignored)",
            examples=examples
        )
        prompt_text = few_shot_cot.format_prompt()
        # We do NOT expect "Custom CoT (ignored)" because use_cot=False
        assert "Custom CoT (ignored)" not in prompt_text

    def test_few_shot_cot_missing_examples(self):
        """
        'examples' is required => if missing/empty => ValueError on format_prompt
        even though CoT is enabled or disabled, doesn't matter.
        """
        few_shot_cot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="System info.",
            user="Translate something.",
            use_cot=True
            # no examples => empty
        )
        with pytest.raises(ValueError) as exc_info:
            few_shot_cot.format_prompt()
        assert "FewShotPrompt requires at least one example" in str(exc_info.value)

    def test_few_shot_cot_missing_user(self):
        """
        user is required. If it's missing => error at format time.
        """
        examples = [{"input": "Foo", "output": "Bar"}]
        few_shot_cot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="You are a reasoning translator",
            examples=examples,
            use_cot=True,
        )
        with pytest.raises(ValueError) as exc_info:
            few_shot_cot.format_prompt()
        assert "Missing required field 'user' or it's empty." in str(exc_info.value)

    def test_few_shot_cot_minimal_valid(self):
        """
        Provide system, user, examples => success with CoT enabled => uses default if empty.
        """
        examples = [{"input": "Hi", "output": "Hola"}]
        few_shot_cot = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="System instructions.",
            user="Please translate.",
            examples=examples,
            use_cot=True
        )
        prompt_text = few_shot_cot.format_prompt()
        # Should have default "Let's think step by step."
        assert "Let's think step by step." in prompt_text

    def test_few_shot_cot_disable_after_enable(self):
        """
        If we enable coT then disable => final prompt should not contain CoT instruction.
        """
        examples = [{"input": "One", "output": "Uno"}]
        fsc = PromptFactory.create_prompt(
            technique=PromptTechnique.FEW_SHOT
        ).configure(
            system="System text.",
            user="Translate to Spanish.",
            use_cot=True,
            cot_instruction="Some CoT text.",
            examples=examples
        )
        # Now disable CoT
        fsc.use_cot = False
        prompt_text = fsc.format_prompt()
        assert "Some CoT text." not in prompt_text
        # System text and examples remain
        assert "System text." in prompt_text
        assert "Input: One" in prompt_text
        assert "Output: Uno" in prompt_text

