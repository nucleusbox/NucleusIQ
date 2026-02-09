# src/nucleusiq/prompts/factory.py

from typing import Type, Dict, TypeVar, cast
from enum import Enum
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq.prompts.few_shot import FewShotPrompt
from nucleusiq.prompts.chain_of_thought import ChainOfThoughtPrompt
from nucleusiq.prompts.auto_chain_of_thought import AutoChainOfThoughtPrompt
from nucleusiq.prompts.retrieval_augmented_generation import RetrievalAugmentedGenerationPrompt
from nucleusiq.prompts.prompt_composer import PromptComposer
from nucleusiq.prompts.meta_prompt import MetaPrompt

# Define a TypeVar for BasePrompt
T = TypeVar('T', bound=BasePrompt)

# Define an Enum for prompt techniques
class PromptTechnique(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    AUTO_CHAIN_OF_THOUGHT = "auto_chain_of_thought"
    RETRIEVAL_AUGMENTED_GENERATION = "retrieval_augmented_generation"
    PROMPT_COMPOSER = "prompt_composer"
    META_PROMPTING = "meta_prompting"

class PromptFactory:
    """
    Factory class to instantiate different prompting techniques.
    """

    prompt_classes: Dict[str, Type[BasePrompt]] = {
        PromptTechnique.ZERO_SHOT.value: ZeroShotPrompt,
        PromptTechnique.FEW_SHOT.value: FewShotPrompt,
        PromptTechnique.CHAIN_OF_THOUGHT.value: ChainOfThoughtPrompt,
        PromptTechnique.AUTO_CHAIN_OF_THOUGHT.value: AutoChainOfThoughtPrompt,
        PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION.value: RetrievalAugmentedGenerationPrompt,
        PromptTechnique.PROMPT_COMPOSER.value: PromptComposer,
        PromptTechnique.META_PROMPTING.value: MetaPrompt
    }

    @classmethod
    def register_prompt(cls, technique: PromptTechnique, prompt_class: Type[BasePrompt]) -> None:
        """
        Register a new prompting technique.

        Args:
            technique (PromptTechnique): The name of the prompting technique.
            prompt_class (Type[BasePrompt]): The class implementing the technique.
        """
        if technique.value in cls.prompt_classes:
            raise ValueError(f"Prompting technique '{technique.value}' is already registered.")
        cls.prompt_classes[technique.value] = prompt_class

    @classmethod
    def create_prompt(cls, technique: PromptTechnique) -> T:
        """
        Creates an instance of the specified prompting technique.

        Args:
            technique (PromptTechnique): The prompting technique to use.

        Returns:
            T: An instance of the specified prompt (specific subclass of BasePrompt).

        Raises:
            ValueError: If the technique is not supported.
        """
        prompt_class = cls.prompt_classes.get(technique.value)
        if not prompt_class:
            available = ", ".join(cls.prompt_classes.keys())
            raise ValueError(
                f"Prompting technique '{technique.value}' is not supported. "
                f"Available techniques: {available}."
            )
        return cast(T, prompt_class())
