from typing import Type, Dict, TypeVar, cast
from enum import Enum
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq.prompts.few_shot import FewShotPrompt
from nucleusiq.prompts.chain_of_thought import ChainOfThoughtPrompt
from nucleusiq.prompts.auto_chain_of_thought import AutoChainOfThoughtPrompt
from nucleusiq.prompts.retrieval_augmented_generation import RetrievalAugmentedGenerationPrompt
from nucleusiq.prompts.prompt_composer import PromptComposer

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

class PromptFactory:
    """
    Factory class to instantiate different prompting techniques.
    """

    prompt_classes: Dict[PromptTechnique, Type[BasePrompt]] = {
        PromptTechnique.ZERO_SHOT: ZeroShotPrompt,
        PromptTechnique.FEW_SHOT: FewShotPrompt,
        PromptTechnique.CHAIN_OF_THOUGHT: ChainOfThoughtPrompt,
        PromptTechnique.AUTO_CHAIN_OF_THOUGHT: AutoChainOfThoughtPrompt,
        PromptTechnique.RETRIEVAL_AUGMENTED_GENERATION: RetrievalAugmentedGenerationPrompt,
        PromptTechnique.PROMPT_COMPOSER: PromptComposer,
    }

    @classmethod
    def register_prompt(cls, technique: PromptTechnique, prompt_class: Type[BasePrompt]) -> None:
        """
        Register a new prompting technique.

        Args:
            technique (PromptTechnique): The name of the prompting technique.
            prompt_class (Type[BasePrompt]): The class implementing the technique.
        """
        if technique in cls.prompt_classes:
            raise ValueError(f"Prompting technique '{technique.value}' is already registered.")
        cls.prompt_classes[technique] = prompt_class

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
        prompt_class = cls.prompt_classes.get(technique)
        if not prompt_class:
            available = ", ".join(t.value for t in cls.prompt_classes.keys())
            raise ValueError(
                f"Prompting technique '{technique.value}' is not supported. "
                f"Available techniques: {available}."
            )
        return cast(T, prompt_class())
