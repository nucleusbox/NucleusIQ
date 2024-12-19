# src/nucleusiq/prompts/factory.py

from typing import Type, Dict, Any, Optional, List
from nucleusiq.prompts.base import BasePrompt
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq.prompts.few_shot import FewShotPrompt
from nucleusiq.prompts.chain_of_thought import ChainOfThoughtPrompt
from nucleusiq.prompts.auto_chain_of_thought import AutoChainOfThoughtPrompt
from nucleusiq.prompts.retrieval_augmented_generation import RetrievalAugmentedGenerationPrompt
from nucleusiq.llms.base_llm import BaseLLM


class PromptFactory:
    """
    Factory class to instantiate different prompting techniques.
    """

    prompt_classes: Dict[str, Type[BasePrompt]] = {
        "zero_shot": ZeroShotPrompt,
        "few_shot": FewShotPrompt,
        "chain_of_thought": ChainOfThoughtPrompt,
        "auto_chain_of_thought": AutoChainOfThoughtPrompt,
        "retrieval_augmented_generation": RetrievalAugmentedGenerationPrompt,
    }

    @classmethod
    def register_prompt(cls, technique: str, prompt_class: Type[BasePrompt]) -> None:
        """
        Register a new prompting technique.

        Args:
            technique (str): The name of the prompting technique.
            prompt_class (Type[BasePrompt]): The class implementing the technique.
        """
        technique = technique.lower()
        if technique in cls.prompt_classes:
            raise ValueError(f"Prompting technique '{technique}' is already registered.")
        cls.prompt_classes[technique] = prompt_class

    @classmethod
    def create_prompt(
        cls,
        technique: str,
        llm: Optional[BaseLLM] = None,
        template: Optional[str] = None,
        input_variables: Optional[List[str]] = None,
        optional_variables: Optional[List[str]] = None,
        partial_variables: Optional[Dict[str, Any]] = None,
        output_parser: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,  # Includes 'examples' and other params
    ) -> BasePrompt:
        """
        Creates an instance of the specified prompting technique.

        Args:
            technique (str): The prompting technique to use.
            llm (Optional[BaseLLM]): Language Model adapter to use.
            template (Optional[str]): The prompt template string.
            input_variables (Optional[List[str]]): Required variables.
            optional_variables (Optional[List[str]]): Optional variables.
            partial_variables (Optional[Dict[str, Any]]): Pre-filled variables.
            output_parser (Optional[Any]): Output parser.
            metadata (Optional[Dict[str, Any]]): Metadata for the prompt.
            tags (Optional[List[str]]): Tags for the prompt.
            **kwargs (Any): Additional variables required by the prompt.

        Returns:
            BasePrompt: An instance of the specified prompt.

        Raises:
            ValueError: If the technique is not supported.
        """
        technique = technique.lower()
        prompt_class = cls.prompt_classes.get(technique)
        if not prompt_class:
            available = ", ".join(cls.prompt_classes.keys())
            raise ValueError(
                f"Prompting technique '{technique}' is not supported. "
                f"Available techniques: {available}."
            )
        try:
            prompt = prompt_class(
                template=template or prompt_class.default_template(),
                input_variables=input_variables or prompt_class.default_input_variables(),
                optional_variables=optional_variables or prompt_class.default_optional_variables(),
                partial_variables=partial_variables or {},
                output_parser=output_parser,
                metadata=metadata,
                tags=tags,
                llm=llm,  # Pass the LLM adapter
                **kwargs,  # Pass additional kwargs like 'examples'
            )
            return prompt
        except TypeError as e:
            raise ValueError(f"Error initializing prompt: {e}")
