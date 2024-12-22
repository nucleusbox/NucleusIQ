# src/nucleusiq/prompts/base.py

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Self

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

# Define a TypeVar for format output type
FormatOutputType = TypeVar("FormatOutputType")


class BasePrompt(BaseModel, ABC):
    """
    Abstract base class for all prompt templates in NucleusIQ.
    Provides a unified interface for constructing, validating, serializing,
    and deserializing prompts.
    """

    # ----------------------------
    # Common fields for all prompts
    # ----------------------------
    template: str = Field(
        default="",
        description="Prompt template string with placeholders."
    )
    input_variables: List[str] = Field(
        default_factory=list,
        description="Names of required variables."
    )
    optional_variables: List[str] = Field(
        default_factory=list,
        description="Names of optional variables."
    )

    system: Optional[str] = Field(
        default=None,
        description="System instructions for the LLM prompt."
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context or background information."
    )
    user: Optional[str] = Field(
        default=None,
        description="User prompt or question."
    )

    partial_variables: Dict[str, Union[Any, Callable[[], Any]]] = Field(
        default_factory=dict,
        description="Partial variables that can be merged into final prompt."
    )
    output_parser: Optional[Callable[[str], Any]] = Field(
        default=None,
        description="Parser to process the LLM's raw output."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata for logging/tracing."
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="Tags for categorizing prompts."
    )

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    # -------------------------------
    # Abstract properties and methods
    # -------------------------------
    @property
    @abstractmethod
    def technique_name(self) -> str:
        """
        Returns the name of the prompting technique.
        Must be implemented by each subclass.
        """
        pass

    @abstractmethod
    def _construct_prompt(self, **kwargs) -> str:
        """
        Subclasses implement how the final prompt is constructed from placeholders.
        """
        pass

    # ----------------------------
    # Field Validators
    # ----------------------------
    @field_validator('template')
    def validate_template_not_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Template cannot be empty.")
        return v

    @field_validator('input_variables', mode='before')
    def validate_input_variables(cls, v):
        if not isinstance(v, list):
            raise ValueError("input_variables must be a list of strings.")
        for var in v:
            if not isinstance(var, str) or not var.isidentifier():
                raise ValueError(f"Invalid input variable name: '{var}'. Must be a valid identifier.")
        return v

    @field_validator('optional_variables', mode='before')
    def validate_optional_variables(cls, v):
        if not isinstance(v, list):
            raise ValueError("optional_variables must be a list of strings.")
        for var in v:
            if not isinstance(var, str) or not var.isidentifier():
                raise ValueError(f"Invalid optional variable name: '{var}'. Must be a valid identifier.")
        return v

    @field_validator('tags', mode='before')
    def validate_tags(cls, v):
        if not isinstance(v, list):
            raise ValueError("tags must be a list of strings.")
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError("Each tag must be a string.")
        return v

    @field_validator('output_parser')
    def validate_output_parser(cls, v):
        if v is not None and not callable(v):
            raise ValueError("output_parser must be callable.")
        return v

    # -------------------------
    # Prompt Construction Logic
    # -------------------------
    def format_prompt(self, **kwargs) -> str:
        """
        Format the prompt with user-provided variables + partial variables,
        plus any field in input_variables/optional_variables that is set on `self`.
        """
        combined_vars = self._merge_partial_variables(**kwargs)

        # Gather fields from self that are in input/optional vars
        for var in self.input_variables + self.optional_variables:
            if hasattr(self, var):
                combined_vars[var] = getattr(self, var)

        # Check for missing required variables
        missing_vars = set(self.input_variables) - set(combined_vars.keys())
        if missing_vars:
            raise ValueError(
                f"Missing variables for prompt: {missing_vars}. "
                f"Received variables: {list(combined_vars.keys())}"
            )

        return self._construct_prompt(**combined_vars)

    # ------------------
    # Configuration Logic
    # ------------------
    def configure(self, **kwargs) -> Self:
        """
        Universal method to set multiple fields at once.
        Each subclass can override or extend if needed.

        Args:
            **kwargs: Field names and their values to configure.

        Returns:
            Self: The instance with updated fields.
        """
        for field_name, value in kwargs.items():
            if not hasattr(self, field_name):
                raise ValueError(f"Field '{field_name}' is not recognized by {self.__class__.__name__}")
            setattr(self, field_name, value)
        return self

    # -------------------------
    # Serialization Logic
    # -------------------------
    def save(self, file_path: Union[Path, str]) -> None:
        """
        Save prompt data to .json or .yaml file.
        """
        data = self.model_dump()
        data["_type"] = self.technique_name

        p = Path(file_path)
        if p.suffix == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        elif p.suffix in (".yaml", ".yml"):
            with open(p, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

    @classmethod
    def load(cls: Type[BasePrompt], file_path: Union[Path, str]) -> BasePrompt:
        """
        Load prompt data from .json or .yaml and instantiate the correct subclass.
        """
        p = Path(file_path)
        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif p.suffix in (".yaml", ".yml"):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

        prompt_type = data.pop("_type", None)
        if not prompt_type:
            raise ValueError("Prompt file missing '_type' field for class resolution.")

        # Use the PromptFactory for dynamic class instantiation
        from .factory import PromptFactory
        subcls = PromptFactory.prompt_classes.get(prompt_type.lower())
        if not subcls:
            available = ", ".join(PromptFactory.prompt_classes.keys())
            raise ValueError(f"Unsupported prompt type '{prompt_type}'. Available techniques: {available}")

        return subcls(**data)

    # -------------------
    # Misc. Base Logic
    # -------------------
    def set_metadata(self, metadata: Dict[str, Any]) -> Self:
        """
        Assign metadata in one shot.

        Args:
            metadata (Dict[str, Any]): Metadata information.

        Returns:
            Self: The updated prompt instance.
        """
        self.metadata = metadata
        return self

    def add_tags(self, tags: List[str]) -> Self:
        """
        Extend the list of tags with new ones.

        Args:
            tags (List[str]): List of tags to add.

        Returns:
            Self: The updated prompt instance.
        """
        self.tags.extend(tags)
        return self

    def set_output_parser(self, parser: Callable[[str], Any]) -> Self:
        """
        Assign a parsing function for LLM outputs.

        Args:
            parser (Callable[[str], Any]): A parser function.

        Returns:
            Self: The updated prompt instance.
        """
        self.output_parser = parser
        return self

    def _merge_partial_variables(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Combine partial variables with user-provided data.
        """
        partial_vals = {
            k: (v() if callable(v) else v) for k, v in self.partial_variables.items()
        }
        return {**partial_vals, **kwargs}
