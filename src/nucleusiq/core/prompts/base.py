# src/nucleusiq/prompts/base.py

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

FormatOutputType = TypeVar("FormatOutputType")


class BasePrompt(BaseModel, ABC):
    """
    Base class for all prompts in NucleusIQ. Allows partial config,
    final validation in `format_prompt()`.
    """

    template: str = Field(
        default="", description="Prompt template string with placeholders."
    )
    input_variables: List[str] = Field(
        default_factory=list, description="Names of required variables."
    )
    optional_variables: List[str] = Field(
        default_factory=list, description="Names of optional variables."
    )

    # We keep these optional so we can set them later
    system: str | None = Field(
        default=None, description="System instructions for the LLM prompt."
    )
    context: str | None = Field(
        default=None, description="Additional context or background information."
    )
    user: str | None = Field(default=None, description="User prompt or question.")

    partial_variables: Dict[str, Any | Callable[[], Any]] = Field(default_factory=dict)
    output_parser: Callable[[str], Any] | None = None
    metadata: Dict[str, Any] | None = None
    tags: List[str] | None = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @property
    @abstractmethod
    def technique_name(self) -> str:
        pass

    @abstractmethod
    def _construct_prompt(self, **kwargs) -> str:
        """
        Subclasses define the final string construction logic.
        """
        pass

    #
    # Basic field-level checks for certain list fields, etc.
    #
    @field_validator("template")
    def validate_template_not_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Template cannot be empty.")
        return v

    @field_validator("input_variables", "optional_variables", mode="before")
    def validate_str_list(cls, v, field):
        """Ensure these are lists of valid strings."""
        if not isinstance(v, list):
            raise ValueError(f"'{field.name}' must be a list of strings.")
        for var in v:
            if not isinstance(var, str) or not var.isidentifier():
                raise ValueError(f"Invalid variable name '{var}' in '{field.name}'.")
        return v

    #
    # The crucial step: We do final validation in `format_prompt()`.
    #
    def format_prompt(self, **kwargs) -> str:
        """
        Merge partial vars, check for required fields, call subclass hook, then build the prompt.
        """
        combined_vars = self._merge_partial_variables(**kwargs)

        # Merge class attributes for input/optional variables
        for var in self.input_variables + self.optional_variables:
            if hasattr(self, var):
                combined_vars[var] = getattr(self, var)

        # Base check: required input_variables must be present & non-empty
        for required_var in self.input_variables:
            val = combined_vars.get(required_var, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError(
                    f"Missing required field '{required_var}' or it's empty. "
                    f"{self.__class__.__name__} requires that field to be set and non-empty."
                )

        # Subclass can inject additional validations here
        self._pre_format_validation(combined_vars)

        # If all checks pass, do final construction
        return self._construct_prompt(**combined_vars)

    def _pre_format_validation(self, combined_vars: Dict[str, Any]) -> None:
        """
        Optional hook for subclass-specific validations.
        By default, does nothing.
        Subclasses can override to impose extra rules.
        """
        pass

    def configure(self, **kwargs) -> Self:
        """
        A flexible method to set multiple fields after instantiation.
        """
        for field_name, value in kwargs.items():
            if not hasattr(self, field_name):
                raise ValueError(
                    f"Field '{field_name}' is not recognized by {self.__class__.__name__}."
                )
            setattr(self, field_name, value)
        return self

    def save(self, file_path: Path | str) -> None:
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
    def load(cls: Type[Self], file_path: Path | str) -> Self:
        p = Path(file_path)
        if p.suffix == ".json":
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        elif p.suffix in (".yaml", ".yml"):
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

        prompt_type = data.pop("_type", None)
        if not prompt_type:
            raise ValueError("Prompt file missing '_type' field for class resolution.")

        from .factory import PromptFactory

        subcls = PromptFactory.prompt_classes.get(prompt_type.lower())
        if not subcls:
            available = ", ".join(PromptFactory.prompt_classes.keys())
            raise ValueError(
                f"Unsupported prompt type '{prompt_type}'. Available: {available}"
            )

        return subcls(**data)

    #
    # Additional convenience methods
    #
    def set_metadata(self, metadata: Dict[str, Any]) -> Self:
        self.metadata = metadata
        return self

    def add_tags(self, tags: List[str]) -> Self:
        self.tags.extend(tags)
        return self

    def set_output_parser(self, parser: Callable[[str], Any]) -> Self:
        self.output_parser = parser
        return self

    def _merge_partial_variables(self, **kwargs: Any) -> Dict[str, Any]:
        # Merges partial_variables with any provided kwargs
        partial_vals = {
            k: (v() if callable(v) else v) for k, v in self.partial_variables.items()
        }
        return {**partial_vals, **kwargs}
