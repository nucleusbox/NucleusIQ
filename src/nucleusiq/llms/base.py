"""Base language models class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from functools import cache 

class BaseLanguageModel():
    """
    Base class for all language models in NucleusIQ.

    This class provides a common interface and shared functionality for all language models.
    It is not intended to be instantiated directly.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @cache
    def get_model_name(self) -> str:
        """Returns the name of the model."""
        return self.model_name
