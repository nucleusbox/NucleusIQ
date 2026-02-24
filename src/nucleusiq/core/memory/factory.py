# src/nucleusiq/memory/factory.py

from enum import Enum
from typing import Any, Dict, Type, Union

from nucleusiq.memory.base import BaseMemory
from nucleusiq.memory.full_history import FullHistoryMemory
from nucleusiq.memory.sliding_window import SlidingWindowMemory
from nucleusiq.memory.summary import SummaryMemory
from nucleusiq.memory.summary_window import SummaryWindowMemory
from nucleusiq.memory.token_budget import TokenBudgetMemory


class MemoryStrategy(Enum):
    """Built-in memory retention strategies."""

    FULL_HISTORY = "full_history"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUDGET = "token_budget"
    SUMMARY = "summary"
    SUMMARY_WINDOW = "summary_window"


_StrategyKey = Union[MemoryStrategy, str]


def _resolve_key(strategy: _StrategyKey) -> str:
    return strategy.value if isinstance(strategy, MemoryStrategy) else strategy


class MemoryFactory:
    """Factory to instantiate memory strategies by name.

    Built-in strategies use the ``MemoryStrategy`` enum.
    External providers register with a plain string key::

        MemoryFactory.register_memory("mem0", Mem0Memory)
        mem = MemoryFactory.create_memory("mem0", api_key="...")

    Built-in usage::

        mem = MemoryFactory.create_memory(MemoryStrategy.SLIDING_WINDOW, window_size=10)
    """

    _registry: Dict[str, Type[BaseMemory]] = {
        MemoryStrategy.FULL_HISTORY.value: FullHistoryMemory,
        MemoryStrategy.SLIDING_WINDOW.value: SlidingWindowMemory,
        MemoryStrategy.TOKEN_BUDGET.value: TokenBudgetMemory,
        MemoryStrategy.SUMMARY.value: SummaryMemory,
        MemoryStrategy.SUMMARY_WINDOW.value: SummaryWindowMemory,
    }

    @classmethod
    def register_memory(
        cls,
        strategy: _StrategyKey,
        memory_class: Type[BaseMemory],
    ) -> None:
        """Register a new memory strategy (built-in or external provider).

        Args:
            strategy: ``MemoryStrategy`` enum member **or** a plain string
                      for external providers (e.g. ``"mem0"``).
            memory_class: Concrete ``BaseMemory`` subclass.
        """
        key = _resolve_key(strategy)
        if key in cls._registry:
            raise ValueError(f"Memory strategy '{key}' is already registered.")
        cls._registry[key] = memory_class

    @classmethod
    def create_memory(
        cls,
        strategy: _StrategyKey,
        **kwargs: Any,
    ) -> BaseMemory:
        """Create an instance of the specified memory strategy.

        Args:
            strategy: ``MemoryStrategy`` enum member **or** a plain
                      string key for a registered provider.
            **kwargs: Forwarded to the strategy constructor
                      (e.g. ``window_size=10``, ``max_tokens=4096``).

        Returns:
            A configured ``BaseMemory`` subclass instance.

        Raises:
            ValueError: If the strategy is not registered.
        """
        key = _resolve_key(strategy)
        memory_class = cls._registry.get(key)
        if not memory_class:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Memory strategy '{key}' is not supported. "
                f"Available strategies: {available}."
            )
        return memory_class(**kwargs)
