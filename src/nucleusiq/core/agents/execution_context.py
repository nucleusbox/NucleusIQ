"""
ExecutionContext — narrow interface for what execution modes need.

Modes should depend on this protocol rather than the full ``Agent``
class.  This decouples modes from Agent internals and makes testing
easier (mock only what the mode actually uses).

``Agent`` satisfies this protocol structurally (no explicit subclass
needed).

Migration path:
    1. Protocol defined here (this file)           ← current
    2. New/refactored modes accept ExecutionContext ← next
    3. Old modes gradually migrate                  ← future
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from nucleusiq.agents.config.agent_config import AgentConfig, AgentState
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.memory.base import BaseMemory
from nucleusiq.prompts.base import BasePrompt


@runtime_checkable
class ExecutionContext(Protocol):
    """Narrow view of an Agent used by execution modes.

    Only lists the attributes / methods that modes actually access.
    Agent implements this structurally.
    """

    llm: BaseLLM | None
    tools: List[Any]
    memory: BaseMemory | None
    prompt: BasePrompt | None
    config: AgentConfig
    role: str
    objective: str
    state: AgentState
    response_format: Any | None

    @property
    def _logger(self) -> Any: ...

    @property
    def _executor(self) -> Any: ...

    @property
    def _current_llm_overrides(self) -> Dict[str, Any]: ...

    def _resolve_response_format(self) -> Any: ...

    def _get_structured_output_kwargs(self, output_config: Any) -> Dict[str, Any]: ...

    def _wrap_structured_output_result(
        self, response: Any, output_config: Any
    ) -> Any: ...
