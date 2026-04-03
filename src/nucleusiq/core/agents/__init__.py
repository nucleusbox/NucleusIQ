"""Agent framework for NucleusIQ.

Public API — what users import:
    Agent, AgentResult, ResultStatus, Task, AgentConfig
    ToolCallRecord, LLMCallRecord   (read-only result types)
    Attachment, AttachmentType       (file attachments)
    StreamEvent, StreamEventType     (streaming)

Advanced API — for building custom tracers/modes:
    ExecutionTracerProtocol          (implement your own tracer backend)
    BaseExecutionMode                (implement a custom execution mode)

Internal — used by the framework, not intended for end users:
    DefaultExecutionTracer, NoOpTracer, UsageTracker, build_*_record, etc.
    These are accessible via subpackage imports but excluded from __all__.
"""

from nucleusiq.agents.agent import Agent
from nucleusiq.agents.agent_result import (
    AgentResult,
    AutonomousDetail,
    LLMCallRecord,
    MemorySnapshot,
    PluginEvent,
    ResultStatus,
    ToolCallRecord,
    ValidationRecord,
)
from nucleusiq.agents.attachments import Attachment, AttachmentType
from nucleusiq.agents.builder.base_agent import BaseAgent
from nucleusiq.agents.chat_models import ChatMessage, LLMCallKwargs, ToolCallRequest
from nucleusiq.agents.observability import ExecutionTracerProtocol
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.plan import Plan, PlanStep
from nucleusiq.agents.react_agent import ReActAgent
from nucleusiq.agents.task import Task
from nucleusiq.streaming.events import StreamEvent, StreamEventType

__all__ = [
    # --- Core (every user needs these) ---
    "Agent",
    "AgentResult",
    "ResultStatus",
    "Task",
    # --- Result sub-models (read-only, from AgentResult fields) ---
    "ToolCallRecord",
    "LLMCallRecord",
    "PluginEvent",
    "MemorySnapshot",
    "AutonomousDetail",
    "ValidationRecord",
    # --- Attachments ---
    "Attachment",
    "AttachmentType",
    # --- Streaming ---
    "StreamEvent",
    "StreamEventType",
    # --- Messages ---
    "ChatMessage",
    # --- Planning ---
    "Plan",
    "PlanStep",
    # --- Advanced: extend the framework ---
    "ExecutionTracerProtocol",
    "BaseExecutionMode",
    "BaseAgent",
    "ReActAgent",
]
