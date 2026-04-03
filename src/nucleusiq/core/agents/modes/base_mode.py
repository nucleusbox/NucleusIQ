"""
Base execution mode interface for NucleusIQ agents.

Each mode implements a distinct execution strategy:
- DirectMode: Fast, simple, no tools (Gear 1)
- StandardMode: Tool-enabled, linear execution (Gear 2)
- AutonomousMode: Orchestration + Critic/Refiner verification (Gear 3)

New modes can be registered via ``Agent.register_mode()`` without
modifying the Agent class (Open/Closed Principle).
"""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent

from nucleusiq.agents.attachments import Attachment
from nucleusiq.agents.chat_models import (
    ChatMessage,
    ToolCallRequest,
    messages_to_dicts,
)
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.messaging.message_builder import MessageBuilder
from nucleusiq.agents.observability import (
    build_llm_call_record,
    build_llm_call_record_from_stream,
    build_tool_call_record,
)
from nucleusiq.agents.task import Task
from nucleusiq.agents.usage.usage_tracker import CallPurpose
from nucleusiq.plugins.base import ModelRequest, ToolRequest
from nucleusiq.streaming.events import StreamEvent, StreamEventType


def build_attachment_prefix(attachments: list[Attachment] | None) -> str:
    """Build a human-readable prefix summarising attached files.

    Used when storing user messages in memory so the LLM sees file
    context even after the raw attachment data is discarded.

    Returns ``""`` when *attachments* is ``None`` or empty.

    Example output::

        [Attached: report.pdf (text, 31.3 KB), chart.png (image_url)]
    """
    if not attachments:
        return ""
    parts: list[str] = []
    for att in attachments:
        label = att.name or "(unnamed)"
        kind = att.type.value
        size = len(att.data) if isinstance(att.data, (bytes, str)) else 0
        if size > 0:
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            parts.append(f"{label} ({kind}, {size_str})")
        else:
            parts.append(f"{label} ({kind})")
    return f"[Attached: {', '.join(parts)}]"


def build_attachment_metadata(
    attachments: list[Attachment] | None,
) -> dict[str, Any] | None:
    """Build lightweight metadata dict for memory storage.

    Returns ``None`` when *attachments* is ``None`` or empty.
    Only stores name, type, and size — never the raw file data.

    Return format::

        {"attachments": [{"name": "report.pdf", "type": "pdf", "size": 32000}, ...]}
    """
    if not attachments:
        return None
    entries: list[dict[str, Any]] = []
    for att in attachments:
        size = len(att.data) if isinstance(att.data, (bytes, str)) else 0
        entry: dict[str, Any] = {"type": att.type.value, "size": size}
        if att.name:
            entry["name"] = att.name
        entries.append(entry)
    return {"attachments": entries}


class BaseExecutionMode(ABC):
    """Strategy interface for agent execution modes.

    Every mode receives the ``agent`` instance so it can access
    ``agent.llm``, ``agent.tools``, ``agent.config``, ``agent.memory``,
    ``agent._executor``, ``agent._logger``, and helper methods like
    ``agent._resolve_response_format()``.

    The mode does **not** own state — the Agent does.

    **Streaming contract:**

    * ``run()`` — non-streaming (returns result)
    * ``run_stream()`` — streaming (yields ``StreamEvent``).
      Default fallback calls ``run()`` and emits a single ``COMPLETE``
      event, so custom modes work without streaming support.

    Shared helpers (``call_llm_stream``, ``_streaming_tool_call_loop``)
    live here so that concrete modes stay DRY.
    """

    @abstractmethod
    async def run(
        self,
        agent: "Agent",
        task: Task,
    ) -> Any:
        """Execute a task using this mode's strategy."""
        ...

    # ------------------------------------------------------------------ #
    # Streaming: public interface                                         #
    # ------------------------------------------------------------------ #

    async def run_stream(
        self,
        agent: "Agent",
        task: Task,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream execution as ``StreamEvent`` objects.

        **Default implementation** — falls back to ``run()`` and yields
        a single ``COMPLETE`` event.  Concrete modes override this for
        real token-by-token streaming.

        Liskov: any mode can be used with ``execute_stream()`` without
        the caller knowing whether real streaming is supported.
        """
        result = await self.run(agent, task)
        text = str(result) if result is not None else ""
        yield StreamEvent.complete_event(text)

    # ------------------------------------------------------------------ #
    # Shared helpers (used by DirectMode, StandardMode, etc.)            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_objective(task: Task | Dict[str, Any]) -> str:
        """Extract the objective string from a Task or dict.

        Accepts both forms for backward compatibility with external callers,
        but internal callers should always pass a ``Task`` instance.
        """
        if isinstance(task, Task):
            return task.objective
        return task.get("objective", "")

    def echo_fallback(self, agent: "Agent", task: Task | Dict[str, Any]) -> str | None:
        """Return an echo result when no LLM is configured, or ``None``."""
        if agent.llm:
            return None
        agent._logger.warning("No LLM configured, falling back to echo mode")
        agent.state = AgentState.COMPLETED
        objective = self.get_objective(task)
        return f"Echo: {objective}"

    def build_messages(
        self,
        agent: "Agent",
        task: Task | Dict[str, Any],
        plan: Any = None,
    ) -> List[ChatMessage]:
        """Convert task (and optional plan) into an LLM-ready message list.

        When agent has memory, prior conversation turns are injected
        between the system message and the current user message so the
        LLM has full conversational context.

        If the agent's LLM provides ``process_attachments()``, it is
        passed to ``MessageBuilder`` so the provider can produce
        API-native file content parts instead of framework-level
        text extraction.
        """
        processor = None
        if agent.llm and hasattr(agent.llm, "process_attachments"):
            processor = agent.llm.process_attachments

        messages = MessageBuilder.build(
            task,
            plan,
            prompt=agent.prompt,
            role=agent.role,
            objective=agent.objective,
            logger=agent._logger,
            attachment_processor=processor,
        )

        if agent.memory:
            memory_ctx = agent.memory.get_context()
            if memory_ctx:
                task_dict = task.to_dict() if isinstance(task, Task) else task
                current_objective = task_dict.get("objective", "")
                filtered = [
                    m
                    for m in memory_ctx
                    if not (
                        m["role"] == "user"
                        and m["content"] == current_objective
                        and m is memory_ctx[-1]
                    )
                ]
                if filtered:
                    insert_idx = 0
                    for i, m in enumerate(messages):
                        if m.role == "system":
                            insert_idx = i + 1
                        else:
                            break
                    for j, mem_msg in enumerate(filtered):
                        messages.insert(
                            insert_idx + j,
                            ChatMessage.from_dict(mem_msg),
                        )

        return messages

    def build_call_kwargs(
        self,
        agent: "Agent",
        messages: List[ChatMessage],
        tool_specs: List[Dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Build the kwargs dict for ``agent.llm.call()``.

        Merges model name, messages, tool specs, max_output_tokens,
        per-execute LLM overrides, and structured-output kwargs.
        """
        output_config = agent._resolve_response_format()
        call_kwargs: Dict[str, Any] = {
            "model": getattr(agent.llm, "model_name", "default"),
            "messages": messages_to_dicts(messages),
            "tools": tool_specs if tool_specs else None,
            "max_output_tokens": max_output_tokens
            or getattr(agent.config, "llm_max_output_tokens", 1024),
        }
        call_kwargs.update(getattr(agent, "_current_llm_overrides", {}))
        call_kwargs.update(agent._get_structured_output_kwargs(output_config))
        return call_kwargs

    @staticmethod
    def validate_response(response: Any) -> None:
        """Raise ``LLMError`` if the LLM response is empty/malformed."""
        if not response or not hasattr(response, "choices") or not response.choices:
            from nucleusiq.llms.errors import LLMError

            raise LLMError("LLM returned empty response")

    @staticmethod
    def extract_content(msg: Any) -> str | None:
        """Extract and normalise text content from an LLM message.

        Handles:
        - Plain string content
        - List-of-parts format ``[{"type": "text", "text": "..."}]``
        - ``None``
        """
        if isinstance(msg, dict):
            raw = msg.get("content")
        else:
            raw = getattr(msg, "content", None)

        if isinstance(raw, str) and raw.strip():
            return raw
        if isinstance(raw, list):
            parts: List[str] = []
            for part in raw:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
            return "\n".join(parts) if parts else None
        return None

    def handle_structured_output(self, agent: "Agent", response: Any) -> Any | None:
        """Return the wrapped structured-output result, or ``None``.

        When a structured-output result is detected the agent state is
        set to COMPLETED.
        """
        output_config = agent._resolve_response_format()
        wrapped = agent._wrap_structured_output_result(response, output_config)
        if isinstance(wrapped, dict) and "output" in wrapped:
            agent.state = AgentState.COMPLETED
            return wrapped
        return None

    # ------------------------------------------------------------------ #
    # File-aware memory helpers                                          #
    # ------------------------------------------------------------------ #

    async def store_task_in_memory(
        self,
        agent: "Agent",
        task: Task,
    ) -> None:
        """Persist the user's task objective (with attachment context) in memory."""
        if not agent.memory:
            return

        content = task.objective
        metadata: dict[str, Any] = {}

        if task.attachments:
            prefix = build_attachment_prefix(task.attachments)
            if prefix:
                content = f"{prefix}\n{content}"
            meta = build_attachment_metadata(task.attachments)
            if meta:
                metadata.update(meta)

        try:
            kwargs: dict[str, Any] = {}
            if metadata:
                kwargs["metadata"] = metadata
            await agent.memory.aadd_message("user", content, **kwargs)
        except Exception as e:
            agent._logger.warning("Failed to store task in memory: %s", e)
            tracer = getattr(agent, "_tracer", None)
            if tracer is not None:
                tracer.record_warning(f"Failed to store task in memory: {e}")

    # ------------------------------------------------------------------ #
    # Plugin-aware LLM and Tool invocation                               #
    # ------------------------------------------------------------------ #

    async def call_llm(
        self,
        agent: "Agent",
        call_kwargs: Dict[str, Any],
        messages: List[ChatMessage] | None = None,
        tool_specs: List[Dict[str, Any]] | None = None,
        *,
        purpose: CallPurpose = CallPurpose.MAIN,
    ) -> Any:
        """Invoke ``agent.llm.call()`` with the full plugin pipeline.

        Constructs a ``ModelRequest`` and runs:
        before_model -> wrap_model_call chain -> after_model.
        Falls back to a direct call when no plugins are registered.

        After the call, usage is recorded in ``agent._usage_tracker``.
        """
        assert agent.llm is not None, "agent.llm must be set before calling call_llm"
        pm = getattr(agent, "_plugin_manager", None)

        t0 = time.perf_counter()
        if pm is None or not pm.has_plugins():
            response = await agent.llm.call(**call_kwargs)
        else:
            reserved = {"model", "messages", "tools", "max_output_tokens"}
            extra = {k: v for k, v in call_kwargs.items() if k not in reserved}

            request = ModelRequest(
                model=call_kwargs.get("model", "default"),
                messages=messages or [],
                tools=tool_specs,
                max_output_tokens=call_kwargs.get("max_output_tokens", 1024),
                call_count=pm.increment_model_calls(),
                agent_name=agent.name,
                extra_kwargs=extra,
            )

            request = await pm.run_before_model(request)
            response = await pm.execute_model_call(request, agent.llm.call)
            response = await pm.run_after_model(request, response)

        duration_ms = (time.perf_counter() - t0) * 1000

        tracker = getattr(agent, "_usage_tracker", None)
        if tracker is not None:
            tracker.record_from_response(purpose, response)

        tracer = getattr(agent, "_tracer", None)
        if tracer is not None:
            model = call_kwargs.get("model") or getattr(response, "model", None)
            tracer.record_llm_call(
                build_llm_call_record(
                    response,
                    call_round=len(tracer.llm_calls) + 1,
                    purpose=purpose.value,
                    duration_ms=duration_ms,
                    model=model,
                )
            )

        return response

    async def call_tool(
        self,
        agent: "Agent",
        tc: ToolCallRequest,
        *,
        tool_round: int = 1,
    ) -> Any:
        """Invoke tool execution with the full plugin pipeline.

        Constructs a ``ToolRequest`` and runs the wrap_tool_call chain.
        Falls back to a direct call when no plugins are registered.
        """
        assert agent._executor is not None, "agent._executor must be set before calling call_tool"
        pm = getattr(agent, "_plugin_manager", None)

        tool_args: Dict[str, Any] = {}
        try:
            tool_args = json.loads(tc.arguments) if tc.arguments else {}
        except (json.JSONDecodeError, TypeError):
            pass

        t0 = time.perf_counter()
        try:
            if pm is None or not pm.has_plugins():
                result = await agent._executor.execute(tc)
            else:
                request = ToolRequest(
                    tool_name=tc.name or "",
                    tool_args=tool_args,
                    tool_call_id=tc.id,
                    call_count=pm.increment_tool_calls(),
                    agent_name=agent.name,
                )
                request._tool_call_request = tc
                result = await pm.execute_tool_call(request, agent._executor.execute)

            duration_ms = (time.perf_counter() - t0) * 1000
            tracer = getattr(agent, "_tracer", None)
            if tracer is not None:
                tracer.record_tool_call(
                    build_tool_call_record(
                        tc,
                        result=result,
                        success=True,
                        duration_ms=duration_ms,
                        round=tool_round,
                        args=tool_args,
                    )
                )
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            tracer = getattr(agent, "_tracer", None)
            if tracer is not None:
                tracer.record_tool_call(
                    build_tool_call_record(
                        tc,
                        result=None,
                        success=False,
                        error=str(e),
                        error_type=type(e).__name__,
                        duration_ms=duration_ms,
                        round=tool_round,
                        args=tool_args,
                    )
                )
            raise

    # ------------------------------------------------------------------ #
    # Streaming helpers (shared by all modes)                             #
    # ------------------------------------------------------------------ #

    async def call_llm_stream(
        self,
        agent: "Agent",
        call_kwargs: Dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Invoke ``agent.llm.call_stream()`` yielding ``StreamEvent`` objects.

        Analogous to ``call_llm()`` but for the streaming path.
        Plugin-aware streaming is deferred to a future release.
        """
        assert agent.llm is not None, "agent.llm must be set before calling call_llm_stream"
        async for event in agent.llm.call_stream(**call_kwargs):
            yield event

    async def _streaming_tool_call_loop(
        self,
        agent: "Agent",
        messages: List[ChatMessage],
        tool_specs: List[Dict[str, Any]] | None,
        *,
        max_tool_calls: int = 30,
        max_output_tokens: int = 2048,
        purpose: CallPurpose = CallPurpose.MAIN,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Reusable streaming LLM ↔ tool loop.

        Used by Direct, Standard, and Autonomous modes.  Yields
        orchestration events so consumers see a structured stream::

            LLM_CALL_START → TOKEN... → LLM_CALL_END
              → TOOL_CALL_START → TOOL_CALL_END → (loop)
            ...
            → COMPLETE (or ERROR)

        Updates *messages* in-place with assistant and tool messages.
        The first call is tagged with *purpose*; subsequent calls after
        tool results are tagged as ``TOOL_LOOP``.
        """
        tool_call_count = 0
        call_round = 0
        empty_retries = 1
        tracker = getattr(agent, "_usage_tracker", None)

        while tool_call_count < max_tool_calls:
            call_round += 1
            current_purpose = purpose if call_round == 1 else CallPurpose.TOOL_LOOP
            yield StreamEvent.llm_start_event(call_round)

            call_kwargs = self.build_call_kwargs(
                agent, messages, tool_specs, max_output_tokens=max_output_tokens
            )

            complete_event: StreamEvent | None = None
            errored = False

            stream_t0 = time.perf_counter()
            async for event in self.call_llm_stream(agent, call_kwargs):
                if event.type == StreamEventType.TOKEN:
                    yield event
                elif event.type == StreamEventType.COMPLETE:
                    complete_event = event
                elif event.type == StreamEventType.ERROR:
                    yield StreamEvent.llm_end_event(call_round)
                    yield event
                    errored = True
                    break

            if errored:
                return

            stream_duration_ms = (time.perf_counter() - stream_t0) * 1000

            yield StreamEvent.llm_end_event(call_round)

            if complete_event is None:
                yield StreamEvent.error_event("LLM stream produced no COMPLETE event")
                return

            if tracker is not None:
                tracker.record_from_stream_metadata(
                    current_purpose,
                    complete_event.metadata,
                    call_round=call_round,
                )

            tracer = getattr(agent, "_tracer", None)
            if tracer is not None:
                tracer.record_llm_call(
                    build_llm_call_record_from_stream(
                        complete_event.metadata,
                        call_round=call_round,
                        purpose=current_purpose.value,
                        duration_ms=stream_duration_ms,
                        model=call_kwargs.get("model"),
                    )
                )

            full_content = complete_event.content or ""
            raw_tool_calls = (complete_event.metadata or {}).get("tool_calls", [])

            # --- Tool calls detected → execute and loop ---
            if raw_tool_calls:
                parsed_calls = [ToolCallRequest.from_raw(tc) for tc in raw_tool_calls]
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=full_content or None,
                        tool_calls=parsed_calls,
                    )
                )

                for tc in parsed_calls:
                    if not tc.name:
                        continue
                    if tool_call_count >= max_tool_calls:
                        agent._logger.warning(
                            "Tool call limit (%d) reached", max_tool_calls
                        )
                        break

                    try:
                        args = json.loads(tc.arguments) if tc.arguments else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    yield StreamEvent.tool_start_event(tc.name, args)

                    try:
                        result = await self.call_tool(
                            agent, tc, tool_round=call_round
                        )
                        result_str = (
                            json.dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                        messages.append(
                            ChatMessage(
                                role="tool",
                                name=tc.name,
                                tool_call_id=tc.id,
                                content=result_str,
                            )
                        )
                        yield StreamEvent.tool_end_event(tc.name, result_str)
                        tool_call_count += 1
                    except Exception as e:
                        yield StreamEvent.error_event(f"Tool '{tc.name}' failed: {e}")
                        return

                continue

            # --- Content returned, no tools → done ---
            if full_content.strip():
                yield StreamEvent.complete_event(
                    full_content, metadata=complete_event.metadata
                )
                return

            # --- Empty response → retry once ---
            if empty_retries > 0:
                empty_retries -= 1
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Your last message was empty. You MUST "
                            "either call a tool or provide a final answer."
                        ),
                    )
                )
                continue

            yield StreamEvent.error_event(
                "LLM returned no content and no tool calls after retry"
            )
            return

        yield StreamEvent.error_event(f"Maximum tool calls ({max_tool_calls}) reached")
