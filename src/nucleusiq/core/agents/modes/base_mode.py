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
from typing import TYPE_CHECKING, Any

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
from nucleusiq.agents.modes.tool_payload import tool_result_to_context_string
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


def _extract_prompt_technique(agent: Any) -> str | None:
    """Safely extract the prompt technique name from an agent's prompt.

    Returns None if the agent has no prompt, no technique attribute,
    or if the technique is not a string/enum value.
    """
    try:
        prompt_obj = getattr(agent, "prompt", None)
        if prompt_obj is None:
            return None
        tech = getattr(prompt_obj, "technique", None)
        if tech is None:
            return None
        if hasattr(tech, "value"):
            val = tech.value
            return str(val) if isinstance(val, str) else None
        return str(tech) if isinstance(tech, str) else None
    except Exception:
        return None


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
    def get_objective(task: Task | dict[str, Any]) -> str:
        """Extract the objective string from a Task or dict.

        Accepts both forms for backward compatibility with external callers,
        but internal callers should always pass a ``Task`` instance.
        """
        if isinstance(task, Task):
            return task.objective
        return task.get("objective", "")

    def echo_fallback(self, agent: "Agent", task: Task | dict[str, Any]) -> str | None:
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
        task: Task | dict[str, Any],
        plan: Any = None,
    ) -> list[ChatMessage]:
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
        messages: list[ChatMessage],
        tool_specs: list[dict[str, Any]] | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``agent.llm.call()``.

        Merges model name, messages, tool specs, max_output_tokens,
        per-execute LLM overrides, and structured-output kwargs.
        """
        output_config = agent._resolve_response_format()
        call_kwargs: dict[str, Any] = {
            "model": getattr(agent.llm, "model_name", "default"),
            "messages": messages_to_dicts(messages),
            "tools": tool_specs if tool_specs else None,
            "max_output_tokens": max_output_tokens
            or getattr(agent.config, "llm_max_output_tokens", 2048),
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
            parts: list[str] = []
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

    @staticmethod
    def _finalize_post_response(agent: "Agent", messages: list[ChatMessage]) -> None:
        """F7 — Run ``engine.post_response`` on the **terminal** state.

        ``call_llm`` / the streaming tool loop already invoke
        ``post_response`` on every intermediate round.  But both paths
        historically skipped masking after the terminal assistant was
        appended — leaving the last round's tool results unmasked for
        downstream consumers (Critic, Refiner) in Autonomous mode.

        This helper closes the gap so streaming and non-streaming end
        in an identical, fully-masked conversation state.  It is an
        idempotent no-op when:

        * no ``ContextEngine`` is attached (e.g. in unit tests);
        * the engine has nothing left to mask;
        * any exception is raised internally (swallowed, same as the
          per-round hook — masking failures must never break the
          user's task).
        """
        engine = getattr(agent, "_context_engine", None)
        if engine is None or not messages:
            return
        try:
            masked = engine.post_response(messages)
            messages[:] = masked
        except Exception:
            pass

    async def call_llm(
        self,
        agent: "Agent",
        call_kwargs: dict[str, Any],
        messages: list[ChatMessage] | None = None,
        tool_specs: list[dict[str, Any]] | None = None,
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

        # Keep reference to the caller's list so post_response always
        # updates it, even when prepare() returns a new compacted list.
        caller_messages = messages

        # Context window management: prepare messages before LLM call
        engine = getattr(agent, "_context_engine", None)
        if engine is not None and messages is not None:
            try:
                messages = await engine.prepare(messages)
                call_kwargs["messages"] = messages_to_dicts(messages)
            except Exception:
                pass

        t0 = time.perf_counter()
        if pm is None or not pm.has_plugins():
            response = await agent.llm.call(**call_kwargs)
        else:
            reserved = {"model", "messages", "tools", "max_output_tokens"}
            extra = {k: v for k, v in call_kwargs.items() if k not in reserved}

            request = ModelRequest(
                model=call_kwargs.get("model", "default"),
                messages=messages
                if messages is not None
                else call_kwargs.get("messages", []),
                tools=tool_specs,
                max_output_tokens=call_kwargs.get("max_output_tokens", 2048),
                call_count=pm.increment_model_calls(),
                agent_name=agent.name,
                extra_kwargs=extra,
            )

            request = await pm.run_before_model(request)
            response = await pm.execute_model_call(request, agent.llm.call)
            response = await pm.run_after_model(request, response)

        duration_ms = (time.perf_counter() - t0) * 1000

        # Post-response hook: observation masking (Tier 0)
        # Always target the *caller's* list so masking is applied
        # consistently regardless of whether prepare() compacted.
        if engine is not None and caller_messages is not None:
            try:
                masked = engine.post_response(caller_messages)
                caller_messages[:] = masked
            except Exception:
                pass

        tracker = getattr(agent, "_usage_tracker", None)
        if tracker is not None:
            tracker.record_from_response(purpose, response)

        tracer = getattr(agent, "_tracer", None)
        if tracer is not None:
            model = call_kwargs.get("model") or getattr(response, "model", None)
            prompt_tech = _extract_prompt_technique(agent)
            tracer.record_llm_call(
                build_llm_call_record(
                    response,
                    call_round=len(tracer.llm_calls) + 1,
                    purpose=purpose.value,
                    duration_ms=duration_ms,
                    model=model,
                    prompt_technique=prompt_tech,
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
        assert agent._executor is not None, (
            "agent._executor must be set before calling call_tool"
        )
        pm = getattr(agent, "_plugin_manager", None)

        tool_args: dict[str, Any] = {}
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
        call_kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Invoke ``agent.llm.call_stream()`` yielding ``StreamEvent`` objects.

        Analogous to ``call_llm()`` but for the streaming path.
        Plugin-aware streaming is deferred to a future release.
        """
        assert agent.llm is not None, (
            "agent.llm must be set before calling call_llm_stream"
        )
        async for event in agent.llm.call_stream(**call_kwargs):
            yield event

    async def _streaming_tool_call_loop(
        self,
        agent: "Agent",
        messages: list[ChatMessage],
        tool_specs: list[dict[str, Any]] | None,
        *,
        max_tool_calls: int = 80,
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
        empty_retries = 2
        tracker = getattr(agent, "_usage_tracker", None)
        pre_synth_snapshot: list[ChatMessage] | None = None

        while tool_call_count < max_tool_calls:
            call_round += 1

            # Snapshot messages *before* masking may compress data.
            # Synthesis needs the full, unmasked context.
            if (
                getattr(agent.config, "enable_synthesis", True)
                and tool_call_count > 0
                and call_round > 2
            ):
                pre_synth_snapshot = list(messages)

            current_purpose = purpose if call_round == 1 else CallPurpose.TOOL_LOOP
            yield StreamEvent.llm_start_event(call_round)

            engine = getattr(agent, "_context_engine", None)
            prepared = messages
            if engine is not None:
                try:
                    prepared = await engine.prepare(messages)
                except Exception:
                    prepared = messages

            call_kwargs = self.build_call_kwargs(
                agent, prepared, tool_specs, max_output_tokens=max_output_tokens
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
                        prompt_technique=_extract_prompt_technique(agent),
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

                from nucleusiq.agents.context.workspace_tools import (
                    is_context_management_tool_name,
                )

                for tc in parsed_calls:
                    if not tc.name:
                        continue
                    # Recall tools (memory operations) bypass the
                    # tool-call budget — see §6.4 of the v2 redesign.
                    if (
                        not is_context_management_tool_name(tc.name)
                        and tool_call_count >= max_tool_calls
                    ):
                        agent._logger.warning(
                            "Tool call limit (%d) reached", max_tool_calls
                        )
                        break

                    try:
                        args = json.loads(tc.arguments) if tc.arguments else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    yield StreamEvent.tool_start_event(tc.name, args)

                    # Context Mgmt v2 — Step 4: idempotent-tool dedup
                    # mirrored from StandardMode._process_tool_calls.
                    # See standard_mode.py for the rationale.
                    from nucleusiq.agents.modes.standard_mode import (
                        _IDEMPOTENT_DEDUP_BANNER,
                        _get_tool_by_name,
                        _hash_tool_args,
                    )

                    dedup_cache: dict[tuple[str, str], str] = (
                        getattr(agent, "_tool_dedup_cache", None) or {}
                    )
                    agent._tool_dedup_cache = dedup_cache

                    tool_obj = _get_tool_by_name(agent, tc.name)
                    is_idempotent = bool(getattr(tool_obj, "idempotent", False))
                    args_hash = _hash_tool_args(tc.arguments or "")
                    cache_key = (tc.name, args_hash)
                    prior_call_id = dedup_cache.get(cache_key)

                    if is_idempotent and prior_call_id is not None:
                        from nucleusiq.agents.context.compactor import (
                            _build_args_preview,
                        )

                        args_preview = _build_args_preview(
                            {"function": {"arguments": tc.arguments or "{}"}}
                        )
                        banner = _IDEMPOTENT_DEDUP_BANNER.format(
                            tool_name=tc.name,
                            args_preview=args_preview,
                            original_call_id=prior_call_id,
                        )
                        agent._logger.info(
                            "Tool dedup (stream): %s args_hash=%s already "
                            "called (original_call_id=%s) — short-circuit",
                            tc.name,
                            args_hash,
                            prior_call_id,
                        )
                        messages.append(
                            ChatMessage(
                                role="tool",
                                name=tc.name,
                                tool_call_id=tc.id,
                                content=banner,
                            )
                        )
                        yield StreamEvent.tool_end_event(tc.name, banner)
                        if not is_context_management_tool_name(tc.name):
                            tool_call_count += 1
                        continue

                    try:
                        result = await self.call_tool(agent, tc, tool_round=call_round)
                        agent._activate_context_state_for_tool_result(
                            tool_name=tc.name,
                            tool_call_id=tc.id,
                            tool_result=result,
                            tool_args=args,
                        )
                        result_str = tool_result_to_context_string(result)

                        # Context window management: compress large tool results
                        engine = getattr(agent, "_context_engine", None)
                        if engine is not None:
                            result_str = engine.ingest_tool_result(result_str, tc.name)

                        messages.append(
                            ChatMessage(
                                role="tool",
                                name=tc.name,
                                tool_call_id=tc.id,
                                content=result_str,
                            )
                        )
                        yield StreamEvent.tool_end_event(tc.name, result_str)
                        # Auto-injected context-management tools do not count
                        # against the user's tool budget — they are memory ops,
                        # not external actions.
                        if not is_context_management_tool_name(tc.name):
                            tool_call_count += 1
                        if is_idempotent and tc.id is not None:
                            dedup_cache[cache_key] = tc.id
                    except Exception as e:
                        yield StreamEvent.error_event(f"Tool '{tc.name}' failed: {e}")
                        return

                # Post-response hook: observation masking (Tier 0) for streaming path
                # Runs after assistant + tool results are in messages, before next LLM call
                engine = getattr(agent, "_context_engine", None)
                if engine is not None:
                    try:
                        masked = engine.post_response(messages)
                        messages[:] = masked
                    except Exception:
                        pass

                continue

            # --- Content returned, no tools → synthesis or done ---
            if full_content.strip():
                synth_threshold = getattr(agent.config, "synthesis_word_threshold", 500)
                if (
                    pre_synth_snapshot is not None
                    and len(full_content.split()) < synth_threshold
                ):
                    agent._logger.info(
                        "Synthesis pass (stream): %d tool calls over %d rounds",
                        tool_call_count,
                        call_round - 1,
                    )
                    call_round += 1
                    yield StreamEvent.llm_start_event(call_round)

                    task_obj = getattr(agent, "_current_task", None) or {}
                    task_text = (
                        task_obj.get("objective", "")
                        if isinstance(task_obj, dict)
                        else str(task_obj or "")
                    )
                    synth_msgs = agent._build_synthesis_messages_from_context(
                        task=task_text or "Complete the requested task.",
                        output_shape=(
                            "Produce the COMPLETE, FULL-LENGTH deliverable "
                            "exactly as described in the user's instructions."
                        ),
                    )
                    if synth_msgs is None:
                        synth_msgs = list(pre_synth_snapshot)
                        synth_msgs.append(
                            ChatMessage(
                                role="user",
                                content=(
                                    "All data gathering is complete. "
                                    "Now produce the COMPLETE, FULL-LENGTH "
                                    "deliverable exactly as described in your "
                                    "instructions. Do not summarize — write "
                                    "the entire output."
                                ),
                            )
                        )

                    engine = getattr(agent, "_context_engine", None)
                    if engine is not None:
                        try:
                            synth_msgs = await engine.prepare(synth_msgs)
                        except Exception:
                            pass
                        # v2 §7 — rehydrate evidence markers so the
                        # tools=None synthesis call has the original
                        # bytes, not just markers.  Fail-open.
                        try:
                            synth_msgs = engine.prepare_for_synthesis(synth_msgs)
                        except Exception as exc:
                            agent._logger.debug(
                                "Streaming synthesis rehydration skipped "
                                "(fail-open): %s",
                                exc,
                            )

                    synth_kwargs = self.build_call_kwargs(
                        agent,
                        synth_msgs,
                        None,
                        max_output_tokens=max_output_tokens,
                    )
                    synth_event: StreamEvent | None = None
                    synth_t0 = time.perf_counter()
                    async for ev in self.call_llm_stream(agent, synth_kwargs):
                        if ev.type == StreamEventType.TOKEN:
                            yield ev
                        elif ev.type == StreamEventType.COMPLETE:
                            synth_event = ev
                        elif ev.type == StreamEventType.ERROR:
                            yield StreamEvent.llm_end_event(call_round)
                            yield ev
                            return

                    synth_dur = (time.perf_counter() - synth_t0) * 1000
                    yield StreamEvent.llm_end_event(call_round)

                    if synth_event is not None:
                        if tracker is not None:
                            tracker.record_from_stream_metadata(
                                CallPurpose.SYNTHESIS,
                                synth_event.metadata,
                                call_round=call_round,
                            )
                        synth_tracer = getattr(agent, "_tracer", None)
                        if synth_tracer is not None:
                            synth_tracer.record_llm_call(
                                build_llm_call_record_from_stream(
                                    synth_event.metadata,
                                    call_round=call_round,
                                    purpose=CallPurpose.SYNTHESIS.value,
                                    duration_ms=synth_dur,
                                    model=synth_kwargs.get("model"),
                                    prompt_technique=_extract_prompt_technique(agent),
                                )
                            )
                        synth_text = (synth_event.content or "").strip()
                        if synth_text:
                            messages.append(
                                ChatMessage(
                                    role="assistant",
                                    content=synth_event.content or "",
                                )
                            )
                            # F7 — synthesis terminal masking.
                            # The non-streaming synthesis path reaches
                            # post_response via its inner ``call_llm``;
                            # the streaming synthesis path bypasses
                            # ``call_llm`` entirely, so we must call it
                            # explicitly here to keep both paths
                            # converging on the same masked final state.
                            self._finalize_post_response(agent, messages)
                            yield StreamEvent.complete_event(
                                synth_event.content or "",
                                metadata=synth_event.metadata,
                            )
                            return
                        agent._logger.warning(
                            "Synthesis pass (stream) returned empty — "
                            "preserving pre-synthesis content"
                        )

                messages.append(ChatMessage(role="assistant", content=full_content))
                # F7 — terminal post_response symmetry (streaming path).
                # Mirrors ``StandardMode._tool_call_loop`` so both modes
                # end in an identical masked state for downstream
                # consumers (Critic/Refiner in Autonomous mode).
                self._finalize_post_response(agent, messages)
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
