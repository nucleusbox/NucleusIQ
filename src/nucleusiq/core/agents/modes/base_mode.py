"""
Base execution mode interface for NucleusIQ agents.

Each mode implements a distinct execution strategy:
- DirectMode: Fast, simple, no tools (Gear 1)
- StandardMode: Tool-enabled, linear execution (Gear 2)
- AutonomousMode: Orchestration + Critic/Refiner verification (Gear 3)

New modes can be registered via ``Agent.register_mode()`` without
modifying the Agent class (Open/Closed Principle).
"""

import asyncio
import contextlib
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
from nucleusiq.agents.diagnostics.run_report import TerminationReason, recorder_for
from nucleusiq.agents.messaging.message_builder import (
    DEFAULT_TASK_CONTEXT_CHARS,
    MessageBuilder,
)
from nucleusiq.agents.modes.loop_guards import (
    ContextToolBudget,
    ProgressTracker,
    deadline_exceeded,
    explicit_timeout,
    is_stalled_tool_content,
)
from nucleusiq.agents.modes.tool_payload import tool_result_to_context_string
from nucleusiq.agents.observability import (
    build_llm_call_record,
    build_llm_call_record_from_stream,
    build_server_tool_call_records,
    build_tool_call_record,
)
from nucleusiq.agents.structured_output.errors import StructuredOutputError
from nucleusiq.agents.structured_output.resolver import get_provider_from_llm
from nucleusiq.agents.task import Task
from nucleusiq.agents.usage.usage_tracker import CallPurpose
from nucleusiq.llms.errors import ContextLengthError, LLMTimeoutError
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


def _usage_field(usage: Any, *names: str) -> int | None:
    """Read the first present token field from a usage object or dict."""
    if usage is None:
        return None
    for name in names:
        value = (
            usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        )
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _record_llm_call_diagnostics(agent: Any, usage: Any) -> None:
    rec = recorder_for(agent)
    if rec is None:
        return
    rec.record_llm_call(
        prompt_tokens=_usage_field(usage, "prompt_tokens", "input_tokens"),
        completion_tokens=_usage_field(usage, "completion_tokens", "output_tokens"),
    )


def _set_termination(
    agent: Any,
    reason: TerminationReason,
    message: str = "",
    *,
    at_tool_call: int | None = None,
) -> None:
    rec = recorder_for(agent)
    if rec is not None:
        rec.set_termination(reason, message, at_tool_call=at_tool_call)


# After this many emergency compactions in one execute(), stop the
# tool loop. Autonomous defaults to 300 tool calls; without a cap the
# model re-fetches after each emergency wipe and never finishes.
_EMERGENCY_LOOP_LIMIT = 3


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

        try:
            from nucleusiq.agents.context.budgets import budgets_for

            context_cap = budgets_for(agent).handoff_chars("task_context")
        except Exception:
            context_cap = DEFAULT_TASK_CONTEXT_CHARS

        messages = MessageBuilder.build(
            task,
            plan,
            prompt=agent.prompt,
            logger=agent._logger,
            attachment_processor=processor,
            max_context_chars=context_cap,
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

    @staticmethod
    def emergency_loop_exhausted(agent: "Agent") -> bool:
        """True when emergency compaction has fired too many times this run."""
        engine = getattr(agent, "_context_engine", None)
        count = getattr(engine, "emergency_count", 0) if engine is not None else 0
        if not isinstance(count, int) or isinstance(count, bool):
            return False
        return count >= _EMERGENCY_LOOP_LIMIT

    # ------------------------------------------------------------------ #
    # Structured-output contract (WS-8)                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def structured_contract(agent: Any) -> Any:
        """Return a ``StructuredOutputContract`` when ``response_format`` is set.

        ``None`` for prose runs, for agents without the resolver (mocks),
        or when the schema cannot be resolved — callers then behave
        exactly as before this contract existed.
        """
        if getattr(agent, "response_format", None) is None:
            return None
        resolver = getattr(agent, "_resolve_response_format", None)
        if not callable(resolver):
            return None
        try:
            output_config = resolver()
        except Exception:
            return None
        from nucleusiq.agents.structured_output.config import OutputSchema
        from nucleusiq.agents.structured_output.contract import (
            StructuredOutputContract,
        )

        # Strict type check: test doubles (MagicMock) expose every
        # attribute, and a phantom contract would trigger finalizer calls.
        if not isinstance(output_config, OutputSchema):
            return None
        try:
            return StructuredOutputContract(output_config)
        except Exception:
            return None

    @staticmethod
    def _note_schema_check(agent: "Agent", check: Any) -> None:
        """Remember the latest schema verdict for ``AgentResult`` typing."""
        with contextlib.suppress(Exception):
            agent._last_schema_check = check
        rec = recorder_for(agent)
        if rec is not None:
            rec.record_decision(
                "structured_output",
                {"valid": bool(check.valid), "errors": (check.errors or "")[:200]},
            )

    async def _structured_finalizer_pass(
        self,
        agent: "Agent",
        messages: list[ChatMessage],
        contract: Any,
        *,
        errors: str | None = None,
    ) -> Any:
        """Tools-free call that must emit the schema — the JSON twin of synthesis.

        With no tools on the request, providers that support constrained
        decoding (``response_format`` json_schema / json_object) can
        enforce the schema server-side; the OpenAI-compatible adapter
        only injects the schema into the prompt when tools are present,
        so removing tools here is what unlocks real enforcement on vLLM.
        """
        final_messages = list(messages)
        final_messages.append(
            ChatMessage(role="user", content=contract.finalizer_instruction(errors))
        )
        engine = getattr(agent, "_context_engine", None)
        if engine is not None:
            try:
                final_messages = engine.prepare_for_synthesis(final_messages)
            except Exception as exc:
                agent._logger.debug(
                    "Finalizer rehydration skipped (fail-open): %s", exc
                )

        call_kwargs = self.build_call_kwargs(
            agent,
            final_messages,
            None,
            max_output_tokens=getattr(agent.config, "llm_max_output_tokens", 2048),
        )
        response = await self.call_llm(
            agent, call_kwargs, final_messages, None, purpose=CallPurpose.SYNTHESIS
        )
        rec = recorder_for(agent)
        if rec is not None:
            rec.record_finalizer_run()
        structured = self.handle_structured_output(agent, response)
        if structured is not None:
            return (
                structured.get("output") if isinstance(structured, dict) else structured
            )
        self.validate_response(response)
        return self.extract_content(response.choices[0].message) or ""

    async def enforce_structured_output(
        self,
        agent: "Agent",
        content: Any,
        messages: list[ChatMessage],
        contract: Any,
    ) -> tuple[Any, Any]:
        """Validate ``content`` against the contract; repair once if needed.

        Returns ``(content, check)``.  On an invalid first answer the
        structured finalizer runs **once**; if it produces valid JSON that
        becomes the content.  Otherwise the caller receives the best
        candidate with an invalid ``check`` so the Autonomous validation
        layer can retry with the exact validator errors.
        """
        check = contract.check(content)
        if check.valid:
            self._note_schema_check(agent, check)
            return content, check

        rec = recorder_for(agent)
        if rec is not None:
            rec.record_schema_validation_failure()
        agent._logger.warning(
            "Output does not satisfy schema %s — running structured finalizer: %s",
            contract.schema_name,
            (check.errors or "")[:160],
        )
        try:
            repaired = await self._structured_finalizer_pass(
                agent, messages, contract, errors=check.errors
            )
        except Exception as exc:
            agent._logger.warning("Structured finalizer failed (fail-open): %s", exc)
            self._note_schema_check(agent, check)
            return content, check

        repaired_check = contract.check(repaired)
        if repaired_check.valid:
            self._note_schema_check(agent, repaired_check)
            out = (
                repaired if isinstance(repaired, str) else repaired_check.canonical_json
            )
            return out, repaired_check

        if rec is not None:
            rec.record_schema_validation_failure()
        self._note_schema_check(agent, repaired_check)
        # Prefer the repaired text when it is at least non-empty JSON-ish;
        # the retry message will carry the validator errors either way.
        best = repaired if isinstance(repaired, str) and repaired.strip() else content
        return best, repaired_check

    @staticmethod
    def should_run_synthesis(agent: "Agent") -> bool:
        """Whether the tools-free prose synthesis pass may run.

        Synthesis exists to expand a terse *prose* wrap-up after tool
        use.  A configured ``response_format`` is the real deliverable
        (JSON / Pydantic).  That payload is almost always under the
        word threshold, so the old "write the full deliverable" nudge
        overwrote schema output with markdown.  Skip synthesis whenever
        a schema is resolved — same effect as ``enable_synthesis=False``.
        """
        if not getattr(getattr(agent, "config", None), "enable_synthesis", True):
            return False
        resolve = getattr(agent, "_resolve_response_format", None)
        if callable(resolve):
            try:
                if resolve() is not None:
                    return False
            except Exception:
                return True
        return True

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
                prepared = await engine.prepare(messages)
                if isinstance(prepared, list):
                    call_kwargs["messages"] = messages_to_dicts(prepared)
                    # prepare() fits a *copy*. If that copy is reduced and
                    # we keep the fat live list, the next turn re-triggers
                    # the same compaction (and emergency can loop toward
                    # Autonomous's 300-call default). Any reduced view is
                    # the continuing transcript — not only emergency.
                    if caller_messages is not None and prepared is not caller_messages:
                        caller_messages[:] = prepared
                        rec = recorder_for(agent)
                        if rec is not None:
                            rec.record_writeback()
            except Exception:
                pass

        t0 = time.perf_counter()
        timeout = explicit_timeout(getattr(agent, "config", None), "llm_call_timeout")
        overflow_attempt = 0
        parse_fallback_used = False
        while True:
            try:
                response = await self._invoke_llm(
                    agent, pm, call_kwargs, messages, tool_specs, timeout
                )
                break
            except ContextLengthError as exc:
                overflow_attempt += 1
                recovered = await self._recover_from_overflow(
                    agent,
                    engine,
                    call_kwargs,
                    messages,
                    caller_messages,
                    attempt=overflow_attempt,
                    error=exc,
                )
                if recovered is None:
                    raise
                messages = recovered
            except StructuredOutputError as exc:
                if parse_fallback_used or not self._disable_provider_parse(
                    agent, call_kwargs, error=exc
                ):
                    raise
                parse_fallback_used = True

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

        _record_llm_call_diagnostics(agent, getattr(response, "usage", None))

        tracer = getattr(agent, "_tracer", None)
        if tracer is not None:
            model = call_kwargs.get("model") or getattr(response, "model", None)
            prompt_tech = _extract_prompt_technique(agent)
            llm_round = len(tracer.llm_calls) + 1
            tracer.record_llm_call(
                build_llm_call_record(
                    response,
                    call_round=llm_round,
                    purpose=purpose.value,
                    duration_ms=duration_ms,
                    model=model,
                    prompt_technique=prompt_tech,
                    provider=get_provider_from_llm(getattr(agent, "llm", None)),
                )
            )
            for stc in build_server_tool_call_records(
                getattr(response, "server_tool_calls", None),
                round=llm_round,
            ):
                tracer.record_tool_call(stc)

        return response

    @staticmethod
    async def _invoke_llm(
        agent: "Agent",
        pm: Any,
        call_kwargs: dict[str, Any],
        messages: list[ChatMessage] | None,
        tool_specs: list[dict[str, Any]] | None,
        timeout: float | None,
    ) -> Any:
        """One provider round-trip through the plugin pipeline, with timeout."""

        llm = agent.llm
        if llm is None:
            raise RuntimeError("agent.llm must be set before calling call_llm")

        async def _do() -> Any:
            if pm is None or not pm.has_plugins():
                return await llm.call(**call_kwargs)
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
            response = await pm.execute_model_call(request, llm.call)
            return await pm.run_after_model(request, response)

        if timeout is None:
            return await _do()
        try:
            return await asyncio.wait_for(_do(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            detail = f"LLM call exceeded llm_call_timeout={timeout:g}s"
            _set_termination(agent, TerminationReason.LLM_TIMEOUT, detail)
            raise LLMTimeoutError(
                detail,
                provider=str(getattr(agent.llm, "provider", "unknown") or "unknown"),
                original_error=exc,
            ) from exc

    @staticmethod
    def _disable_provider_parse(
        agent: "Agent",
        call_kwargs: dict[str, Any],
        *,
        error: Exception,
    ) -> bool:
        """Turn a provider-side structured-output parse failure into a
        recoverable step (WS-8).

        In NATIVE mode the provider receives ``response_format=(fmt,
        SchemaType)`` and parses the reply into ``SchemaType`` itself.  A
        model that returns two JSON objects, JSON followed by prose, or a
        field that fails validation makes that parse raise
        ``StructuredOutputError`` **inside** ``llm.call`` — before the
        core contract (repair → finalizer → ``schema_invalid``) ever sees
        the text.  Left alone the run ends in ``ResultStatus.ERROR`` for
        something the framework knows how to handle.

        This strips the schema type (keeping the server-side JSON
        constraint) so the retry returns the raw reply and the normal
        contract path decides.  Returns ``True`` when the kwargs were
        changed and one retry is warranted, ``False`` otherwise.
        """
        fmt = call_kwargs.get("response_format")
        if isinstance(fmt, tuple) and len(fmt) == 2 and isinstance(fmt[1], type):
            provider_format: Any = fmt[0]
        elif isinstance(fmt, type):
            # Bare schema class: every provider builds its wire format from
            # the class *and* parses into it.  Ask the resolved OutputSchema
            # for the same wire format so the constraint survives without
            # the parse.
            try:
                output_config = agent._resolve_response_format()
                provider = (
                    get_provider_from_llm(getattr(agent, "llm", None)) or "openai"
                )
                provider_format = output_config.for_provider(provider)
            except Exception:
                return False
            if not isinstance(provider_format, dict):
                return False
        else:
            return False
        call_kwargs["response_format"] = (provider_format, None)
        detail = f"{type(error).__name__}: {str(error)[:160]}"
        agent._logger.warning(
            "Provider could not parse the structured reply (%s). Retrying once "
            "without provider-side parsing; the output contract will validate "
            "the raw text instead.",
            detail,
        )
        rec = recorder_for(agent)
        if rec is not None:
            with contextlib.suppress(Exception):
                rec.record_event("structured_parse_fallback", detail)
        return True

    async def _recover_from_overflow(
        self,
        agent: "Agent",
        engine: Any,
        call_kwargs: dict[str, Any],
        messages: list[ChatMessage] | None,
        caller_messages: list[ChatMessage] | None,
        *,
        attempt: int,
        error: Exception,
    ) -> list[ChatMessage] | None:
        """Turn a provider ``ContextLengthError`` into a recoverable step.

        Attempt 1: force the emergency compaction tier and write the
        reduced transcript back (our token estimate under-counted; the
        server is the authority).  Attempt 2: shrink ``max_output_tokens``
        so ``prompt + reply`` fits the window.  Anything further records
        ``termination_reason=context_overflow`` with the numbers and
        re-raises so the caller fails with a precise message instead of
        a bare 400.

        Returns the message list to retry with, or ``None`` to give up.
        """
        rec = recorder_for(agent)
        window = int(getattr(engine, "resolved_max_tokens", 0) or 0)
        counter = getattr(engine, "token_counter", None)
        prompt_tokens = 0
        if counter is not None and messages is not None:
            with contextlib.suppress(Exception):
                prompt_tokens = int(counter.count_messages(messages))
        max_out = call_kwargs.get("max_output_tokens")

        if attempt == 1 and engine is not None and messages is not None:
            try:
                reduced = await engine.force_emergency(messages)
            except Exception as exc:
                agent._logger.debug("Forced emergency compaction failed: %s", exc)
                reduced = messages
            after = prompt_tokens
            if counter is not None:
                with contextlib.suppress(Exception):
                    after = int(counter.count_messages(reduced))
            if after < prompt_tokens:
                agent._logger.warning(
                    "Provider rejected the prompt as too long (%s). Forced "
                    "emergency compaction %d → %d tokens; retrying once.",
                    str(error)[:120],
                    prompt_tokens,
                    after,
                )
                if isinstance(reduced, list):
                    call_kwargs["messages"] = messages_to_dicts(reduced)
                    if caller_messages is not None:
                        caller_messages[:] = reduced
                if rec is not None:
                    rec.record_compaction("emergency_compactor", emergency=True)
                    rec.record_writeback()
                    rec.record_event(
                        "context_overflow_recovery",
                        f"emergency compaction {prompt_tokens}->{after} tokens",
                    )
                return reduced
            # Nothing to compact — fall through to the max_tokens step.
            attempt = 2

        if attempt == 2 and isinstance(max_out, int) and window > 0:
            new_max = window - prompt_tokens - 256
            if 128 <= new_max < max_out:
                agent._logger.warning(
                    "Provider rejected the request as too long (%s). Reducing "
                    "max_output_tokens %d → %d (window %d, prompt ≈ %d); "
                    "retrying once.",
                    str(error)[:120],
                    max_out,
                    new_max,
                    window,
                    prompt_tokens,
                )
                call_kwargs["max_output_tokens"] = new_max
                if rec is not None:
                    rec.record_event(
                        "context_overflow_recovery",
                        f"max_output_tokens {max_out}->{new_max}",
                    )
                return messages if messages is not None else []

        detail = (
            f"Context overflow: window={window or 'unknown'} prompt≈{prompt_tokens} "
            f"max_output_tokens={max_out} — {str(error)[:200]}"
        )
        agent._logger.error("%s", detail)
        _set_termination(agent, TerminationReason.CONTEXT_OVERFLOW, detail)
        return None

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

        # Resolve any optional ``source`` label the tool advertises
        # (e.g. ``"mcp://server=github (path=A)"`` from MCPBoundTool).
        # We look up by name once and read the attribute defensively so
        # non-MCP tools simply yield ``None``.
        tool_source: str | None = None
        if tc.name:
            for _t in agent.tools or []:
                if getattr(_t, "name", None) == tc.name:
                    src = getattr(_t, "source", None)
                    if isinstance(src, str):
                        tool_source = src
                    break

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
                        source=tool_source,
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
                        source=tool_source,
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
        recorder = recorder_for(agent)
        cap_getter = getattr(agent.config, "get_effective_max_context_tool_calls", None)
        context_cap = 2 * max_tool_calls
        if callable(cap_getter):
            try:
                value = cap_getter()
                if isinstance(value, int):
                    context_cap = value
            except Exception:
                pass
        context_budget = ContextToolBudget(context_cap)
        progress = ProgressTracker()
        contract = self.structured_contract(agent)
        stop_reason = TerminationReason.TOOL_BUDGET
        stop_detail = f"Maximum tool calls ({max_tool_calls}) reached"

        while tool_call_count < max_tool_calls:
            if deadline_exceeded(agent):
                stop_reason = TerminationReason.DEADLINE
                stop_detail = (
                    f"Wall-clock budget (max_execution_time="
                    f"{getattr(agent.config, 'max_execution_time', '?')}s) exhausted "
                    f"after {tool_call_count} tool calls"
                )
                break
            call_round += 1
            if recorder is not None:
                recorder.record_round(call_round)

            # Snapshot messages *before* masking may compress data.
            # Synthesis needs the full, unmasked context.
            if (
                self.should_run_synthesis(agent)
                and tool_call_count > 0
                and call_round > 2
            ):
                pre_synth_snapshot = list(messages)

            current_purpose = purpose if call_round == 1 else CallPurpose.TOOL_LOOP
            yield StreamEvent.llm_start_event(call_round)

            engine = getattr(agent, "_context_engine", None)
            if engine is not None:
                try:
                    compacted = await engine.prepare(messages)
                    if isinstance(compacted, list) and compacted is not messages:
                        messages[:] = compacted
                        if recorder is not None:
                            recorder.record_writeback()
                except Exception:
                    pass

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
                _set_termination(agent, TerminationReason.ERROR, "LLM stream error")
                return

            stream_duration_ms = (time.perf_counter() - stream_t0) * 1000

            yield StreamEvent.llm_end_event(call_round)

            if complete_event is None:
                _set_termination(
                    agent,
                    TerminationReason.ERROR,
                    "LLM stream produced no COMPLETE event",
                )
                yield StreamEvent.error_event("LLM stream produced no COMPLETE event")
                return

            if tracker is not None:
                tracker.record_from_stream_metadata(
                    current_purpose,
                    complete_event.metadata,
                    call_round=call_round,
                )
            _record_llm_call_diagnostics(
                agent, (complete_event.metadata or {}).get("usage")
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
                        provider=get_provider_from_llm(getattr(agent, "llm", None)),
                    )
                )
                for stc in build_server_tool_call_records(
                    (complete_event.metadata or {}).get("server_tool_calls"),
                    round=call_round,
                ):
                    tracer.record_tool_call(stc)

            full_content = complete_event.content or ""
            raw_tool_calls = (complete_event.metadata or {}).get("tool_calls", [])

            # --- Tool calls detected → execute and loop ---
            if raw_tool_calls:
                parsed_calls = [ToolCallRequest.from_raw(tc) for tc in raw_tool_calls]
                round_start = len(messages)
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
                    is_ctx_tool = is_context_management_tool_name(tc.name)
                    # Recall tools (memory operations) bypass the
                    # tool-call budget — see §6.4 of the v2 redesign —
                    # but consume their own ``ContextToolBudget``.
                    if not is_ctx_tool and tool_call_count >= max_tool_calls:
                        agent._logger.warning(
                            "Tool call limit (%d) reached", max_tool_calls
                        )
                        break
                    if is_ctx_tool:
                        context_budget.consume()
                    if recorder is not None:
                        recorder.record_tool_call(context_management=is_ctx_tool)

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
                        if recorder is not None:
                            recorder.record_dedup_banner()
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
                        step_timeout = explicit_timeout(
                            getattr(agent, "config", None), "step_timeout"
                        )
                        if step_timeout is None:
                            result = await self.call_tool(
                                agent, tc, tool_round=call_round
                            )
                        else:
                            try:
                                result = await asyncio.wait_for(
                                    self.call_tool(agent, tc, tool_round=call_round),
                                    timeout=step_timeout,
                                )
                            except asyncio.TimeoutError:
                                result = (
                                    f"Error: Tool '{tc.name}' timed out after "
                                    f"{step_timeout:g}s (step_timeout). Continue without it."
                                )
                                agent._logger.warning("%s", result)
                                if recorder is not None:
                                    recorder.record_tool_error()
                        agent._activate_context_state_for_tool_result(
                            tool_name=tc.name,
                            tool_call_id=tc.id,
                            tool_result=result,
                            tool_args=args,
                        )
                        result_str = tool_result_to_context_string(result)
                        if recorder is not None and is_stalled_tool_content(result_str):
                            recorder.record_recall_error()

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
                        if not is_ctx_tool:
                            tool_call_count += 1
                        if is_idempotent and tc.id is not None:
                            dedup_cache[cache_key] = tc.id
                    except Exception as e:
                        if recorder is not None:
                            recorder.record_tool_error()
                        _set_termination(
                            agent, TerminationReason.ERROR, f"Tool '{tc.name}' failed"
                        )
                        yield StreamEvent.error_event(f"Tool '{tc.name}' failed: {e}")
                        return

                # Guards run on the raw round *before* masking so the
                # signature sees real tool content.
                verdict = progress.observe(messages[round_start:])
                if recorder is not None:
                    if verdict.stalled_round:
                        recorder.record_stalled_round()
                    if verdict.identical_streak > 1:
                        recorder.record_no_progress_round()

                # Post-response hook: observation masking (Tier 0) for streaming path
                # Runs after assistant + tool results are in messages, before next LLM call
                engine = getattr(agent, "_context_engine", None)
                if engine is not None:
                    try:
                        masked = engine.post_response(messages)
                        messages[:] = masked
                    except Exception:
                        pass

                if verdict.no_progress:
                    stop_reason = TerminationReason.NO_PROGRESS
                    stop_detail = f"No progress: {verdict.reason}"
                    agent._logger.warning(
                        "Stopping tool loop — %s (round %d)", verdict.reason, call_round
                    )
                    break
                if verdict.should_nudge:
                    nudge = progress.nudge_message()
                    if nudge is not None:
                        agent._logger.warning(
                            "Loop guard: repeated tool round detected (round %d) — "
                            "nudging model to answer with existing evidence",
                            call_round,
                        )
                        messages.append(nudge)
                        if recorder is not None:
                            recorder.record_event("no_progress_nudge")
                if context_budget.exhausted:
                    stop_reason = TerminationReason.CONTEXT_TOOL_BUDGET
                    stop_detail = (
                        f"Context-management tool budget ({context_budget.cap}) "
                        "exhausted"
                    )
                    agent._logger.warning("%s", stop_detail)
                    break
                if self.emergency_loop_exhausted(agent):
                    stop_reason = TerminationReason.EMERGENCY_COMPACTION
                    stop_detail = (
                        "Repeated emergency compaction — context cannot absorb "
                        "more tool results"
                    )
                    agent._logger.warning(
                        "Stopping tool loop after repeated emergency "
                        "compaction — context cannot absorb more tool results"
                    )
                    break

                continue

            # --- Content returned, no tools → synthesis or done ---
            if full_content.strip():
                synth_threshold = getattr(agent.config, "synthesis_word_threshold", 500)
                if contract is not None:
                    # Schema is the deliverable: check, repair once, never
                    # run the prose synthesis pass.
                    repaired, _check = await self.enforce_structured_output(
                        agent, full_content, messages, contract
                    )
                    if isinstance(repaired, str) and repaired != full_content:
                        # Tokens already streamed were the rejected draft;
                        # the COMPLETE event below carries the final JSON.
                        yield StreamEvent.thinking_event(
                            f"Structured finalizer produced {contract.schema_name} JSON"
                        )
                        full_content = repaired
                elif (
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
                    synthesis_messages: list[ChatMessage] = list(synth_msgs)

                    engine = getattr(agent, "_context_engine", None)
                    if engine is not None:
                        try:
                            compacted = await engine.prepare(synthesis_messages)
                            if isinstance(compacted, list):
                                synthesis_messages = compacted
                        except Exception:
                            pass
                        # v2 §7 — rehydrate evidence markers so the
                        # tools=None synthesis call has the original
                        # bytes, not just markers.  Fail-open.
                        try:
                            rehydrated = engine.prepare_for_synthesis(
                                synthesis_messages
                            )
                            if isinstance(rehydrated, list):
                                synthesis_messages = rehydrated
                        except Exception as exc:
                            agent._logger.debug(
                                "Streaming synthesis rehydration skipped "
                                "(fail-open): %s",
                                exc,
                            )

                    synth_kwargs = self.build_call_kwargs(
                        agent,
                        synthesis_messages,
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
                                    provider=get_provider_from_llm(
                                        getattr(agent, "llm", None)
                                    ),
                                )
                            )
                            for stc in build_server_tool_call_records(
                                (synth_event.metadata or {}).get("server_tool_calls"),
                                round=call_round,
                            ):
                                synth_tracer.record_tool_call(stc)
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
                            _set_termination(agent, TerminationReason.COMPLETED)
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
                _set_termination(agent, TerminationReason.COMPLETED)
                yield StreamEvent.complete_event(
                    full_content, metadata=complete_event.metadata
                )
                return

            # --- Empty response → retry once ---
            if empty_retries > 0:
                empty_retries -= 1
                if recorder is not None:
                    recorder.record_empty_response()
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

            _set_termination(
                agent,
                TerminationReason.EMPTY_RESPONSE,
                "LLM returned no content and no tool calls after retry",
            )
            yield StreamEvent.error_event(
                "LLM returned no content and no tool calls after retry"
            )
            return

        _set_termination(agent, stop_reason, stop_detail, at_tool_call=tool_call_count)
        yield StreamEvent.error_event(stop_detail)
