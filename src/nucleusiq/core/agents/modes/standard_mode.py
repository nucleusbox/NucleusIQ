"""
StandardMode — Gear 2: Tool-enabled, linear execution.

Logic: Input -> Decision -> Tool Execution -> Result

Use Cases: "Check the weather", "Query database", "Search information"

Characteristics:
- Tool execution enabled
- Linear flow (no loops)
- Fire-and-forget (tries once, returns error if fails)
- Optional memory
- Multiple tool calls supported
"""

import hashlib
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nucleusiq.agents.agent import Agent
    from nucleusiq.tools.base_tool import BaseTool

from nucleusiq.agents.chat_models import ChatMessage, ToolCallRequest
from nucleusiq.agents.components.executor import Executor
from nucleusiq.agents.config.agent_config import AgentState
from nucleusiq.agents.modes.base_mode import BaseExecutionMode
from nucleusiq.agents.modes.tool_payload import tool_result_to_context_string
from nucleusiq.agents.task import Task
from nucleusiq.agents.usage.usage_tracker import CallPurpose
from nucleusiq.plugins.errors import PluginHalt
from nucleusiq.streaming.events import StreamEvent, StreamEventType

# Context Mgmt v2 — Step 4 (re-fetch loop fix).
# When an idempotent tool is invoked with arguments identical to a
# prior call within the same execution, the agent layer returns this
# banner instead of re-executing.  The banner names the original call
# id so the model can correlate it with the result already in its
# transcript (or recall it via recall_tool_result if the result has
# since been masked).  Dedup is *opt-in per tool* (see
# ``BaseTool.idempotent``); live-data tools (weather, stock, time)
# default to False and are never deduped.
_IDEMPOTENT_DEDUP_BANNER = (
    "[duplicate idempotent call — short-circuited]\n"
    "tool: {tool_name}\n"
    "args: {args_preview}\n"
    "You already called this tool with these exact arguments earlier in "
    "this execution (original tool_call_id: {original_call_id}).\n"
    "The earlier result is in your conversation history above — either "
    "as the original tool message, or as an [observation consumed] "
    "marker if the masker has since fired.\n"
    "Do NOT re-fetch.  Use the prior result, or call "
    "recall_tool_result(ref=...) if it was masked.  Make progress with "
    "what you already have."
)


def _hash_tool_args(args_str: str) -> str:
    """Stable short hash of a JSON-serialised tool-call arguments string.

    Uses a canonical JSON re-serialise so semantically-equal args
    (different key order, equivalent whitespace) hash to the same
    value.  Falls back to the raw string if parsing fails (still
    deterministic within a run).
    """
    try:
        parsed = json.loads(args_str) if args_str else {}
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        canonical = args_str or ""
    return hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :16
    ]


def _get_tool_by_name(agent: "Agent", name: str | None) -> "BaseTool | None":
    """Return the tool registered on ``agent`` matching ``name`` (or None)."""
    if not name:
        return None
    for t in agent.tools or []:
        if getattr(t, "name", None) == name:
            return t
    return None


class StandardMode(BaseExecutionMode):
    """Gear 2: Standard mode — tool-enabled, linear execution (max 80 by default)."""

    async def run(self, agent: "Agent", task: Task) -> Any:
        """Execute a task with tool-calling loop."""
        agent._logger.debug("Executing in STANDARD mode (tool-enabled, linear)")
        agent.state = AgentState.EXECUTING

        # Reset per-execution idempotent-tool dedup cache so retries
        # (e.g. autonomous SimpleRunner re-invoking StandardMode after a
        # validator failure) start fresh and aren't blocked by a prior
        # attempt's call history.
        agent._tool_dedup_cache = {}

        # Fast path: no LLM -> echo
        echo = self.echo_fallback(agent, task)
        if echo is not None:
            return echo

        # Ensure executor is ready
        self._ensure_executor(agent)

        # Convert tools to LLM-specific format
        tool_specs = self._get_tool_specs(agent)

        # Build initial messages
        messages = self.build_messages(agent, task)

        # Persist user objective (with attachment context) in memory
        await self.store_task_in_memory(agent, task)

        try:
            result = await self._tool_call_loop(agent, task, messages, tool_specs)
            agent._last_messages = messages
            return result
        except PluginHalt:
            raise
        except Exception as e:
            agent._logger.error("Error during standard execution: %s", str(e))
            agent.state = AgentState.ERROR
            from nucleusiq.agents.errors import AgentExecutionError

            raise AgentExecutionError(
                f"Standard mode execution failed: {e}",
                mode="standard",
                original_error=e,
            ) from e

    # ------------------------------------------------------------------ #
    # Streaming                                                           #
    # ------------------------------------------------------------------ #

    async def run_stream(
        self, agent: "Agent", task: Task
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a Standard mode execution.

        Delegates to the shared ``_streaming_tool_call_loop`` and
        persists the final result in agent memory.
        """
        agent._logger.debug("Streaming in STANDARD mode (tool-enabled, linear)")
        agent.state = AgentState.EXECUTING

        # Same per-execution dedup cache reset as run() — see comment there.
        agent._tool_dedup_cache = {}

        echo = self.echo_fallback(agent, task)
        if echo is not None:
            yield StreamEvent.complete_event(echo)
            return

        self._ensure_executor(agent)
        tool_specs = self._get_tool_specs(agent)
        messages = self.build_messages(agent, task)
        max_tool_calls = agent.config.get_effective_max_tool_calls()

        await self.store_task_in_memory(agent, task)

        final_content: str | None = None

        try:
            async for event in self._streaming_tool_call_loop(
                agent,
                messages,
                tool_specs,
                max_tool_calls=max_tool_calls,
                max_output_tokens=getattr(agent.config, "llm_max_output_tokens", 2048),
            ):
                if event.type == StreamEventType.COMPLETE:
                    final_content = event.content
                yield event

            agent._last_messages = messages
            agent.state = AgentState.COMPLETED

            if final_content:
                await self._store_in_memory(agent, task, final_content)

        except PluginHalt:
            raise
        except Exception as e:
            agent._logger.error("Streaming error in standard mode: %s", e)
            agent.state = AgentState.ERROR
            yield StreamEvent.error_event(str(e))

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_executor(self, agent: "Agent") -> None:
        """Lazily create an Executor if the agent does not have one."""
        if not hasattr(agent, "_executor") or agent._executor is None:
            if agent.llm:
                agent._executor = Executor(agent.llm, agent.tools)
            else:
                from nucleusiq.agents.errors import AgentConfigError

                raise AgentConfigError(
                    "Cannot execute in standard mode: LLM not available",
                    mode="standard",
                )

    def _get_tool_specs(self, agent: "Agent") -> list[dict[str, Any]]:
        """Return LLM-formatted tool specifications."""
        if agent.tools and agent.llm:
            return agent.llm.convert_tool_specs(agent.tools)
        return []

    async def _tool_call_loop(
        self,
        agent: "Agent",
        task: Task | dict[str, Any],
        messages: list[ChatMessage],
        tool_specs: list[dict[str, Any]],
        *,
        purpose_override: CallPurpose | None = None,
    ) -> Any:
        """Core tool-calling loop: LLM -> tool -> LLM -> ... -> final answer.

        ``purpose_override`` lets other roles (e.g. the Reviser in
        ``AutonomousMode``) reuse this engine while tagging every LLM
        call with a role-specific ``CallPurpose`` (``REFINER``, etc.).
        When ``None``, the default per-round classification is used
        (``MAIN`` for round 1, ``TOOL_LOOP`` thereafter).
        """
        max_tool_calls = agent.config.get_effective_max_tool_calls()
        tool_call_count = 0
        call_round = 0
        empty_retries_remaining = 2
        pre_synth_snapshot: list[ChatMessage] | None = None

        while tool_call_count < max_tool_calls:
            call_round += 1

            # Snapshot messages *before* call_llm (which runs
            # post_response and may mask tool results).  Synthesis
            # needs the full, unmasked context to generate output.
            if agent.config.enable_synthesis and tool_call_count > 0 and call_round > 2:
                pre_synth_snapshot = list(messages)

            if purpose_override is not None:
                purpose = purpose_override
            else:
                purpose = CallPurpose.MAIN if call_round == 1 else CallPurpose.TOOL_LOOP
            call_kwargs = self.build_call_kwargs(
                agent,
                messages,
                tool_specs or None,
                max_output_tokens=getattr(agent.config, "llm_max_output_tokens", 2048),
            )
            response = await self.call_llm(
                agent, call_kwargs, messages, tool_specs or None, purpose=purpose
            )

            structured = self.handle_structured_output(agent, response)
            if structured is not None:
                return structured

            self.validate_response(response)

            msg = response.choices[0].message
            tool_calls = self._get_tool_calls(msg)
            refusal = self._get_refusal(msg)
            content = self.extract_content(msg)

            if refusal:
                agent.state = AgentState.ERROR
                return f"Error: LLM refused request: {refusal}"

            if tool_calls:
                result = await self._process_tool_calls(
                    agent, msg, tool_calls, messages, tool_round=call_round
                )
                if result is not None:
                    return result
                # Auto-injected recall tools (memory operations) do not
                # consume the tool-call budget — see §6.4 of the v2
                # redesign.  The user's quota is for *external actions*.
                #
                # ``tool_calls`` here is still the raw provider-shape
                # list (OpenAI uses ``tc.function.name``); we route it
                # through ``_parse_tool_call`` so the recall check sees
                # the canonical name regardless of wire format.  Without
                # this, OpenAI-shaped recall calls would be counted
                # because ``getattr(tc, "name", None)`` returns ``None``.
                from nucleusiq.agents.context.workspace_tools import (
                    is_context_management_tool_name,
                )

                tool_call_count += sum(
                    1
                    for tc in tool_calls
                    if not is_context_management_tool_name(self._parse_tool_call(tc)[1])
                )
                continue

            if content:
                synth_threshold = getattr(agent.config, "synthesis_word_threshold", 500)
                if (
                    pre_synth_snapshot is not None
                    and len(content.split()) < synth_threshold
                ):
                    agent._logger.info(
                        "Synthesis pass: %d tool calls over %d rounds — "
                        "re-calling LLM without tools for final output",
                        tool_call_count,
                        call_round - 1,
                    )
                    synth = await self._synthesis_pass(agent, pre_synth_snapshot)
                    if synth.strip():
                        content = synth

                messages.append(ChatMessage(role="assistant", content=content))
                # F7 — terminal post_response symmetry.  Masks the last
                # round's tool results (which now have an assistant
                # after them) so Critic/Refiner see the same masked
                # conversation as every intermediate round.
                self._finalize_post_response(agent, messages)
                agent.state = AgentState.COMPLETED
                await self._store_in_memory(agent, task, content)
                return content

            if empty_retries_remaining > 0:
                empty_retries_remaining -= 1
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

            agent._logger.error("LLM returned no tool calls and no content after retry")
            agent.state = AgentState.ERROR
            objective = self.get_objective(task)
            return (
                f"Error: LLM did not respond. Task "
                f"'{objective[:80]}...' may require AUTONOMOUS mode "
                "for multi-step execution."
            )

        agent._logger.warning("Maximum tool calls (%d) reached", max_tool_calls)
        if agent.config.enable_synthesis and tool_call_count > 0:
            agent._logger.info(
                "Tool-call budget exhausted after %d calls — forcing "
                "tools-free synthesis from current compacted context",
                tool_call_count,
            )
            try:
                content = await self._synthesis_pass(agent, list(messages))
            except Exception as e:
                agent._logger.error("Synthesis after tool-call cap failed: %s", e)
                agent.state = AgentState.ERROR
                return (
                    f"Error: Maximum tool calls ({max_tool_calls}) reached; "
                    f"synthesis failed: {e}"
                )

            if content.strip():
                messages.append(ChatMessage(role="assistant", content=content))
                self._finalize_post_response(agent, messages)
                agent.state = AgentState.COMPLETED
                await self._store_in_memory(agent, task, content)
                return content

        agent.state = AgentState.ERROR
        return f"Error: Maximum tool calls ({max_tool_calls}) reached"

    # ------------------------------------------------------------------ #
    # Tool-call extraction helpers                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_tool_calls(msg: Any) -> list | None:
        """Extract tool_calls list from a message (dict or object)."""
        if isinstance(msg, dict):
            calls = msg.get("tool_calls")
        else:
            calls = getattr(msg, "tool_calls", None)
        if calls and isinstance(calls, list) and len(calls) > 0:
            return calls
        return None

    @staticmethod
    def _get_refusal(msg: Any) -> str | None:
        """Extract refusal string from a message (dict or object)."""
        if isinstance(msg, dict):
            return msg.get("refusal")
        return getattr(msg, "refusal", None)

    async def _process_tool_calls(
        self,
        agent: "Agent",
        msg: Any,
        tool_calls: list,
        messages: list[ChatMessage],
        *,
        tool_round: int = 1,
    ) -> str | None:
        """Execute tool calls and append results to the message list.

        Returns an error string if any tool fails (fire-and-forget),
        otherwise ``None`` (continue loop).

        Context Mgmt v2 — Step 4: when a tool declares
        ``idempotent=True`` and the same ``(tool_name, args)`` was
        already invoked in this execution, we short-circuit by
        appending a dedup banner instead of re-executing the tool.
        Non-idempotent tools (the default — weather, stock, news,
        live-data) are always re-executed.
        """
        from nucleusiq.tools.errors import ToolExecutionError

        raw_content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        parsed_calls = [ToolCallRequest.from_raw(tc) for tc in tool_calls]
        messages.append(
            ChatMessage(
                role="assistant",
                content=raw_content,
                tool_calls=parsed_calls,
            )
        )

        # Per-execution dedup cache.  Lazily initialised on the agent
        # so it spans all tool rounds in a single run() call but is
        # fresh for the next run().  Maps (tool_name, args_hash) →
        # original tool_call_id, so dedup banners can point the model
        # back at the canonical earlier result.
        dedup_cache: dict[tuple[str, str], str] = (
            getattr(agent, "_tool_dedup_cache", None) or {}
        )
        agent._tool_dedup_cache = dedup_cache

        for tc in parsed_calls:
            if not tc.name:
                agent._logger.warning("Tool call missing function name, skipping")
                continue

            agent._logger.info("Tool requested: %s", tc.name)

            tool = _get_tool_by_name(agent, tc.name)
            is_idempotent = bool(getattr(tool, "idempotent", False))

            args_hash = _hash_tool_args(tc.arguments or "")
            cache_key = (tc.name, args_hash)
            prior_call_id = dedup_cache.get(cache_key)

            if is_idempotent and prior_call_id is not None:
                # Short-circuit the duplicate — do NOT execute the tool.
                from nucleusiq.agents.context.compactor import _build_args_preview

                args_preview = _build_args_preview(
                    {"function": {"arguments": tc.arguments or "{}"}}
                )
                banner = _IDEMPOTENT_DEDUP_BANNER.format(
                    tool_name=tc.name,
                    args_preview=args_preview,
                    original_call_id=prior_call_id,
                )
                agent._logger.info(
                    "Tool dedup: %s with args_hash=%s already called "
                    "(original_call_id=%s) — returning short-circuit banner",
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
                continue

            try:
                tool_result = await self.call_tool(agent, tc, tool_round=tool_round)
                try:
                    tool_args = json.loads(tc.arguments) if tc.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}
                agent._activate_context_state_for_tool_result(
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    tool_result=tool_result,
                    tool_args=tool_args,
                )
                tool_result_str = tool_result_to_context_string(tool_result)

                # Context window management: compress large tool results
                engine = getattr(agent, "_context_engine", None)
                if engine is not None:
                    tool_result_str = engine.ingest_tool_result(
                        tool_result_str, tc.name
                    )

                messages.append(
                    ChatMessage(
                        role="tool",
                        name=tc.name,
                        tool_call_id=tc.id,
                        content=tool_result_str,
                    )
                )

                # Record this call for future dedup *only* if the tool
                # opted in.  Non-idempotent calls are tracked-not-deduped
                # (we don't store them, so duplicates always execute).
                if is_idempotent and tc.id is not None:
                    dedup_cache[cache_key] = tc.id
            except ToolExecutionError:
                raise
            except Exception as e:
                agent._logger.error("Tool execution failed: %s", e)
                agent.state = AgentState.ERROR
                return f"Error: Tool '{tc.name}' execution failed: {str(e)}"

        return None

    @staticmethod
    def _parse_tool_call(
        tool_call: Any,
    ) -> tuple:
        """Parse a single tool call into ``(id, name, arguments_str)``.

        Accepts both the flat canonical format ``{"id", "name", "arguments"}``
        and SDK objects with ``function`` attribute.
        """
        if isinstance(tool_call, dict):
            tc_id = tool_call.get("id")
            fn_info = tool_call.get("function")
            if isinstance(fn_info, dict):
                fn_name = fn_info.get("name")
                fn_args_str = fn_info.get("arguments", "{}")
            else:
                fn_name = tool_call.get("name")
                fn_args_str = tool_call.get("arguments", "{}")
        else:
            tc_id = getattr(tool_call, "id", None)
            fn_info = getattr(tool_call, "function", None)
            if fn_info is not None:
                fn_name = getattr(fn_info, "name", None)
                fn_args_str = getattr(fn_info, "arguments", "{}")
            else:
                fn_name = getattr(tool_call, "name", None)
                fn_args_str = getattr(tool_call, "arguments", "{}")
        return tc_id, fn_name, fn_args_str

    async def _synthesis_pass(
        self,
        agent: "Agent",
        messages: list[ChatMessage],
    ) -> str:
        """Final LLM call without tools to break mode inertia.

        After multiple rounds of tool calls the model tends to stay in
        a data-gathering mindset and returns a terse status update
        instead of the full deliverable.  Re-calling with the same
        messages but **no tool specs** AND a synthesis nudge forces the
        model into generation mode so it produces the full output.

        Context Mgmt v2 — §7: when a :class:`ContextEngine` is attached
        we pre-rehydrate recent evidence markers before the synthesis
        call, because the model cannot call ``recall_tool_result`` in
        a tools=None pass.  The rehydration is best-effort, fits in
        budget, and is silently skipped if anything goes wrong.
        """
        task_obj = getattr(agent, "_current_task", None) or {}
        task_text = (
            task_obj.get("objective", "")
            if isinstance(task_obj, dict)
            else str(task_obj or "")
        )
        synth_messages = agent._build_synthesis_messages_from_context(
            task=task_text or "Complete the requested task.",
            output_shape=(
                "Produce the COMPLETE, FULL-LENGTH deliverable exactly as "
                "described in the user's instructions."
            ),
        )
        if synth_messages is None:
            synth_messages = list(messages)
            synth_messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "All data gathering is complete. "
                        "Now produce the COMPLETE, FULL-LENGTH deliverable "
                        "exactly as described in your instructions. "
                        "Do not summarize — write the entire output."
                    ),
                )
            )

        # I4 — give the Generator the same rehydration the Critic /
        # Refiner already have.  The engine returns the message list
        # unchanged when nothing is offloaded (zero-overhead).
        engine = getattr(agent, "_context_engine", None)
        if engine is not None:
            try:
                synth_messages = engine.prepare_for_synthesis(synth_messages)
            except Exception as exc:
                agent._logger.debug(
                    "Synthesis rehydration skipped (fail-open): %s", exc
                )

        call_kwargs = self.build_call_kwargs(
            agent,
            synth_messages,
            None,
            max_output_tokens=getattr(agent.config, "llm_max_output_tokens", 2048),
        )
        response = await self.call_llm(
            agent,
            call_kwargs,
            synth_messages,
            None,
            purpose=CallPurpose.SYNTHESIS,
        )
        self.validate_response(response)
        synth_content = self.extract_content(response.choices[0].message) or ""
        if not synth_content.strip():
            agent._logger.warning(
                "Synthesis pass returned empty — preserving pre-synthesis content"
            )
        return synth_content

    async def _store_in_memory(self, agent: "Agent", task: Any, content: str) -> None:
        """Persist result in agent memory."""
        if agent.memory:
            try:
                await agent.memory.aadd_message("assistant", content)
            except Exception as e:
                agent._logger.warning("Failed to store in memory: %s", e)
