"""``OpenAICompatibleLLM`` — generic OpenAI-compatible chat provider.

Orchestration only.  Payload construction, error translation, token
counting, streaming, capability resolution and validation each live in their
own module, and every one of them is injectable, so this class stays small
and the package is testable without patching the ``openai`` SDK.

**One instance describes one model on one endpoint.**  The framework sends
``model=getattr(llm, "model_name", ...)`` and calls ``get_context_window()``
once, with no model argument, to size the entire context budget — so a
per-call model switch would silently invalidate that budget.  To serve
several models, build several instances.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import openai
from nucleusiq.llms.base_llm import BaseLLM
from nucleusiq.llms.errors import InvalidRequestError, ModelNotFoundError
from nucleusiq.streaming.events import StreamEvent

from .._shared.model_probe import ModelProbe
from .._shared.redact import Redactor
from .._shared.response_models import NormalizedResponse, normalize_response
from .._shared.retry import PROVIDER, call_with_retry
from .._shared.tokenizer import (
    TokenCounter,
    build_token_counter,
    tokenizer_backend_available,
)
from .._shared.wire import PayloadBuilder
from ..auth import AuthStrategy, CredentialSource, build_auth
from ..config import ConfigResolver, ResolvedConfig
from ..llm_params import OpenAICompatibleLLMParams
from ..structured_output.inbound import normalize_response_format
from ..structured_output.policy import build_policy
from ..tools.converter import convert_tool_spec
from ..validation import ValidationReport, build_report
from .stream_adapter import StreamOutcome, stream_events

logger = logging.getLogger(__name__)

__all__ = ["OpenAICompatibleLLM"]

ENV_BASE_URL = "OPENAI_COMPATIBLE_BASE_URL"
ENV_MODEL = "OPENAI_COMPATIBLE_MODEL"

# The openai SDK requires a non-empty api_key even when the credential
# travels in a custom header or the server needs none at all.
_PLACEHOLDER_KEY = "EMPTY"


class OpenAICompatibleLLM(BaseLLM):
    """Chat provider for any OpenAI-compatible inference server.

    Args:
        base_url: Endpoint root including the ``/v1`` prefix, which is
            appended if omitted. Defaults to ``$OPENAI_COMPATIBLE_BASE_URL``.
        model_name: The served model name (vLLM's ``--served-model-name``, or
            the deployment name on Azure). Never parsed for heuristics.
            Also accepted as ``model=``. Defaults to
            ``$OPENAI_COMPATIBLE_MODEL``.
        api_key: A credential, or a sync/async callable returning one.
            Callables resolve per request, so rotation and per-tenant keys
            need no agent rebuild. Absence never raises — unauthenticated
            local servers are normal. Defaults to
            ``$OPENAI_COMPATIBLE_API_KEY``.
        auth: An :class:`~nucleusiq_openai_compatible.auth.AuthStrategy` for
            servers wanting the credential outside ``Authorization: Bearer``
            (Azure's ``api-key``, gateways' ``X-API-Key``). Mutually
            exclusive with *api_key*.
        context_window: Context length in tokens. **Recommended** — it
            avoids a probe round-trip and guarantees correct budgeting.
        engine: Capability preset; see
            :data:`~nucleusiq_openai_compatible.capabilities.ENGINE_PRESETS`.
        tokenizer: Hugging Face repo id or ``tokenizer.json`` path for exact
            token counts. Requires the ``[tokenizer]`` extra.
        probe_context_window: Read ``max_model_len`` from ``/v1/models``
            when *context_window* is not given.
        validate_model: Verify at first use that the server serves
            *model_name*, raising with the served list if it does not.
        probe: Injected :class:`ModelProbe` (tests).
        token_counter: Injected :class:`TokenCounter` (tests).

    Raises:
        nucleusiq.llms.errors.InvalidRequestError: Any Layer-1 validation
            failure — bad URL, unknown engine, implausible context window,
            conflicting credentials.

    Example:
        >>> llm = OpenAICompatibleLLM(  # doctest: +SKIP
        ...     base_url="http://gpu-node-1:8000/v1",
        ...     model_name="gemma-4-27b-it",
        ...     context_window=32_768,
        ...     engine="vllm",
        ... )
    """

    PROVIDER_NAME = "openai_compatible"
    """Distinct from ``"openai"``: the wire dialect is shared, but the cloud's
    guarantees are not, and support varies per engine."""

    NATIVE_TOOL_TYPES: frozenset = frozenset()
    """Empty: a self-hosted server executes no tools server-side, so no tool
    type can route away from Chat Completions."""

    @property
    def supports_native_structured_output(self) -> bool:
        """Whether this deployment can enforce a JSON schema server-side.

        Introspection for callers choosing an engine, and the switch the
        structured-output policy reads. Resolved per instance from the engine
        preset plus any explicit ``supports_json_schema=`` override, because
        the same wire dialect is served by engines that enforce schemas (vLLM,
        SGLang) and engines that do not (stock TGI, llama.cpp, Ollama's
        ``/v1`` shim).

        ``False`` does not disable structured output: the policy sends
        ``json_object`` with the schema injected into the prompt instead. Mode
        selection stays with the agent, which always resolves ``AUTO`` to
        NATIVE — degrading the transport is this adapter's job, not core's.
        """
        return self._config.supports_json_schema

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        *,
        model: str | None = None,
        api_key: CredentialSource | None = None,
        auth: AuthStrategy | None = None,
        engine: str = "generic",
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        tokenizer: str | None = None,
        supports_tools: bool | None = None,
        supports_json_schema: bool | None = None,
        supports_parallel_tool_calls: bool | None = None,
        max_tokens_field: str | None = None,
        structured_output_with_tools: str = "prompt",
        strict_capabilities: bool = False,
        is_reasoning_model: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        probe_context_window: bool = True,
        validate_model: bool = False,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, Any] | None = None,
        http_client: Any | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        async_mode: bool = True,
        temperature: float = 0.7,
        llm_params: OpenAICompatibleLLMParams | None = None,
        probe: ModelProbe | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__()

        resolved_model = _pick_model(model_name, model)
        self._auth = build_auth(auth=auth, api_key=api_key)
        self._llm_params = llm_params

        self.async_mode = async_mode
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        self._token_counter: TokenCounter = token_counter or build_token_counter(
            tokenizer
        )

        self._config: ResolvedConfig = ConfigResolver.resolve(
            base_url=base_url or os.getenv(ENV_BASE_URL, ""),
            model=resolved_model,
            engine=engine,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            supports_tools=supports_tools,
            supports_json_schema=supports_json_schema,
            supports_parallel_tool_calls=supports_parallel_tool_calls,
            max_tokens_field=max_tokens_field,
            structured_output_with_tools=structured_output_with_tools,
            strict_capabilities=strict_capabilities
            or bool(llm_params and llm_params.strict_capabilities),
            tokenizer=tokenizer,
            has_tokenizer_backend=(
                token_counter is not None or tokenizer_backend_available()
            ),
            is_reasoning_model=is_reasoning_model,
            chat_template_kwargs=chat_template_kwargs,
        )

        # The framework reads ``llm.model_name``; keep it authoritative.
        self.model_name = self._config.model
        self.base_url = self._config.base_url

        self._probe_enabled = probe_context_window
        self._validate_model = validate_model
        self._probed = False
        self._reasoning_warned = False

        self._default_headers = dict(default_headers or {})
        self._default_query = dict(default_query or {})
        self._http_client = http_client

        self._policy = build_policy(self._config.structured_output_with_tools)
        self._builder = PayloadBuilder(self._config)
        self._redactor = Redactor(header_names=list(self._auth.header_names))

        self._client = self._build_client()
        self._probe = probe or ModelProbe(self._build_client)
        self._last_stream: StreamOutcome | None = None

    # ------------------------------------------------------------------ #
    # Client construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_client(self) -> Any:
        """Create the SDK client.

        SDK-level retries are disabled: ``_shared.retry`` owns backoff so
        that rate-limit policy, error classification and logging are shared
        with every other NucleusIQ provider.
        """
        return openai.AsyncOpenAI(
            api_key=_PLACEHOLDER_KEY,
            base_url=self._config.base_url,
            timeout=self.timeout,
            max_retries=0,
            default_headers=self._default_headers or None,
            default_query=self._default_query or None,
            http_client=self._http_client,
        )

    async def _authorized_client(
        self, *, api_key: CredentialSource | None = None
    ) -> Any:
        """Return a client carrying freshly resolved credentials.

        ``with_options`` returns a shallow copy sharing the connection pool,
        so per-request and per-tenant credentials cost nothing and never
        mutate shared state.
        """
        strategy = build_auth(api_key=api_key) if api_key is not None else self._auth
        material = await strategy.resolve()

        self._redactor = Redactor(
            secrets=material.secrets,
            header_names=list(strategy.header_names),
        )

        if not material.headers and material.api_key is None:
            return self._client
        return self._client.with_options(
            api_key=material.api_key or _PLACEHOLDER_KEY,
            default_headers={**self._default_headers, **material.headers} or None,
        )

    # ------------------------------------------------------------------ #
    # Capabilities / introspection                                        #
    # ------------------------------------------------------------------ #

    @property
    def capabilities(self) -> ResolvedConfig:
        """The immutable resolved configuration for this instance."""
        return self._config

    @property
    def is_reasoning_model(self) -> bool:
        """Whether this model spends output tokens on internal thinking.

        Load-bearing for correctness, not just telemetry: the framework
        widens the token budget for internal calls (Critic, Refiner,
        Decomposer) when this is ``True``, because thinking and visible
        output share one completion budget.  Left ``False`` on a thinking
        model, those calls get truncated mid-thought and return nothing
        usable.

        Set explicitly via ``is_reasoning_model=True`` — it is never inferred
        from the model name, since a served name is arbitrary on self-hosted
        deployments.  If a response arrives with separated thinking while
        this is ``False``, a one-time warning is logged.
        """
        return self._config.is_reasoning_model

    def get_context_window(self) -> int:
        """Return the resolved context window in tokens.

        Sized from ``context_window=`` when given, else from a
        ``/v1/models`` probe performed on first use, else a conservative
        8192-token floor.  Because the framework may read this *before* the
        first call, passing ``context_window=`` explicitly is recommended.
        """
        return self._config.context_window

    def estimate_tokens(self, text: str) -> int:
        """Count tokens with the configured counter."""
        return self._token_counter.count(text)

    def _convert_tool_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        return convert_tool_spec(spec)

    def __repr__(self) -> str:
        # NEVER include credentials — secrets must not leak via repr / logs.
        return (
            f"{type(self).__name__}(base_url={self._config.base_url!r}, "
            f"model={self._config.model!r}, engine={self._config.engine!r}, "
            f"context_window={self._config.context_window}, auth=<redacted>)"
        )

    # ------------------------------------------------------------------ #
    # Probe / validation                                                  #
    # ------------------------------------------------------------------ #

    async def _ensure_probed(self) -> None:
        """Run the one-shot ``/v1/models`` probe, if it is still useful."""
        if self._probed:
            return
        self._probed = True

        needs_window = (
            self._probe_enabled and self._config.context_window_source == "default"
        )
        if not needs_window and not self._validate_model:
            return

        result = await self._probe.probe(model=self._config.model)

        if needs_window and result.context_window:
            self._config = self._config.with_context_window(
                result.context_window, "probe"
            )
            self._builder = PayloadBuilder(self._config)
            logger.info(
                "Discovered context window %d for model %r from /v1/models",
                result.context_window,
                self._config.model,
            )

        if (
            self._validate_model
            and result.reachable
            and not result.has_model(self._config.model)
        ):
            raise ModelNotFoundError.from_provider_error(
                provider=PROVIDER,
                message=(
                    f"Model {self._config.model!r} is not served by "
                    f"{self._config.base_url}. This server serves: "
                    f"{', '.join(result.model_ids) or '(none reported)'}."
                ),
                status_code=404,
            )

    async def validate(self) -> ValidationReport:
        """Preflight the endpoint and return a structured report.

        Never raises for a reachability or model-name problem — those are
        reported as errors on the report so a caller can print actionable
        guidance instead of handling an exception.
        """
        self._probed = True
        result = await self._probe.probe(model=self._config.model)

        if (
            self._config.context_window_source == "default"
            and self._probe_enabled
            and result.context_window
        ):
            self._config = self._config.with_context_window(
                result.context_window, "probe"
            )
            self._builder = PayloadBuilder(self._config)

        return build_report(
            config=self._config,
            reachable=result.reachable,
            served_models=result.model_ids,
            probe_error=result.error,
            model_found=result.has_model(self._config.model),
        )

    # ------------------------------------------------------------------ #
    # Request assembly                                                    #
    # ------------------------------------------------------------------ #

    def _merge_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Layer explicit call kwargs over configured ``llm_params``."""
        merged: dict[str, Any] = {}
        if self._llm_params is not None:
            merged.update(self._llm_params.to_call_kwargs())
        merged.update(kwargs)
        return merged

    def _apply_schema(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        schema: dict[str, Any] | None,
        schema_name: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
        """Apply the structured-output policy to messages and payload."""
        if not schema:
            return messages, None, None

        decision = self._policy.decide(
            schema=schema,
            schema_name=schema_name,
            has_tools=bool(tools),
            supports_json_schema=self._config.supports_json_schema,
            suppresses_tools=self._config.structured_output_suppresses_tools,
        )
        if decision.prompt_instruction:
            messages = _inject_instruction(messages, decision.prompt_instruction)
        return messages, decision.response_format, decision.reason

    def _prepare(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any | None,
        max_output_tokens: int | None,
        temperature: float | None,
        stop: list[str] | None,
        stream: bool,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Build the wire payload for one call."""
        merged = self._merge_params(kwargs)
        schema = merged.pop("response_schema", None)
        schema_name = merged.pop("response_schema_name", "response")

        # The core agent delivers structured output as a ``response_format``
        # kwarg. Left in ``merged`` it would bypass the policy and be sent
        # verbatim alongside tools, which is exactly the combination that
        # suppresses tool calls on vLLM.
        passthrough_format: dict[str, Any] | None = None
        if "response_format" in merged:
            inbound = normalize_response_format(merged.pop("response_format"))
            if inbound is not None:
                if inbound.schema is not None and schema is None:
                    schema, schema_name = inbound.schema, inbound.name
                elif inbound.passthrough is not None:
                    passthrough_format = inbound.passthrough

        converted = [convert_tool_spec(t) for t in tools] if tools else None
        messages, response_format, reason = self._apply_schema(
            messages=messages,
            tools=converted,
            schema=schema,
            schema_name=schema_name,
        )

        payload = self._builder.build(
            messages=messages,
            tools=converted,
            tool_choice=tool_choice if converted else None,
            max_output_tokens=max_output_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            stop=stop,
            response_format=response_format or passthrough_format,
            stream=stream,
            extra=merged,
        )
        return payload, reason

    # ------------------------------------------------------------------ #
    # Non-streaming call                                                  #
    # ------------------------------------------------------------------ #

    async def call(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = "auto",
        max_output_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """Send a Chat Completions request and return a normalized response.

        Args:
            model: Ignored beyond a consistency check — one instance serves
                one model. A mismatch raises rather than silently using a
                different model than the context budget was sized for.
            messages: Chat messages.
            tools: Tool specs; converted to OpenAI function tools.
            tool_choice: Tool selection mode. Omitted when no tools.
            max_output_tokens: Output cap, emitted under the configured wire
                field name.
            stop: Stop sequences.
            **kwargs: Provider params (``seed``, ``user``, ``extra_body``,
                ``parallel_tool_calls``, ``response_schema``, …). Unsupported
                and OpenAI-only keys are stripped.

        Raises:
            nucleusiq.llms.errors.LLMError: Mapped, credential-free errors.
        """
        _check_model(model, self._config.model)
        await self._ensure_probed()

        payload, reason = self._prepare(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop=stop,
            stream=False,
            kwargs={
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                **kwargs,
            },
        )

        client = await self._authorized_client(api_key=kwargs.get("api_key"))
        raw = await call_with_retry(
            lambda: client.chat.completions.create(**payload),
            max_retries=self.max_retries,
            async_mode=self.async_mode,
            logger=logger,
            redactor=self._redactor,
        )

        response = normalize_response(raw)
        if not response.usage_reported:
            logger.debug("Server reported no usage; token counts are estimates.")
        if reason:
            logger.debug("Structured-output policy applied: %s", reason)
        self._inspect_reasoning(response)
        return response

    def _inspect_reasoning(self, response: NormalizedResponse) -> None:
        """Surface thinking-mode misconfiguration as actionable warnings."""
        if response.reasoning_only:
            logger.warning(
                "The server returned thinking output but no answer content "
                "(reasoning=%d chars, content empty). This usually means the "
                "model's chat template and the server's --reasoning-parser "
                "disagree about whether thinking is on. Pass the thinking "
                "toggle explicitly, e.g. "
                "chat_template_kwargs={'enable_thinking': True} for Qwen3 or "
                "Gemma, or {'thinking': True} for Granite and DeepSeek.",
                len(response.reasoning or ""),
            )
        elif (
            response.has_reasoning
            and not self._config.is_reasoning_model
            and not self._reasoning_warned
        ):
            self._reasoning_warned = True
            logger.warning(
                "This model returned separated thinking output but was not "
                "declared as a reasoning model. Pass is_reasoning_model=True "
                "so internal agent calls (Critic, Refiner, Decomposer) get a "
                "token budget that covers thinking as well as the answer."
            )

    # ------------------------------------------------------------------ #
    # Streaming call                                                      #
    # ------------------------------------------------------------------ #

    async def call_stream(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = "auto",
        max_output_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a response as ``StreamEvent`` objects.

        Yields a ``TOKEN`` event per content delta, then a ``COMPLETE``
        event with the accumulated text. Fragmented tool-call deltas are
        merged into whole calls, and usage-only trailer chunks are handled.
        """
        _check_model(model, self._config.model)
        await self._ensure_probed()

        payload, _ = self._prepare(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
            kwargs={
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                **kwargs,
            },
        )

        client = await self._authorized_client(api_key=kwargs.get("api_key"))
        chunks = await call_with_retry(
            lambda: client.chat.completions.create(**payload),
            max_retries=self.max_retries,
            async_mode=self.async_mode,
            logger=logger,
            redactor=self._redactor,
        )

        outcome = StreamOutcome()
        async for event in stream_events(
            chunks, outcome=outcome, model=self._config.model
        ):
            yield event

        self._last_stream = outcome
        if outcome.response is not None:
            self._inspect_reasoning(outcome.response)

    @property
    def last_stream(self) -> StreamOutcome | None:
        """Accumulated result of the most recent :meth:`call_stream`.

        Streams are single-use, so tool calls, usage and thinking output from
        the last stream are held here for the caller to record afterwards.
        """
        return self._last_stream


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #


def _pick_model(model_name: str | None, model: str | None) -> str:
    """Reconcile the ``model_name`` / ``model`` aliases.

    ``model_name`` matches every other NucleusIQ provider; ``model`` matches
    the OpenAI SDK and vLLM CLI. Both are accepted, and disagreement is an
    error rather than a silent preference.
    """
    if model_name and model and model_name != model:
        raise InvalidRequestError.from_provider_error(
            provider=PROVIDER,
            message=(
                f"model_name={model_name!r} and model={model!r} disagree. "
                "Pass only one — they are aliases."
            ),
        )
    return model_name or model or os.getenv(ENV_MODEL, "")


def _check_model(requested: str | None, configured: str) -> None:
    """Reject a per-call model switch.

    ``get_context_window()`` is per-instance and the framework reads it once,
    so honouring a different model here would leave the context budget sized
    for the wrong window.
    """
    if requested and requested != configured:
        raise InvalidRequestError.from_provider_error(
            provider=PROVIDER,
            message=(
                f"This provider instance is configured for model "
                f"{configured!r} but the call requested {requested!r}. One "
                "instance serves exactly one model, because the context "
                "budget is sized from its context window. Construct a "
                "second OpenAICompatibleLLM for the other model."
            ),
        )


def _inject_instruction(
    messages: list[dict[str, Any]], instruction: str
) -> list[dict[str, Any]]:
    """Append *instruction* to the system message, or prepend a new one."""
    out = [dict(m) for m in messages]
    for message in out:
        if message.get("role") == "system":
            existing = message.get("content")
            message["content"] = (
                f"{existing}\n\n{instruction}"
                if isinstance(existing, str)
                else instruction
            )
            return out
    return [{"role": "system", "content": instruction}, *out]
