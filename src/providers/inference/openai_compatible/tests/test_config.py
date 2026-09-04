"""URL normalization, the precedence chain, and Layer-1 validation."""

from __future__ import annotations

from dataclasses import replace

import pytest
from nucleusiq.llms.errors import InvalidRequestError
from nucleusiq_openai_compatible.capabilities import (
    DEFAULT_CONTEXT_WINDOW,
    get_engine_profile,
)
from nucleusiq_openai_compatible.config import (
    ConfigResolver,
    normalize_base_url,
)


class TestNormalizeBaseUrl:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("http://gpu:8000", "http://gpu:8000/v1"),
            ("http://gpu:8000/", "http://gpu:8000/v1"),
            ("http://gpu:8000/v1", "http://gpu:8000/v1"),
            ("http://gpu:8000/v1/", "http://gpu:8000/v1"),
            ("https://api.together.xyz/v1", "https://api.together.xyz/v1"),
            (
                "https://r.openai.azure.com/openai/v1/",
                "https://r.openai.azure.com/openai/v1",
            ),
            ("http://127.0.0.1:11434", "http://127.0.0.1:11434/v1"),
        ],
    )
    def test_appends_v1_when_missing(self, given: str, expected: str) -> None:
        assert normalize_base_url(given) == expected

    def test_preserves_gateway_prefix(self) -> None:
        assert (
            normalize_base_url("https://gw.corp/llm/proxy")
            == "https://gw.corp/llm/proxy/v1"
        )

    def test_preserves_query_string(self) -> None:
        assert "api-version=preview" in normalize_base_url(
            "https://r.openai.azure.com/openai/v1?api-version=preview"
        )

    @pytest.mark.parametrize("given", ["", "   ", None, 42])
    def test_rejects_empty(self, given: object) -> None:
        with pytest.raises(InvalidRequestError, match="base_url is required"):
            normalize_base_url(given)  # type: ignore[arg-type]

    @pytest.mark.parametrize("given", ["ftp://host/v1", "gpu-node-1:8000", "//host"])
    def test_rejects_bad_scheme(self, given: str) -> None:
        with pytest.raises(InvalidRequestError, match="http:// or https://"):
            normalize_base_url(given)

    def test_rejects_missing_host(self) -> None:
        with pytest.raises(InvalidRequestError, match="missing a host"):
            normalize_base_url("http:///v1")


class TestContextWindowPrecedence:
    """explicit -> probe -> engine preset -> conservative default."""

    def test_explicit_wins(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1",
            model="m",
            context_window=32_768,
            probed_context_window=8_192,
        )
        assert (cfg.context_window, cfg.context_window_source) == (32_768, "explicit")

    def test_probe_used_when_no_explicit(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1", model="m", probed_context_window=40_960
        )
        assert (cfg.context_window, cfg.context_window_source) == (40_960, "probe")

    def test_engine_preset_before_default(self, monkeypatch) -> None:
        preset = replace(get_engine_profile("vllm"), default_context_window=16_384)
        monkeypatch.setattr(
            "nucleusiq_openai_compatible.config.get_engine_profile",
            lambda _: preset,
        )
        cfg = ConfigResolver.resolve(base_url="http://x:1", model="m", engine="vllm")
        assert (cfg.context_window, cfg.context_window_source) == (16_384, "engine")

    def test_falls_back_conservatively_with_warning(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            cfg = ConfigResolver.resolve(base_url="http://x:1", model="mystery")
        assert (cfg.context_window, cfg.context_window_source) == (
            DEFAULT_CONTEXT_WINDOW,
            "default",
        )
        assert "8192" in caplog.text and "mystery" in caplog.text

    def test_default_is_not_openai_128k(self) -> None:
        assert DEFAULT_CONTEXT_WINDOW == 8_192, (
            "over-reporting the window makes the context engine skip "
            "compaction and the server reject the request"
        )

    def test_with_context_window_returns_a_copy(self) -> None:
        cfg = ConfigResolver.resolve(base_url="http://x:1", model="m")
        updated = cfg.with_context_window(65_536, "probe")
        assert (updated.context_window, updated.context_window_source) == (
            65_536,
            "probe",
        )
        assert cfg.context_window == DEFAULT_CONTEXT_WINDOW, "original must be frozen"


class TestCapabilityOverrides:
    def test_preset_supplies_defaults(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1", model="m", engine="vllm", context_window=1_024
        )
        assert cfg.supports_tools
        assert cfg.supports_json_schema
        assert cfg.extra_body_allowed
        assert cfg.structured_output_suppresses_tools

    def test_explicit_override_beats_preset(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1",
            model="m",
            engine="vllm",
            context_window=1_024,
            supports_tools=False,
            supports_json_schema=False,
        )
        assert not cfg.supports_tools
        assert not cfg.supports_json_schema

    def test_generic_is_conservative(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1", model="m", context_window=1_024
        )
        assert not cfg.supports_json_schema
        assert not cfg.extra_body_allowed
        assert not cfg.supports_reasoning


class TestLayerOneValidation:
    @pytest.mark.parametrize("model", ["", "   ", None])
    def test_model_required(self, model: object) -> None:
        with pytest.raises(InvalidRequestError, match="model is required"):
            ConfigResolver.resolve(base_url="http://x:1", model=model)  # type: ignore[arg-type]

    def test_model_is_stripped(self) -> None:
        cfg = ConfigResolver.resolve(base_url="http://x:1", model="  gemma  ")
        assert cfg.model == "gemma"

    def test_unknown_engine_lists_valid_names(self) -> None:
        with pytest.raises(InvalidRequestError) as exc:
            ConfigResolver.resolve(base_url="http://x:1", model="m", engine="vlm")
        message = str(exc.value)
        assert "vllm" in message and "generic" in message

    @pytest.mark.parametrize("window", [10, 0, -1, 10**9])
    def test_implausible_window_rejected(self, window: int) -> None:
        with pytest.raises(InvalidRequestError, match="plausible range"):
            ConfigResolver.resolve(
                base_url="http://x:1", model="m", context_window=window
            )

    @pytest.mark.parametrize("window", [True, "32768", 3.5])
    def test_non_integer_window_rejected(self, window: object) -> None:
        with pytest.raises(InvalidRequestError, match="integer number of tokens"):
            ConfigResolver.resolve(
                base_url="http://x:1",
                model="m",
                context_window=window,  # type: ignore[arg-type]
            )

    def test_unknown_structured_output_mode(self) -> None:
        with pytest.raises(InvalidRequestError, match="structured_output_with_tools"):
            ConfigResolver.resolve(
                base_url="http://x:1",
                model="m",
                context_window=1_024,
                structured_output_with_tools="inject",
            )

    def test_bad_max_tokens_field(self) -> None:
        with pytest.raises(InvalidRequestError, match="max_tokens_field"):
            ConfigResolver.resolve(
                base_url="http://x:1",
                model="m",
                context_window=1_024,
                max_tokens_field="maxTokens",
            )

    def test_max_completion_tokens_accepted(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1",
            model="m",
            context_window=1_024,
            max_tokens_field="max_completion_tokens",
        )
        assert cfg.max_tokens_field == "max_completion_tokens"

    @pytest.mark.parametrize("value", [0, -5, True, "500"])
    def test_bad_max_output_tokens(self, value: object) -> None:
        with pytest.raises(InvalidRequestError, match="positive integer"):
            ConfigResolver.resolve(
                base_url="http://x:1",
                model="m",
                context_window=4_096,
                max_output_tokens=value,  # type: ignore[arg-type]
            )

    def test_max_output_cannot_exceed_window(self) -> None:
        with pytest.raises(InvalidRequestError, match="exceeds the context window"):
            ConfigResolver.resolve(
                base_url="http://x:1",
                model="m",
                context_window=4_096,
                max_output_tokens=8_192,
            )


class TestReasoningConfig:
    def test_flags_come_from_preset(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1", model="m", engine="vllm", context_window=1_024
        )
        assert cfg.supports_reasoning
        assert cfg.supports_reasoning_effort

    def test_declaring_reasoning_on_incapable_engine_warns(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            cfg = ConfigResolver.resolve(
                base_url="http://x:1",
                model="m",
                engine="tgi",
                context_window=1_024,
                is_reasoning_model=True,
            )
        assert cfg.is_reasoning_model, "the flag is still honoured"
        assert "--reasoning-parser" in caplog.text

    def test_strict_mode_raises_instead(self) -> None:
        with pytest.raises(InvalidRequestError, match="reasoning-parser"):
            ConfigResolver.resolve(
                base_url="http://x:1",
                model="m",
                engine="tgi",
                context_window=1_024,
                is_reasoning_model=True,
                strict_capabilities=True,
            )

    def test_chat_template_kwargs_are_copied(self) -> None:
        source = {"enable_thinking": True}
        cfg = ConfigResolver.resolve(
            base_url="http://x:1",
            model="m",
            engine="vllm",
            context_window=1_024,
            chat_template_kwargs=source,
        )
        source["enable_thinking"] = False
        assert cfg.chat_template_kwargs == {"enable_thinking": True}


class TestTokenCountMethod:
    def test_heuristic_without_tokenizer(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1", model="m", context_window=1_024
        )
        assert cfg.token_count_method == "heuristic"

    def test_tokenizer_when_backend_present(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1",
            model="m",
            context_window=1_024,
            tokenizer="google/gemma-4-27b-it",
            has_tokenizer_backend=True,
        )
        assert cfg.token_count_method == "tokenizer"

    def test_heuristic_when_backend_missing(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1",
            model="m",
            context_window=1_024,
            tokenizer="google/gemma-4-27b-it",
            has_tokenizer_backend=False,
        )
        assert cfg.token_count_method == "heuristic"


class TestSummary:
    def test_summary_is_json_friendly(self) -> None:
        cfg = ConfigResolver.resolve(
            base_url="http://x:1", model="m", engine="vllm", context_window=2_048
        )
        summary = cfg.summary()
        assert summary["context_window_source"] == "explicit"
        assert summary["model"] == "m"
        assert set(summary) >= {"base_url", "engine", "supports_reasoning"}
