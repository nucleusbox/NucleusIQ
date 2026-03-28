"""Tests for nucleusiq_gemini._shared.model_config."""

from nucleusiq_gemini._shared.model_config import (
    get_context_window,
    get_max_output_tokens,
    get_model_info,
    supports_function_calling,
    supports_structured_output,
    supports_thinking,
)


class TestGetModelInfo:
    def test_exact_match(self):
        info = get_model_info("gemini-2.5-flash")
        assert info is not None
        assert info.context_window == 1_048_576

    def test_prefix_match(self):
        info = get_model_info("gemini-2.5-flash-preview-05-06")
        assert info is not None
        assert info.supports_thinking is True

    def test_unknown_model(self):
        info = get_model_info("unknown-model")
        assert info is None

    def test_empty_string(self):
        info = get_model_info("")
        assert info is None

    def test_case_insensitive(self):
        info = get_model_info("Gemini-2.5-Flash")
        assert info is not None

    def test_gemini_25_pro(self):
        info = get_model_info("gemini-2.5-pro")
        assert info is not None
        assert info.supports_thinking is True
        assert info.supports_code_execution is True

    def test_gemini_20_flash(self):
        info = get_model_info("gemini-2.0-flash")
        assert info is not None
        assert info.max_output_tokens == 8_192
        assert info.supports_thinking is False

    def test_gemini_15_pro(self):
        info = get_model_info("gemini-1.5-pro")
        assert info is not None
        assert info.context_window == 2_097_152


class TestGetContextWindow:
    def test_known_model(self):
        assert get_context_window("gemini-2.5-flash") == 1_048_576

    def test_unknown_model_default(self):
        assert get_context_window("unknown") == 1_048_576

    def test_15_pro(self):
        assert get_context_window("gemini-1.5-pro") == 2_097_152


class TestGetMaxOutputTokens:
    def test_25_model(self):
        assert get_max_output_tokens("gemini-2.5-flash") == 65_536

    def test_20_model(self):
        assert get_max_output_tokens("gemini-2.0-flash") == 8_192

    def test_unknown_default(self):
        assert get_max_output_tokens("unknown") == 8_192


class TestSupportsThinking:
    def test_25_flash(self):
        assert supports_thinking("gemini-2.5-flash") is True

    def test_25_pro(self):
        assert supports_thinking("gemini-2.5-pro") is True

    def test_20_flash(self):
        assert supports_thinking("gemini-2.0-flash") is False

    def test_unknown(self):
        assert supports_thinking("unknown") is False


class TestSupportsFunctionCalling:
    def test_all_known_support(self):
        models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ]
        for m in models:
            assert supports_function_calling(m) is True

    def test_unknown_default_true(self):
        assert supports_function_calling("unknown") is True


class TestSupportsStructuredOutput:
    def test_known_models(self):
        assert supports_structured_output("gemini-2.5-flash") is True

    def test_unknown_default_true(self):
        assert supports_structured_output("unknown") is True
