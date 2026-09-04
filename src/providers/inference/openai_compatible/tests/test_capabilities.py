"""Engine presets and capability gating."""

from __future__ import annotations

import pytest
from nucleusiq_openai_compatible.capabilities import (
    ENGINE_PRESETS,
    MAX_TOKENS_FIELDS,
    REASONING_FIELDS,
    THINKING_TEMPLATE_KWARGS,
    check_parallel_tool_calls,
    get_engine_profile,
    known_engines,
)


class TestRegistry:
    def test_every_documented_engine_present(self) -> None:
        expected = {
            "vllm",
            "sglang",
            "tgi",
            "llamacpp",
            "lmstudio",
            "ollama",
            "nim",
            "openrouter",
            "together",
            "fireworks",
            "deepinfra",
            "databricks",
            "litellm",
            "azure",
            "generic",
        }
        assert set(known_engines()) == expected

    def test_known_engines_sorted(self) -> None:
        assert list(known_engines()) == sorted(known_engines())

    def test_preset_key_matches_its_name(self) -> None:
        for key, profile in ENGINE_PRESETS.items():
            assert profile.name == key

    def test_every_preset_uses_a_legal_tokens_field(self) -> None:
        for profile in ENGINE_PRESETS.values():
            assert profile.max_tokens_field in MAX_TOKENS_FIELDS

    def test_presets_are_immutable(self) -> None:
        with pytest.raises(AttributeError):
            get_engine_profile("vllm").supports_tools = False  # type: ignore[misc]

    def test_lookup_is_case_insensitive_and_trimmed(self) -> None:
        assert get_engine_profile("  VLLM  ").name == "vllm"

    def test_unknown_engine_lists_alternatives(self) -> None:
        with pytest.raises(ValueError) as exc:
            get_engine_profile("vllm-server")
        message = str(exc.value)
        assert "vllm" in message and "generic" in message

    @pytest.mark.parametrize("engine", ["", "   ", None, 5])
    def test_blank_engine_rejected(self, engine: object) -> None:
        with pytest.raises(ValueError, match="valid values"):
            get_engine_profile(engine)  # type: ignore[arg-type]


class TestPresetContent:
    def test_vllm_profile(self) -> None:
        vllm = get_engine_profile("vllm")
        assert vllm.context_probe_field == "max_model_len"
        assert vllm.extra_body_allowed
        assert vllm.structured_output_suppresses_tools
        assert vllm.supports_reasoning
        assert vllm.supports_reasoning_effort

    def test_generic_is_the_conservative_default(self) -> None:
        generic = get_engine_profile("generic")
        assert not generic.supports_json_schema
        assert not generic.extra_body_allowed
        assert not generic.supports_reasoning
        assert not generic.supports_reasoning_effort

    def test_azure_notes_explain_its_quirks(self) -> None:
        notes = get_engine_profile("azure").notes
        assert "/openai/v1" in notes
        assert "deployment" in notes
        assert "api-key" in notes

    def test_every_preset_has_operator_notes(self) -> None:
        for name, profile in ENGINE_PRESETS.items():
            assert profile.notes.strip(), f"{name} should carry guidance"

    def test_reasoning_effort_implies_reasoning(self) -> None:
        for name, profile in ENGINE_PRESETS.items():
            if profile.supports_reasoning_effort:
                assert profile.supports_reasoning, (
                    f"{name} accepts reasoning_effort so it must also be able "
                    "to surface reasoning"
                )

    def test_reasoning_field_order(self) -> None:
        assert REASONING_FIELDS == ("reasoning", "reasoning_content"), (
            "newer vLLM emits 'reasoning'; the legacy name must remain a "
            "fallback, not the primary"
        )

    def test_thinking_kwarg_map_documents_both_spellings(self) -> None:
        assert set(THINKING_TEMPLATE_KWARGS.values()) == {
            "enable_thinking",
            "thinking",
        }


class TestParallelToolCallGate:
    def test_passthrough_when_supported(self) -> None:
        assert check_parallel_tool_calls(
            requested=True, supported=True, engine="vllm", strict=False
        )

    def test_not_requested_is_never_gated(self) -> None:
        assert not check_parallel_tool_calls(
            requested=False, supported=False, engine="generic", strict=True
        )

    def test_warns_and_disables_when_unsupported(self) -> None:
        with pytest.warns(UserWarning, match="does not declare support"):
            forward = check_parallel_tool_calls(
                requested=True, supported=False, engine="generic", strict=False
            )
        assert forward is False

    def test_raises_under_strict(self) -> None:
        with pytest.raises(ValueError, match="parallel_tool_calls"):
            check_parallel_tool_calls(
                requested=True, supported=False, engine="generic", strict=True
            )

    def test_message_names_the_remedy(self) -> None:
        with pytest.raises(ValueError) as exc:
            check_parallel_tool_calls(
                requested=True, supported=False, engine="tgi", strict=True
            )
        assert "supports_parallel_tool_calls=True" in str(exc.value)
