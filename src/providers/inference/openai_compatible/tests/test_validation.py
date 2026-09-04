"""Preflight reports — the messages an operator actually reads."""

from __future__ import annotations

from nucleusiq_openai_compatible.config import ConfigResolver
from nucleusiq_openai_compatible.validation import ValidationReport, build_report


def make_config(**overrides):
    kwargs = {
        "base_url": "http://gpu:8000/v1",
        "model": "gemma-4-27b-it",
        "engine": "vllm",
        "context_window": 32_768,
    }
    kwargs.update(overrides)
    return ConfigResolver.resolve(**kwargs)


def report(**overrides) -> ValidationReport:
    kwargs = {
        "config": make_config(),
        "reachable": True,
        "served_models": ("gemma-4-27b-it",),
        "probe_error": None,
        "model_found": True,
    }
    kwargs.update(overrides)
    return build_report(**kwargs)


class TestHealthy:
    def test_ok(self) -> None:
        assert report().ok
        assert report().errors == ()

    def test_carries_config_snapshot(self) -> None:
        assert report().config["model"] == "gemma-4-27b-it"

    def test_context_window_reported(self) -> None:
        result = report()
        assert result.context_window == 32_768
        assert result.context_window_source == "explicit"


class TestErrors:
    def test_unreachable_names_the_url_and_cause(self) -> None:
        result = report(reachable=False, probe_error="connection refused")
        assert not result.ok
        assert "http://gpu:8000/v1/models" in result.errors[0]
        assert "connection refused" in result.errors[0]
        assert "/v1 prefix" in result.errors[0]

    def test_unreachable_without_a_cause(self) -> None:
        assert not report(reachable=False, probe_error=None).ok

    def test_wrong_model_lists_what_is_served(self) -> None:
        result = report(model_found=False, served_models=("llama-3", "qwen3"))
        assert not result.ok
        assert "llama-3, qwen3" in result.errors[0], (
            "this is the whole point: turn an opaque 404 into the actual "
            "list of served models"
        )

    def test_wrong_model_with_no_list(self) -> None:
        result = report(model_found=False, served_models=())
        assert "(none reported)" in result.errors[0]

    def test_unreachable_takes_precedence_over_model(self) -> None:
        result = report(reachable=False, model_found=False, probe_error="down")
        assert len(result.errors) == 1, (
            "an unreachable server cannot also report a missing model"
        )


class TestWarnings:
    def test_default_context_window_warns(self) -> None:
        result = report(config=make_config(context_window=None, engine="generic"))
        assert any("context_window=<tokens>" in w for w in result.warnings)

    def test_explicit_window_does_not_warn(self) -> None:
        assert not any("context_window=" in w for w in report().warnings)

    def test_no_tool_support_warns(self) -> None:
        result = report(config=make_config(supports_tools=False))
        assert any("rely on tools will not work" in w for w in result.warnings)

    def test_warnings_do_not_fail_the_report(self) -> None:
        result = report(config=make_config(supports_tools=False))
        assert result.ok, "a warning is advice, not a failure"


class TestReasoningGuidance:
    def test_reasoning_on_incapable_engine_warns(self) -> None:
        config = make_config(engine="tgi", is_reasoning_model=True)
        result = report(config=config)
        assert any("--reasoning-parser" in w for w in result.warnings)

    def test_capable_engine_undeclared_gets_a_note(self) -> None:
        result = report(config=make_config(is_reasoning_model=False))
        assert any("is_reasoning_model=True" in n for n in result.notes)

    def test_missing_thinking_toggle_is_explained(self) -> None:
        result = report(config=make_config(is_reasoning_model=True))
        note = next(n for n in result.notes if "chat_template_kwargs" in n)
        assert "enable_thinking" in note
        assert "Qwen3" in note, (
            "Qwen3 defaults thinking on while Gemma defaults it off; the "
            "operator needs both halves of that"
        )

    def test_configured_thinking_gets_no_note(self) -> None:
        config = make_config(
            is_reasoning_model=True, chat_template_kwargs={"enable_thinking": True}
        )
        assert not any(
            "No chat_template_kwargs" in n for n in report(config=config).notes
        )


class TestNotes:
    def test_structured_output_conflict_noted(self) -> None:
        assert any("suppresses tool calls" in n for n in report().notes)

    def test_heuristic_counting_noted(self) -> None:
        assert any("4 chars/token" in n for n in report().notes)

    def test_engine_notes_included(self) -> None:
        result = report(config=make_config(engine="azure"))
        assert any("deployment" in n for n in result.notes)


class TestRender:
    def test_healthy_render(self) -> None:
        text = report().render()
        assert "validation: OK" in text
        assert "reachable          : True" in text
        assert "32768" in text

    def test_failed_render(self) -> None:
        text = report(reachable=False, probe_error="refused").render()
        assert "validation: FAILED" in text
        assert "error" in text

    def test_served_models_listed(self) -> None:
        text = report(served_models=("a", "b")).render()
        assert "a, b" in text

    def test_no_served_models_line_when_empty(self) -> None:
        text = report(served_models=(), model_found=True).render()
        assert "served models" not in text

    def test_every_category_rendered(self) -> None:
        config = make_config(
            context_window=None, supports_tools=False, engine="generic"
        )
        text = report(config=config, reachable=False, probe_error="x").render()
        assert "error" in text and "warning" in text and "note" in text

    def test_render_is_multiline_text(self) -> None:
        assert len(report().render().splitlines()) >= 4


class TestValueSemantics:
    def test_frozen(self) -> None:
        import pytest

        with pytest.raises(AttributeError):
            report().ok = False  # type: ignore[misc]

    def test_collections_are_tuples(self) -> None:
        result = report()
        assert isinstance(result.errors, tuple)
        assert isinstance(result.warnings, tuple)
        assert isinstance(result.notes, tuple)
