"""Token counting: exact tokenizer when declared, safe heuristic otherwise."""

from __future__ import annotations

import pytest
from nucleusiq_openai_compatible._shared import tokenizer as tokenizer_module
from nucleusiq_openai_compatible._shared.tokenizer import (
    HeuristicCounter,
    HFTokenizerCounter,
    TokenCounter,
    build_token_counter,
    tokenizer_backend_available,
)


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Stands in for ``tokenizers.Tokenizer``; one token per whitespace word."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name

    def encode(self, text: str) -> FakeEncoding:
        return FakeEncoding(list(range(len(text.split()))))


@pytest.fixture
def fake_tokenizers(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``tokenizers`` module so no model download happens."""
    import sys
    import types

    module = types.ModuleType("tokenizers")

    class Tokenizer:
        loaded: list[str] = []
        failure: Exception | None = None

        @classmethod
        def from_pretrained(cls, name: str) -> FakeTokenizer:
            if cls.failure is not None:
                raise cls.failure
            cls.loaded.append(name)
            return FakeTokenizer(name)

        @classmethod
        def from_file(cls, path: str) -> FakeTokenizer:
            if cls.failure is not None:
                raise cls.failure
            cls.loaded.append(path)
            return FakeTokenizer(path)

    Tokenizer.loaded = []
    Tokenizer.failure = None
    module.Tokenizer = Tokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    return Tokenizer


class TestHeuristic:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(HeuristicCounter(), TokenCounter)

    def test_method_label(self) -> None:
        assert HeuristicCounter().method == "heuristic"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("a" * 400, 100), ("a" * 4, 1), ("a" * 40, 10), ("a" * 1_000, 250)],
    )
    def test_four_chars_per_token(self, text: str, expected: int) -> None:
        assert HeuristicCounter().count(text) == expected

    def test_never_returns_zero(self) -> None:
        # Reporting 0 would let the budget planner treat content as free.
        assert HeuristicCounter().count("") == 1
        assert HeuristicCounter().count("hi") == 1

    def test_monotonic(self) -> None:
        counter = HeuristicCounter()
        assert counter.count("a" * 100) < counter.count("a" * 1_000)

    def test_matches_framework_default(self) -> None:
        from nucleusiq.llms.base_llm import BaseLLM

        text = "the quick brown fox jumps over the lazy dog" * 5
        assert HeuristicCounter().count(text) == max(1, len(text) // 4), (
            "budgets computed here must agree with BaseLLM.estimate_tokens "
            "when no tokenizer is declared"
        )
        assert BaseLLM is not None

    def test_repr(self) -> None:
        assert repr(HeuristicCounter()) == "HeuristicCounter()"

    def test_is_stateless(self) -> None:
        assert HeuristicCounter().__slots__ == ()


class TestBackendProbe:
    def test_returns_bool(self) -> None:
        assert isinstance(tokenizer_backend_available(), bool)

    def test_true_when_importable(self, fake_tokenizers) -> None:
        assert tokenizer_backend_available() is True

    def test_false_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args, **kwargs):
            if name == "tokenizers":
                raise ImportError("no module named tokenizers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert tokenizer_backend_available() is False


class TestHFTokenizerCounter:
    def test_counts_with_the_model_tokenizer(self, fake_tokenizers) -> None:
        counter = HFTokenizerCounter("google/gemma-4-27b-it")
        assert counter.count("one two three") == 3
        assert counter.method == "tokenizer"

    def test_loads_by_repo_id(self, fake_tokenizers) -> None:
        HFTokenizerCounter("google/gemma-4-27b-it")
        assert fake_tokenizers.loaded == ["google/gemma-4-27b-it"]

    def test_loads_local_json_from_file(self, fake_tokenizers) -> None:
        HFTokenizerCounter("/models/gemma/tokenizer.json")
        assert fake_tokenizers.loaded == ["/models/gemma/tokenizer.json"]

    def test_never_returns_zero(self, fake_tokenizers) -> None:
        assert HFTokenizerCounter("google/gemma-4-27b-it").count("") == 1

    def test_load_failure_is_actionable(self, fake_tokenizers) -> None:
        fake_tokenizers.failure = OSError("404 repo not found")
        with pytest.raises(ValueError) as exc:
            HFTokenizerCounter("nope/nope")
        message = str(exc.value)
        assert "404 repo not found" in message
        assert "omit tokenizer=" in message, (
            "the error must tell the operator how to proceed without it"
        )

    def test_load_failure_is_loud_not_silent(self, fake_tokenizers) -> None:
        fake_tokenizers.failure = OSError("offline")
        with pytest.raises(ValueError):
            HFTokenizerCounter("nope/nope")
        # A silent downgrade to the heuristic would corrupt budgets invisibly
        # for a caller who explicitly asked for exact counts.

    def test_missing_extra_names_the_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args, **kwargs):
            if name == "tokenizers":
                raise ImportError("no module named tokenizers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError) as exc:
            HFTokenizerCounter("google/gemma-4-27b-it")
        assert "[tokenizer]" in str(exc.value)

    def test_repr_names_the_tokenizer(self, fake_tokenizers) -> None:
        assert "gemma" in repr(HFTokenizerCounter("google/gemma-4-27b-it"))

    def test_satisfies_protocol(self, fake_tokenizers) -> None:
        assert isinstance(HFTokenizerCounter("google/gemma-4-27b-it"), TokenCounter)


class TestFactory:
    def test_none_gives_heuristic(self) -> None:
        assert isinstance(build_token_counter(None), HeuristicCounter)

    def test_empty_string_gives_heuristic(self) -> None:
        assert isinstance(build_token_counter(""), HeuristicCounter)

    def test_name_gives_exact_counter(self, fake_tokenizers) -> None:
        counter = build_token_counter("google/gemma-4-27b-it")
        assert isinstance(counter, HFTokenizerCounter)
        assert counter.method == "tokenizer"

    def test_module_exports(self) -> None:
        assert set(tokenizer_module.__all__) == {
            "HFTokenizerCounter",
            "HeuristicCounter",
            "TokenCounter",
            "build_token_counter",
            "tokenizer_backend_available",
        }
