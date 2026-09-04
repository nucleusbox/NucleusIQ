"""``/v1/models`` discovery — never fatal, always cached."""

from __future__ import annotations

from typing import Any

from nucleusiq_openai_compatible._shared.model_probe import ModelProbe, ProbeResult


class _Models:
    def __init__(self, cards: list[Any], error: Exception | None = None) -> None:
        self._cards = cards
        self._error = error
        self.calls = 0

    async def list(self) -> Any:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return type("Listing", (), {"data": self._cards})()


class _Client:
    def __init__(self, cards: list[Any], error: Exception | None = None) -> None:
        self.models = _Models(cards, error)


def probe_for(cards: list[Any], error: Exception | None = None) -> ModelProbe:
    return ModelProbe(lambda: _Client(cards, error))


class TestSuccess:
    async def test_reads_ids_and_window(self) -> None:
        result = await probe_for(
            [{"id": "gemma", "max_model_len": 32_768}, {"id": "qwen"}]
        ).probe(model="gemma")
        assert result.reachable
        assert result.model_ids == ("gemma", "qwen")
        assert result.context_window == 32_768

    async def test_window_taken_from_the_requested_model(self) -> None:
        result = await probe_for(
            [
                {"id": "small", "max_model_len": 4_096},
                {"id": "big", "max_model_len": 128_000},
            ]
        ).probe(model="big")
        assert result.context_window == 128_000

    async def test_sole_model_card_accepted_despite_name_mismatch(self) -> None:
        # Servers often serve under a local path rather than the name passed.
        result = await probe_for(
            [{"id": "/models/gemma-4", "max_model_len": 32_768}]
        ).probe(model="gemma-4-27b-it")
        assert result.context_window == 32_768

    async def test_attribute_style_cards(self) -> None:
        card = type("Card", (), {"id": "gemma", "max_model_len": 8_192})()
        result = await probe_for([card]).probe(model="gemma")
        assert result.model_ids == ("gemma",)
        assert result.context_window == 8_192

    async def test_alternate_field_names(self) -> None:
        for field in ("max_context_length", "context_length", "max_sequence_length"):
            result = await probe_for([{"id": "m", field: 16_384}]).probe(model="m")
            assert result.context_window == 16_384, field

    async def test_nested_metadata(self) -> None:
        result = await probe_for(
            [{"id": "m", "meta": {"max_model_len": 65_536}}, {"id": "other"}]
        ).probe(model="m")
        assert result.context_window == 65_536

    async def test_no_window_published(self) -> None:
        result = await probe_for([{"id": "a"}, {"id": "b"}]).probe(model="a")
        assert result.reachable
        assert result.context_window is None

    async def test_invalid_window_values_ignored(self) -> None:
        for bad in (0, -1, True, "32768", None):
            result = await probe_for([{"id": "m", "max_model_len": bad}]).probe(
                model="m"
            )
            assert result.context_window is None, bad

    async def test_model_dump_cards(self) -> None:
        class Card:
            id = "m"
            max_model_len = 2_048

            def model_dump(self) -> dict[str, Any]:
                return {"id": "m", "max_model_len": 2_048}

        result = await probe_for([Card()]).probe(model="m")
        assert result.raw == {"id": "m", "max_model_len": 2_048}

    async def test_model_dump_failure_is_tolerated(self) -> None:
        class Card:
            id = "m"
            max_model_len = 2_048

            def model_dump(self) -> dict[str, Any]:
                raise RuntimeError("boom")

        result = await probe_for([Card()]).probe(model="m")
        assert result.context_window == 2_048
        assert result.raw == {}


class TestFailure:
    async def test_never_raises(self) -> None:
        result = await probe_for([], error=RuntimeError("connection refused")).probe()
        assert result.reachable is False
        assert "connection refused" in (result.error or "")

    async def test_failure_logged_at_debug_only(self, caplog) -> None:
        with caplog.at_level("DEBUG"):
            await probe_for([], error=RuntimeError("down")).probe()
        assert "non-fatal" in caplog.text
        assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]

    async def test_client_construction_failure_handled(self) -> None:
        def factory():
            raise OSError("no route to host")

        result = await ModelProbe(factory).probe()
        assert result.reachable is False


class TestCaching:
    async def test_probe_runs_once(self) -> None:
        client = _Client([{"id": "m", "max_model_len": 4_096}])
        probe = ModelProbe(lambda: client)
        await probe.probe(model="m")
        await probe.probe(model="m")
        assert client.models.calls == 1, (
            "an agent run must not pay a round-trip per call"
        )

    async def test_cached_property(self) -> None:
        probe = probe_for([{"id": "m"}])
        assert probe.cached is None
        await probe.probe(model="m")
        assert probe.cached is not None

    async def test_failures_are_cached_too(self) -> None:
        client = _Client([], error=RuntimeError("down"))
        probe = ModelProbe(lambda: client)
        await probe.probe()
        await probe.probe()
        assert client.models.calls == 1


class TestHasModel:
    def test_present(self) -> None:
        assert ProbeResult(reachable=True, model_ids=("a", "b")).has_model("a")

    def test_absent(self) -> None:
        assert not ProbeResult(reachable=True, model_ids=("a",)).has_model("z")

    def test_empty_list_is_permissive(self) -> None:
        assert ProbeResult(reachable=True, model_ids=()).has_model("anything"), (
            "an uninformative probe must never block a working configuration"
        )
