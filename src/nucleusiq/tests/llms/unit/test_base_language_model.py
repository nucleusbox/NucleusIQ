"""Tests for llms/base.py — BaseLanguageModel."""

from pydantic import BaseModel


class TestBaseLanguageModel:
    def _make_concrete(self, **kwargs):
        from nucleusiq.llms.base import BaseLanguageModel

        class ConcreteLM(BaseLanguageModel, BaseModel):
            pass

        return ConcreteLM(**kwargs)

    def test_metadata_defaults_to_none(self):
        lm = self._make_concrete()
        assert lm.metadata is None

    def test_metadata_can_be_set(self):
        lm = self._make_concrete(metadata={"key": "value"})
        assert lm.metadata == {"key": "value"}
