"""Tests for llms/base.py — BaseLanguageModel and tokenizer helpers."""

import pytest
from unittest.mock import patch, MagicMock


class TestBaseLanguageModel:

    def test_get_token_ids_custom(self):
        from nucleusiq.llms.base import BaseLanguageModel

        class ConcreteLM(BaseLanguageModel):
            pass

        lm = ConcreteLM()
        lm.custom_get_token_ids = lambda t: [1, 2, 3]
        assert lm.get_token_ids("hello") == [1, 2, 3]

    def test_get_num_tokens_custom(self):
        from nucleusiq.llms.base import BaseLanguageModel

        class ConcreteLM(BaseLanguageModel):
            pass

        lm = ConcreteLM()
        lm.custom_get_token_ids = lambda t: list(range(len(t)))
        assert lm.get_num_tokens("abcde") == 5

    def test_get_token_ids_default(self):
        """Default tokenizer (GPT-2) if available, else ImportError."""
        from nucleusiq.llms.base import BaseLanguageModel

        class ConcreteLM(BaseLanguageModel):
            pass

        lm = ConcreteLM()
        lm.custom_get_token_ids = None
        try:
            ids = lm.get_token_ids("hello world")
            assert isinstance(ids, list)
            assert len(ids) > 0
        except ImportError:
            pytest.skip("tokenizers not installed")

    def test_get_tokenizer_import_error(self):
        from nucleusiq.llms.base import get_tokenizer
        get_tokenizer.cache_clear()
        try:
            result = get_tokenizer()
            assert result is not None
        except ImportError:
            pytest.skip("tokenizers not installed")
