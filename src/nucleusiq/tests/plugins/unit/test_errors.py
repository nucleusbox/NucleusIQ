"""Unit tests for plugin error types."""

import pytest
from nucleusiq.plugins.errors import PluginError, PluginHalt


class TestPluginHalt:
    def test_with_result(self):
        exc = PluginHalt("done")
        assert exc.result == "done"
        assert "done" in str(exc)

    def test_without_result(self):
        exc = PluginHalt()
        assert exc.result is None
        assert "halted" in str(exc).lower()

    def test_with_complex_result(self):
        data = {"status": "blocked", "reason": "limit"}
        exc = PluginHalt(data)
        assert exc.result is data

    def test_is_exception(self):
        assert issubclass(PluginHalt, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(PluginHalt) as exc_info:
            raise PluginHalt("test_result")
        assert exc_info.value.result == "test_result"


class TestPluginError:
    def test_basic(self):
        exc = PluginError("something broke")
        assert str(exc) == "something broke"

    def test_is_exception(self):
        assert issubclass(PluginError, Exception)
