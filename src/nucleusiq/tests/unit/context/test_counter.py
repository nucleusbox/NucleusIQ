"""Tests for TokenCounter protocol and DefaultTokenCounter."""

from nucleusiq.agents.chat_models import ChatMessage
from nucleusiq.agents.context.counter import DefaultTokenCounter, TokenCounter


class TestDefaultTokenCounter:
    def test_count_short(self):
        counter = DefaultTokenCounter()
        assert counter.count("hello") >= 1

    def test_count_empty(self):
        counter = DefaultTokenCounter()
        assert counter.count("") == 1  # min 1

    def test_count_long(self):
        counter = DefaultTokenCounter()
        text = "a" * 400
        assert counter.count(text) == 100

    def test_count_messages_simple(self):
        counter = DefaultTokenCounter()
        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello world"),
        ]
        total = counter.count_messages(msgs)
        assert total > 0
        assert total >= 8  # at least 4 per message framing

    def test_count_messages_multimodal(self):
        counter = DefaultTokenCounter()
        msgs = [
            ChatMessage(
                role="user",
                content=[{"type": "text", "text": "Describe this image"}],
            ),
        ]
        total = counter.count_messages(msgs)
        assert total > 4  # framing + text content

    def test_count_messages_with_name(self):
        counter = DefaultTokenCounter()
        msgs = [
            ChatMessage(role="tool", content="result data", name="calculator"),
        ]
        total = counter.count_messages(msgs)
        assert total > counter.count("result data")  # includes name overhead

    def test_implements_protocol(self):
        counter = DefaultTokenCounter()
        assert isinstance(counter, TokenCounter)
