"""Wire helpers for Ollama chat (messages + options)."""

from __future__ import annotations

from nucleusiq_ollama._shared.wire import (
    _extract_text_and_images,
    _split_data_url,
    build_chat_kwargs,
    build_options,
    sanitize_messages,
    tool_arguments_to_json_string,
)

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAen63NgAAAAASUVORK5CYII="
JPEG_B64 = "/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3aXRoIEdJTVD/4QFISGV4O7Yk"


def test_sanitize_messages_coerces_flat_tool_calls() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
            ],
        }
    ]
    out = sanitize_messages(msgs)
    tc0 = out[0]["tool_calls"][0]
    assert tc0["type"] == "function"
    assert tc0["id"] == "call_1"
    assert tc0["function"]["name"] == "get_weather"
    assert tc0["function"]["arguments"] == {"city": "Paris"}


def test_sanitize_messages_decodes_json_string_arguments_for_ollama_client() -> None:
    """The ``ollama`` client validates ``function.arguments`` as a Mapping.

    Agent history replays assistant tool calls with JSON-string arguments
    (OpenAI shape); sending them back verbatim fails validation on every
    multi-turn tool conversation.
    """
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_document",
                        "arguments": '{"path": "a.pdf"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "noop", "arguments": ""},
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "broken", "arguments": "not json"},
                },
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {"name": "already", "arguments": {"k": 1}},
                },
            ],
        }
    ]
    out = sanitize_messages(msgs)
    args = [tc["function"]["arguments"] for tc in out[0]["tool_calls"]]
    assert args == [{"path": "a.pdf"}, {}, {"_raw": "not json"}, {"k": 1}]
    # The original message is not mutated.
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.pdf"}'

    from ollama._types import Message

    Message(**out[0])  # must validate against the client's schema


def test_build_options_maps_num_predict_and_stop_scalar() -> None:
    o = build_options(
        max_output_tokens=256,
        temperature=0.2,
        top_p=0.95,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        stop=["."],
        seed=42,
    )
    assert o["num_predict"] == 256
    assert o["temperature"] == 0.2
    assert o["stop"] == "."


def test_build_chat_kwargs_includes_think_and_format() -> None:
    fmt = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    kwargs = build_chat_kwargs(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        format_payload=fmt,
        options={},
        think=True,
        keep_alive="5m",
        stream=False,
        tool_choice="auto",
    )
    assert kwargs["model"] == "m"
    assert kwargs["format"] == fmt
    assert kwargs["think"] is True
    assert kwargs["keep_alive"] == "5m"


def test_tool_arguments_to_json_string() -> None:
    assert tool_arguments_to_json_string({"x": 1}) == '{"x": 1}'
    assert tool_arguments_to_json_string("{}") == "{}"


def test_sanitize_messages_non_dict_tool_call_entry() -> None:
    msgs = [{"role": "assistant", "tool_calls": [None, object()]}]
    out = sanitize_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["name"] == ""


# --- vision wire ----------------------------------------------------------- #


def test_split_data_url_decodes_base64() -> None:
    assert _split_data_url(f"data:image/png;base64,{PNG_B64}") == PNG_B64
    assert _split_data_url(f"data:image/jpeg;base64,{JPEG_B64}") == JPEG_B64


def test_split_data_url_rejects_non_base64_urls() -> None:
    assert _split_data_url("https://example.com/x.png") is None
    assert _split_data_url("data:image/png;utf8,hello") is None
    assert _split_data_url("data:image/png;base64") is None  # no comma
    assert _split_data_url("not a url") is None
    assert _split_data_url(None) is None  # type: ignore[arg-type]


def test_extract_text_and_images_passthrough() -> None:
    text, images = _extract_text_and_images("hello")
    assert text == "hello"
    assert images == []

    text, images = _extract_text_and_images(None)
    assert text is None
    assert images == []


def test_extract_text_and_images_splits_openai_parts() -> None:
    content = [
        {"type": "text", "text": "what is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{JPEG_B64}"},
        },
        {"type": "text", "text": "and this one?"},
    ]
    text, images = _extract_text_and_images(content)
    assert text == "what is in this image?\nand this one?"
    assert images == [PNG_B64, JPEG_B64]


def test_extract_text_and_images_skips_http_image_url(caplog) -> None:
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]
    with caplog.at_level("WARNING"):
        text, images = _extract_text_and_images(content)
    assert text == "look"
    assert images == []
    assert any("HTTP image_url" in r.message for r in caplog.records)


def test_extract_text_and_images_supports_raw_image_blocks() -> None:
    content = [
        {"type": "text", "text": "raw"},
        {"type": "image", "data": PNG_B64},
    ]
    text, images = _extract_text_and_images(content)
    assert text == "raw"
    assert images == [PNG_B64]


def test_extract_text_and_images_ignores_garbage_parts() -> None:
    text, images = _extract_text_and_images([{"type": "audio"}, "not a dict", 42])
    assert text is None
    assert images == []


def test_extract_text_and_images_non_str_content() -> None:
    # Numeric / object content is coerced to str.
    text, images = _extract_text_and_images(123)  # type: ignore[arg-type]
    assert text == "123"
    assert images == []


def test_sanitize_messages_decodes_user_vision_message() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                },
            ],
        }
    ]
    out = sanitize_messages(msgs)
    user = out[0]
    assert user["content"] == "describe"
    assert user["images"] == [PNG_B64]


def test_sanitize_messages_merges_with_preexisting_images() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                }
            ],
            "images": [JPEG_B64],
        }
    ]
    out = sanitize_messages(msgs)
    assert out[0]["images"] == [JPEG_B64, PNG_B64]


def test_sanitize_messages_leaves_text_only_user_content_intact() -> None:
    msgs = [{"role": "user", "content": "hello"}]
    out = sanitize_messages(msgs)
    assert out[0]["content"] == "hello"
    assert "images" not in out[0]


def test_sanitize_messages_handles_system_vision_message() -> None:
    msgs = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "system context"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                },
            ],
        }
    ]
    out = sanitize_messages(msgs)
    assert out[0]["content"] == "system context"
    assert out[0]["images"] == [PNG_B64]


def test_sanitize_messages_empty_text_with_images_yields_empty_content() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                }
            ],
        }
    ]
    out = sanitize_messages(msgs)
    assert out[0]["content"] == ""
    assert out[0]["images"] == [PNG_B64]
