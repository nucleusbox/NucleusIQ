"""Additional coverage for :mod:`nucleusiq_anthropic._shared.wire`."""

from __future__ import annotations

from nucleusiq_anthropic._shared.wire import (
    anthropic_tool_choice,
    drop_unsupported_sampling,
    flatten_tools,
    split_system,
    translate_messages,
)


def test_drop_unsupported_sampling_strips_known_keys() -> None:
    d = drop_unsupported_sampling(
        {
            "frequency_penalty": 1,
            "max_output_tokens": 9,
            "foo": "bar",
        }
    )

    assert "frequency_penalty" not in d
    assert "max_output_tokens" not in d
    assert d.get("foo") == "bar"


def test_translate_image_url_http_and_data() -> None:
    _, msgs = translate_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://x.example/a.png"},
                    },
                    {"type": "image_url", "image_url": "https://y.example/b.png"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                    {"type": "image_url", "image_url": {"url": ""}},
                ],
            }
        ]
    )

    blk = msgs[0]["content"]
    assert blk[0]["source"]["type"] == "url"
    assert blk[1]["source"]["url"] == "https://y.example/b.png"
    assert blk[2]["source"]["type"] == "base64"
    assert blk[2]["source"]["data"] == "QUJD"


def test_translate_user_empty_and_fallback_role() -> None:
    _, m1 = translate_messages([{"role": "user", "content": "   "}])
    assert m1[0]["content"] == [{"type": "text", "text": ""}]

    _, m2 = translate_messages([{"role": "debugger", "content": None}])
    assert m2[0]["role"] == "user"


def test_translate_system_leftover_updates_system(monkeypatch) -> None:

    msgs = [{"role": "system", "content": "orphan"}, {"role": "user", "content": "hi"}]

    from nucleusiq_anthropic._shared import wire as w

    def fake_split(mlist):
        return None, mlist

    monkeypatch.setattr(w, "split_system", fake_split)
    sy, rest = translate_messages(list(msgs))
    assert sy == "orphan"
    assert len(rest) == 1


def test_translate_tools_function_role_and_tool_list_content() -> None:
    _, out = translate_messages(
        [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]},
            {
                "role": "function",
                "tool_call_id": "a",
                "content": [{"k": "v"}],
            },
        ]
    )
    assistant = next(m for m in out if m["role"] == "assistant")
    assert any(b["type"] == "text" and b["text"] == "" for b in assistant["content"])

    blocks = next(
        m["content"]
        for m in out
        if m["role"] == "user" and isinstance(m["content"], list)
    )

    assert "k" in blocks[0]["content"]


def test_tool_calls_object_fallback() -> None:
    class _TC:
        pass

    tc = _TC()
    tc.id = "oid"
    tc.function = {"name": "f", "arguments": "not-json"}

    _, out = translate_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tc],
            }
        ]
    )

    blk = next(m for m in out if m["role"] == "assistant")["content"]
    tu = next(b for b in blk if b["type"] == "tool_use")

    assert tu["name"] == "f"


def test_flatten_non_dict_tool_entry() -> None:
    sentinel = ("native",)

    flat = flatten_tools([sentinel])
    assert flat == [sentinel]


def test_anthropic_tool_choice_variants() -> None:
    assert anthropic_tool_choice("OFF") == {"type": "none"}

    wrapped = anthropic_tool_choice({"type": "custom", "x": 1})
    assert wrapped["type"] == "custom"


def test_anthropic_tool_choice_missing_function_name() -> None:

    assert anthropic_tool_choice({"type": "function", "function": {}}) == {
        "type": "auto",
    }


def test_raw_image_pass_through() -> None:
    blob = {"type": "image", "source": {"type": "base64"}}

    _, out = translate_messages([{"role": "user", "content": [blob]}])
    assert out[0]["content"][0]["type"] == "image"


def test_user_message_list_mixed_plain_and_parts() -> None:
    _, out = translate_messages(
        [{"role": "user", "content": ["skip-me", {"type": "text", "text": " kept "}]}],
    )

    blk = out[0]["content"]
    assert len(blk) == 1
    assert blk[0]["text"] == " kept "


def test_data_url_plaintext_encoded() -> None:
    _, out = translate_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:text/plain;charset=utf-8,h%69",
                        },
                    },
                ],
            },
        ],
    )

    assert out[0]["content"][0]["source"]["data"] == "hi"


def test_data_url_without_comma_ignored_for_image_blocks() -> None:
    _, out = translate_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:broken-without-comma-section"},
                    },
                ],
            },
        ],
    )

    blk = out[0]["content"]
    assert blk == [{"type": "text", "text": ""}]


def test_text_part_blank_is_dropped_so_user_fallback_empty_block() -> None:
    _, out = translate_messages(
        [{"role": "user", "content": [{"type": "text", "text": ""}]}],
    )

    assert out[0]["content"] == [{"type": "text", "text": ""}]


def test_translation_tool_arguments_non_object_json() -> None:

    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "t2",
                    "function": {"name": "noop", "arguments": "true"},
                },
            ],
            "content": "",
        },
        {"role": "tool", "tool_call_id": "t2", "content": "ok"},
    ]

    _, out = translate_messages(msgs)
    asst = next(m for m in out if m["role"] == "assistant")
    tc_block = next(b for b in asst["content"] if b["type"] == "tool_use")
    assert tc_block["input"] == {"value": True}


def test_assistant_multipart_non_text_skipped() -> None:
    _, out = translate_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image", "source": {"type": "url"}},
                ],
            },
        ],
    )

    blk = next(m for m in out if m["role"] == "assistant")["content"]
    assert blk[0]["text"] == "a"


def test_tool_batch_stops_at_non_tool_and_none_content() -> None:
    _, out = translate_messages(
        [
            {"role": "tool", "tool_call_id": "a", "content": None},
            {"role": "user", "content": "next"},
        ],
    )

    assert len(out) == 2

    summary = out[0]["content"][0]

    assert summary["type"] == "tool_result"
    assert summary["tool_use_id"] == "a"
    assert summary["content"] == ""
    assert out[1]["content"][0]["text"] == "next"


def test_anthropic_tool_choice_unknown_type_returns_none() -> None:
    assert anthropic_tool_choice(()) is None


def test_split_system_list_only_text_slices() -> None:
    sy, rest = split_system(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "a"}, {"type": "other"}],
            }
        ],
    )

    assert sy == "a"
    assert not rest


def test_tool_use_arguments_dict_copy() -> None:
    _, out = translate_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "d1",
                        "function": {"name": "g", "arguments": {"p": 1}},
                    },
                ],
                "content": "",
            },
            {"role": "user", "content": "z"},
        ],
    )

    asst = next(m for m in out if m["role"] == "assistant")
    block = next(b for b in asst["content"] if b["type"] == "tool_use")
    assert block["input"] == {"p": 1}
