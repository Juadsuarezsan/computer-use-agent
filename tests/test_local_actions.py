from __future__ import annotations

import json
import struct

import pytest

from src.api.schemas import Action
from src.local_model.actions import (
    SYSTEM_PROMPT,
    build_messages,
    encode_action,
    parse_action,
    png_size,
    user_text,
)

W, H = 1024, 768


def fake_png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
    )


@pytest.mark.parametrize(
    "action",
    [
        Action(type="click", coords=(512, 384)),
        Action(type="double_click", coords=(10, 20)),
        Action(type="right_click", coords=(1000, 700)),
        Action(type="mouse_move", coords=(300, 300)),
        Action(type="type", text="Calle 45 #12-30, Bogotá"),
        Action(type="key", key="ctrl+a"),
        Action(type="key", key="Return"),
        Action(type="scroll", coords=(500, 400), scroll_direction="up", scroll_amount=5),
        Action(type="scroll", scroll_direction="down", scroll_amount=3),
        Action(type="wait", duration_ms=200),
        Action(type="task_complete", text="4.2.1-legacy"),
    ],
)
def test_round_trip(action: Action):
    text = encode_action(action, W, H)
    json.loads(text)  # always valid JSON
    parsed, error = parse_action(text, W, H)
    assert error is None
    assert parsed.type == action.type
    if action.coords:
        assert parsed.coords is not None
        assert abs(parsed.coords[0] - action.coords[0]) <= 2
        assert abs(parsed.coords[1] - action.coords[1]) <= 2
    if action.type == "type":
        assert parsed.text == action.text
    if action.type == "key":
        assert parsed.key.lower() in {action.key.lower(), "return"}
    if action.type == "task_complete":
        assert parsed.text == action.text
    if action.type == "scroll":
        assert parsed.scroll_direction == action.scroll_direction
        assert parsed.scroll_amount == action.scroll_amount


def test_enter_is_encoded_as_enter():
    assert json.loads(encode_action(Action(type="key", key="Return"), W, H))["action"] == "ENTER"


def test_positions_are_relative_and_below_one():
    payload = json.loads(encode_action(Action(type="click", coords=(1024, 768)), W, H))
    assert all(0 <= v < 1 for v in payload["position"])


def test_encode_rejects_non_outputs():
    with pytest.raises(ValueError):
        encode_action(Action(type="screenshot"), W, H)
    with pytest.raises(ValueError):
        encode_action(Action(type="click"), W, H)
    with pytest.raises(ValueError):
        encode_action(Action(type="click", coords=(1, 1)), 0, H)


def test_parse_is_tolerant_to_showui_style_and_noise():
    action, error = parse_action(
        "Sure! {'action': 'CLICK', 'value': None, 'position': [0.5, 0.25]} done", W, H
    )
    assert error is None
    assert action.type == "click" and action.coords == (512, 192)
    action, error = parse_action('{"action": "input", "value": "Alice", "position": null}', W, H)
    assert error is None and action.type == "type" and action.text == "Alice"


def test_parse_pixel_coordinates_are_clipped():
    action, error = parse_action('{"action": "CLICK", "position": [300, 5000]}', W, H)
    assert error is None and action.coords == (300, H - 1)


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        "{not: valid",
        "[1, 2]",
        '{"action": "CLICK", "position": null}',
        '{"action": "CLICK", "position": ["a", "b"]}',
        '{"action": "INPUT", "value": ""}',
        '{"action": "KEY", "value": null}',
        '{"action": "FLY"}',
    ],
)
def test_parse_errors_fall_back_to_wait(text: str):
    action, error = parse_action(text, W, H)
    assert error
    assert action.type == "wait"


def test_parse_scroll_defaults():
    action, error = parse_action('{"action": "SCROLL", "value": "sideways", "amount": "x"}', W, H)
    assert error is None
    assert action.scroll_direction == "down" and action.scroll_amount == 3


def test_parse_answer_aliases():
    action, _ = parse_action('{"action": "DONE", "value": null}', W, H)
    assert action.type == "task_complete" and action.text == "done"


def test_messages_and_prompt():
    msgs = build_messages("Do X", ["a", "b"], target="t")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"][0]["text"] == SYSTEM_PROMPT
    assert msgs[1]["content"][0] == {"type": "image"}
    assert "1. a\n2. b" in msgs[1]["content"][1]["text"]
    assert len(build_messages("Do X", [])) == 2
    assert "Previous actions: none" in user_text("Do X", [])


def test_png_size():
    assert png_size(fake_png(1366, 768)) == (1366, 768)
    with pytest.raises(ValueError):
        png_size(b"GIF89a....................")
