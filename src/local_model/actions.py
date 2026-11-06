"""Action text format and prompt shared by training and inference.

The model answers with one JSON object per step, close to the native format of
ShowUI (``{"action": "CLICK", "value": null, "position": [0.49, 0.42]}``), so
the base model is not asked to learn a new syntax from scratch. Positions are
relative to the screenshot (``0 <= x, y < 1``), which makes the format
independent of the screen resolution.

The parser is tolerant: it accepts JSON or Python literals (single quotes,
``None``), extra text around the object and upper/lower-case action names.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from src.api.schemas import Action

POSITION_DECIMALS = 3

SYSTEM_PROMPT = (
    "You are a GUI agent operating a desktop application through screenshots. "
    "Given the task, the previous actions and the current screenshot, output the next action "
    "as exactly one JSON object with the keys 'action', 'value' and 'position'.\n"
    "Actions:\n"
    "- CLICK, DOUBLE_CLICK, RIGHT_CLICK, HOVER: 'position' is [x, y] relative to the screenshot "
    "(0 to 1), 'value' is null.\n"
    "- INPUT: 'value' is the text to type into the focused field, 'position' is null.\n"
    "- ENTER: press the Enter key.\n"
    "- KEY: 'value' is a key combination such as 'ctrl+a' or 'Tab'.\n"
    "- SCROLL: 'value' is up, down, left or right; optional 'amount' (default 3).\n"
    "- WAIT: the screen is still loading.\n"
    "- ANSWER: the task is done; 'value' is the requested value or a short summary."
)

_POINTER = {
    "click": "CLICK",
    "double_click": "DOUBLE_CLICK",
    "right_click": "RIGHT_CLICK",
    "mouse_move": "HOVER",
}
_POINTER_BACK = {v: k for k, v in _POINTER.items()}
_ENTER_KEYS = {"return", "enter", "kp_enter"}
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def png_size(png: bytes) -> tuple[int, int]:
    """Width and height read from the PNG header (no image library needed)."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def _relative(value: int, size: int) -> float:
    """Pixel coordinate -> relative coordinate in ``[0, 1)``."""
    if size <= 0:
        raise ValueError("screen size must be positive")
    rel: float = min(max(value / size, 0.0), 1.0 - 10.0**-POSITION_DECIMALS)
    return round(rel, POSITION_DECIMALS)


def _absolute(value: float, size: int) -> int:
    """Relative coordinate -> pixel coordinate clipped to the screen."""
    return int(min(max(round(float(value) * size), 0), size - 1))


def encode_action(action: Action, width: int, height: int) -> str:
    """Return the JSON text the model must produce for ``action``.

    Raises:
        ValueError: for actions that never appear as model outputs (``screenshot``)
            or pointer actions without coordinates.
    """
    payload: dict[str, Any]
    if action.type in _POINTER:
        if action.coords is None:
            raise ValueError(f"{action.type} without coordinates")
        x, y = action.coords
        payload = {
            "action": _POINTER[action.type],
            "value": None,
            "position": [_relative(x, width), _relative(y, height)],
        }
    elif action.type == "type":
        payload = {"action": "INPUT", "value": action.text or "", "position": None}
    elif action.type == "key":
        key = action.key or ""
        if key.lower() in _ENTER_KEYS:
            payload = {"action": "ENTER", "value": None, "position": None}
        else:
            payload = {"action": "KEY", "value": key, "position": None}
    elif action.type == "scroll":
        position = (
            [_relative(action.coords[0], width), _relative(action.coords[1], height)]
            if action.coords
            else None
        )
        payload = {
            "action": "SCROLL",
            "value": action.scroll_direction,
            "position": position,
            "amount": action.scroll_amount,
        }
    elif action.type == "wait":
        payload = {"action": "WAIT", "value": None, "position": None}
    elif action.type == "task_complete":
        payload = {"action": "ANSWER", "value": action.text or "", "position": None}
    else:
        raise ValueError(f"action {action.type!r} is not a model output")
    return json.dumps(payload, ensure_ascii=False)


def _load_object(text: str) -> dict[str, Any]:
    match = _OBJECT_RE.search(text)
    if match is None:
        raise ValueError("no object in model output")
    raw = match.group(0)
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        try:
            loaded = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"unparseable action: {raw[:120]!r}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("model output is not an object")
    return loaded


def _position(payload: dict[str, Any], width: int, height: int) -> tuple[int, int] | None:
    pos = payload.get("position")
    if not isinstance(pos, list | tuple) or len(pos) != 2:
        return None
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return None
    if x > 1.0 or y > 1.0:  # an un-tuned checkpoint may answer in pixels: clip to the screen
        return (int(min(max(x, 0), width - 1)), int(min(max(y, 0), height - 1)))
    return (_absolute(x, width), _absolute(y, height))


def parse_action(text: str, width: int, height: int) -> tuple[Action, str | None]:
    """Parse the model output into an :class:`Action`.

    Returns:
        The action and an error message. On any error the action is a harmless
        ``wait`` so the loop can continue and the verifier can react.
    """
    fallback = Action(type="wait", duration_ms=200)
    try:
        payload = _load_object(text)
    except ValueError as exc:
        return fallback, str(exc)
    name = str(payload.get("action", "")).strip().upper()
    value = payload.get("value")
    if name in _POINTER_BACK:
        coords = _position(payload, width, height)
        if coords is None:
            return fallback, f"{name} requires a position"
        return Action(type=_POINTER_BACK[name], coords=coords), None
    if name in {"INPUT", "TYPE"}:
        if not isinstance(value, str) or not value:
            return fallback, "INPUT requires a non-empty value"
        return Action(type="type", text=value), None
    if name == "ENTER":
        return Action(type="key", key="Return"), None
    if name == "KEY":
        if not isinstance(value, str) or not value:
            return fallback, "KEY requires a key name"
        return Action(type="key", key=value), None
    if name == "SCROLL":
        direction = str(value or "down").lower()
        if direction not in {"up", "down", "left", "right"}:
            direction = "down"
        try:
            amount = int(payload.get("amount", 3) or 3)
        except (TypeError, ValueError):
            amount = 3
        return (
            Action(
                type="scroll",
                coords=_position(payload, width, height),
                scroll_direction=direction,
                scroll_amount=max(0, min(amount, 100)),
            ),
            None,
        )
    if name == "WAIT":
        return Action(type="wait", duration_ms=200), None
    if name in {"ANSWER", "DONE", "TASK_COMPLETE"}:
        return Action(type="task_complete", text=str(value) if value is not None else "done"), None
    return fallback, f"unknown action {name!r}"


def user_text(task: str, history: list[str]) -> str:
    """Text part of the user turn: task, previous actions and the question."""
    lines = [f"Task: {task}"]
    if history:
        lines.append("Previous actions:")
        lines.extend(f"{i}. {h}" for i, h in enumerate(history, start=1))
    else:
        lines.append("Previous actions: none")
    lines.append("Next action:")
    return "\n".join(lines)


def build_messages(task: str, history: list[str], target: str | None = None) -> list[dict[str, Any]]:
    """Chat messages in the Qwen2-VL format (the image goes before the text).

    Args:
        task: Natural-language task.
        history: Encoded previous actions (most recent last, already truncated).
        target: Encoded answer; when given, an assistant turn is appended (training).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text(task, history)}],
        },
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages
