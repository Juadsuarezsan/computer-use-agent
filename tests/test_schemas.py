import pytest
from pydantic import ValidationError

from src.api.schemas import MAX_TASK_LENGTH, Action, TaskRequest


def test_action_rejects_negative_coordinates():
    with pytest.raises(ValidationError):
        Action(type="click", coords=(-1, 10))


def test_action_rejects_off_screen_coordinates():
    with pytest.raises(ValidationError):
        Action(type="click", coords=(5000, 10))


def test_action_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Action.model_validate({"type": "click", "coords": [1, 1], "shell": "rm -rf /"})


def test_action_summary_formats():
    assert Action(type="click", coords=(1, 2)).summary() == "click(1, 2)"
    assert Action(type="type", text="hi").summary() == "type('hi')"
    assert Action(type="key", key="ctrl+a").summary() == "key(ctrl+a)"
    assert Action(type="scroll").summary() == "scroll(down, 3)"
    assert Action(type="wait", duration_ms=5).summary() == "wait(5ms)"
    assert Action(type="task_complete", text="x").summary() == "task_complete('x')"
    assert Action(type="screenshot").summary() == "screenshot"


def test_task_request_rejects_blank_and_oversized():
    with pytest.raises(ValidationError):
        TaskRequest(task="   ")
    with pytest.raises(ValidationError):
        TaskRequest(task="x" * (MAX_TASK_LENGTH + 1))
    with pytest.raises(ValidationError):
        TaskRequest(task="ok task", max_steps=0)


def test_task_request_strips_whitespace():
    assert TaskRequest(task="  open the form  ").task == "open the form"
