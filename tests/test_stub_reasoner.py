import pytest

from src.agent.protocols import Observation, UIElement
from src.agent.reasoner import StubReasoner


def _obs(elements=None, fields=None) -> Observation:
    return Observation(description="screen", elements=elements or [], fields=fields or {})


async def test_stub_emits_plan_for_open_folder():
    r = StubReasoner()
    actions = []
    for i in range(5):
        out = await r.next_action("Open My Documents folder", _obs(), i)
        actions.append(out.action.type)
    assert "task_complete" in actions


async def test_stub_resets_state():
    r = StubReasoner()
    for i in range(3):
        await r.next_action("test", _obs(), i)
    assert len(r.history) == 3
    r.reset()
    assert r.history == []


async def test_scripted_plan_resolves_selectors_to_pixels():
    r = StubReasoner()
    r.reset({"plan": [{"action": "click", "target": "#ok"}, {"action": "task_complete", "text": "done"}]})
    obs = _obs([UIElement("#ok", "button", "OK", 120, 40)])
    out = await r.next_action("t", obs, 1)
    assert out.action.type == "click"
    assert out.action.coords == (120, 40)
    assert "#ok" in out.reasoning
    out = await r.next_action("t", obs, 2)
    assert out.action.type == "task_complete"
    assert out.action.text == "done"


async def test_scripted_plan_waits_then_gives_up_when_target_missing():
    r = StubReasoner(max_wait_retries=2)
    r.reset({"plan": [{"action": "click", "target": "#missing"}]})
    first = await r.next_action("t", _obs(), 1)
    second = await r.next_action("t", _obs(), 2)
    third = await r.next_action("t", _obs(), 3)
    assert first.action.type == "wait" and second.action.type == "wait"
    assert third.action.type == "task_complete"
    assert "never became visible" in third.reasoning


async def test_answer_from_reads_fields():
    r = StubReasoner()
    r.reset({"plan": [{"action": "task_complete", "answer_from": "a,b"}]})
    out = await r.next_action("t", _obs(fields={"a": "1", "b": "2"}), 1)
    assert out.action.text == "1, 2"


async def test_plan_exhausted_completes():
    r = StubReasoner()
    r.reset({"plan": [{"action": "screenshot"}]})
    await r.next_action("t", _obs(), 1)
    out = await r.next_action("t", _obs(), 2)
    assert out.action.type == "task_complete"
    assert "exhausted" in out.reasoning


async def test_other_plan_actions():
    r = StubReasoner()
    r.reset(
        {
            "plan": [
                {"action": "type", "text": "hi"},
                {"action": "key", "key": "ctrl+a"},
                {"action": "scroll", "direction": "up", "amount": 2},
                {"action": "wait", "duration_ms": 10},
                {"action": "click", "coords": [5, 6]},
            ]
        }
    )
    kinds = [(await r.next_action("t", _obs(), i)).action for i in range(5)]
    assert [k.type for k in kinds] == ["type", "key", "scroll", "wait", "click"]
    assert kinds[2].scroll_direction == "up" and kinds[2].scroll_amount == 2
    assert kinds[4].coords == (5, 6)


async def test_unknown_plan_action_raises():
    r = StubReasoner()
    r.reset({"plan": [{"action": "teleport"}]})
    with pytest.raises(ValueError):
        await r.next_action("t", _obs(), 1)


def test_feedback_is_recorded():
    r = StubReasoner()
    r.feedback("verifier: no change")
    assert r.feedback_notes == ["verifier: no change"]
