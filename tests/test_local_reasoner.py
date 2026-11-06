from __future__ import annotations

import pytest

from src.agent.cascade import CascadeReasoner
from src.agent.local_reasoner import Generation, HFVisionBackend, LocalVLMReasoner
from src.agent.orchestrator import build_reasoner, build_verifier, run_task
from src.agent.protocols import Observation, ReasonerOutput
from src.agent.reasoner import StubReasoner
from src.agent.verifier import HeuristicVerifier
from src.api.schemas import Action
from tests.test_local_actions import fake_png


class ScriptedBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def __call__(self, messages, png):
        self.calls.append(messages)
        return Generation(text=self.outputs.pop(0), tokens_in=900, tokens_out=20)


def obs(png: bytes | None = None) -> Observation:
    return Observation(png=fake_png(1000, 500) if png is None else png)


async def test_local_reasoner_parses_and_keeps_history():
    backend = ScriptedBackend(
        [
            '{"action": "CLICK", "value": null, "position": [0.5, 0.5]}',
            '{"action": "INPUT", "value": "Alice", "position": null}',
            "garbage",
            '{"action": "ANSWER", "value": "ok", "position": null}',
        ]
    )
    r = LocalVLMReasoner(backend, history_len=4)
    r.reset({})
    out = await r.next_action("task", obs(), 1)
    assert out.action == Action(type="click", coords=(500, 250))
    assert out.cost_usd == 0.0 and out.tokens_in == 900
    out = await r.next_action("task", obs(), 2)
    assert out.action.type == "type"
    assert "1. " in backend.calls[1][1]["content"][1]["text"]
    out = await r.next_action("task", obs(), 3)
    assert out.action.type == "wait" and "parse error" in out.reasoning
    assert len(r.history) == 2  # failed parses do not enter the history
    out = await r.next_action("task", obs(), 4)
    assert out.action.type == "task_complete"
    r.feedback("verifier: nothing changed")
    assert r.feedback_notes
    r.reset(None)
    assert r.history == [] and r.feedback_notes == []


async def test_local_reasoner_history_is_truncated():
    backend = ScriptedBackend(['{"action": "WAIT"}'] * 5)
    r = LocalVLMReasoner(backend, history_len=2)
    for step in range(1, 6):
        await r.next_action("t", obs(), step)
    last_prompt = backend.calls[-1][1]["content"][1]["text"]
    assert "1. " in last_prompt and "2. " in last_prompt and "3. " not in last_prompt


async def test_local_reasoner_without_png_completes():
    r = LocalVLMReasoner(ScriptedBackend([]))
    out = await r.next_action("t", Observation(png=None), 1)
    assert out.action.type == "task_complete"


async def test_local_reasoner_runs_in_the_loop(fake_vm):
    """FakeVM has no PNG: the loop must still end cleanly."""
    response = await run_task(
        task="t", max_steps=3, reasoner=LocalVLMReasoner(ScriptedBackend([])), vm=fake_vm
    )
    assert response.total_steps == 1


def test_hf_backend_is_lazy():
    backend = HFVisionBackend(adapter="x")
    assert backend._model is None and backend.adapter == "x"


class CountingReasoner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.resets = 0
        self.notes: list[str] = []

    def reset(self, spec=None) -> None:
        self.resets += 1

    def feedback(self, note: str) -> None:
        self.notes.append(note)

    async def next_action(self, task, observation, step):
        self.calls += 1
        return ReasonerOutput(Action(type="wait"), f"{self.name} step {step}")


async def test_cascade_escalates_after_consecutive_failures():
    primary, fallback = CountingReasoner("local"), CountingReasoner("claude")
    c = CascadeReasoner(primary, fallback, escalate_after=2)
    c.reset({"id": "x"})
    await c.next_action("t", obs(), 1)
    c.feedback("verifier: failed")
    await c.next_action("t", obs(), 2)  # 1 consecutive failure
    await c.next_action("t", obs(), 3)  # success in between: counter resets
    c.feedback("verifier: failed")
    await c.next_action("t", obs(), 4)
    c.feedback("verifier: failed")
    out = await c.next_action("t", obs(), 5)  # 2 consecutive -> escalate
    assert c.escalated and c.escalations == 1
    assert out.reasoning.startswith("[claude]")
    assert primary.calls == 4 and fallback.calls == 1
    c.feedback("verifier: failed")
    assert fallback.notes == ["verifier: failed"]
    c.reset(None)
    assert not c.escalated and c.active is primary


async def test_cascade_without_fallback_never_escalates():
    primary = CountingReasoner("local")
    c = CascadeReasoner(primary, None, escalate_after=1)
    c.reset(None)
    for step in range(1, 5):
        c.feedback("verifier: failed")
        await c.next_action("t", obs(), step)
    assert not c.escalated and primary.calls == 4
    assert c.name == "cascade(local->none)"
    with pytest.raises(ValueError):
        CascadeReasoner(primary, None, escalate_after=0)


def test_build_reasoner_modes(settings, monkeypatch):
    s = settings.model_copy(update={"reasoner_mode": "stub"})
    assert isinstance(build_reasoner(s), StubReasoner)
    s = settings.model_copy(update={"reasoner_mode": "local", "local_adapter": "models/a"})
    local = build_reasoner(s)
    assert isinstance(local, LocalVLMReasoner) and local.backend.adapter == "models/a"
    s = settings.model_copy(update={"reasoner_mode": "cascade"})
    cascade = build_reasoner(s)
    assert isinstance(cascade, CascadeReasoner) and cascade.fallback is None  # no key -> never escalates
    assert isinstance(build_verifier(s), HeuristicVerifier)
    s = settings.model_copy(update={"reasoner_mode": "claude"})
    with pytest.raises(ValueError):
        build_reasoner(s)
    s = settings.model_copy(update={"reasoner_mode": "auto"})
    assert isinstance(build_reasoner(s), StubReasoner)


def test_local_mode_never_uses_paid_verifier(settings):
    s = settings.model_copy(update={"reasoner_mode": "local", "anthropic_api_key": "sk-test"})
    assert isinstance(build_verifier(s), HeuristicVerifier)
    cascade = build_reasoner(s.model_copy(update={"reasoner_mode": "cascade"}))
    assert isinstance(cascade, CascadeReasoner) and cascade.fallback is not None
