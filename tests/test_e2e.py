"""End-to-end tests of the LangGraph loop on the FakeVM (no browser, no LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from src.agent.orchestrator import build_graph, build_reasoner, build_verifier, run_task
from src.agent.protocols import Observation, ReasonerOutput
from src.agent.reasoner import ClaudeComputerUseReasoner, StubReasoner
from src.agent.verifier import ClaudeVerifier, HeuristicVerifier
from src.agent.vm_executor import FakeVM
from src.api.schemas import Action
from src.storage.audit import SQLiteAuditLog


class ScriptedReasoner:
    name = "scripted"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.i = 0
        self.notes: list[str] = []

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        self.i = 0

    def feedback(self, note: str) -> None:
        self.notes.append(note)

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        action = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return ReasonerOutput(action, f"scripted {action.type}")


async def test_e2e_open_folder_completes():
    out = await run_task(
        task="Open the My Documents folder", max_steps=8, vm=FakeVM(), reasoner=StubReasoner()
    )
    assert out.total_steps >= 1
    assert out.success is True
    assert out.steps[-1].action.type == "task_complete"
    assert out.reasoner == "stub" and out.vm == "fake"
    assert out.trace_id and out.latency_ms >= 0
    assert out.ground_truth_checked is False


async def test_e2e_safety_layer_blocks_dangerous_action():
    reasoner = ScriptedReasoner([Action(type="type", text="rm -rf /")])
    g = build_graph(FakeVM(), reasoner, verifier=None)
    state = await g.ainvoke(
        {"trace_id": "t", "task": "x", "step": 0, "max_steps": 3, "steps_taken": [], "blocked": 0}
    )
    assert state["blocked"] == 1
    assert state["success"] is False
    assert state["steps_taken"][0].safety_blocked is True
    assert "rm -rf" in state["steps_taken"][0].safety_reason


async def test_e2e_graph_node_names_do_not_collide_with_state_keys():
    g = build_graph(FakeVM(), StubReasoner())
    nodes = set(g.get_graph().nodes)
    assert {"observe", "verify", "reason", "safety", "execute"} <= nodes
    assert "step" not in nodes


async def test_e2e_max_steps_reached_is_a_failure():
    reasoner = ScriptedReasoner([Action(type="click", coords=(1, 1))])
    out = await run_task(task="loop forever", max_steps=4, vm=FakeVM(), reasoner=reasoner)
    assert out.success is False
    assert out.total_steps == 4
    assert out.final_output == "max_steps_or_blocked"


async def test_e2e_hitl_block_and_allow_modes():
    actions = [Action(type="type", text="please transfer funds now"), Action(type="task_complete", text="ok")]
    blocked = await run_task(task="pay", max_steps=3, vm=FakeVM(), reasoner=ScriptedReasoner(actions))
    assert blocked.success is False and blocked.blocked_actions == 1
    allowed = await run_task(
        task="pay", max_steps=3, vm=FakeVM(), reasoner=ScriptedReasoner(actions), hitl_mode="allow"
    )
    assert allowed.success is True and allowed.blocked_actions == 0


async def test_e2e_safety_disabled_executes_dangerous_text():
    vm = FakeVM()
    out = await run_task(
        task="x",
        max_steps=3,
        vm=vm,
        reasoner=ScriptedReasoner([Action(type="type", text="sudo rm -rf /"), Action(type="task_complete")]),
        safety_enabled=False,
    )
    assert out.blocked_actions == 0
    assert vm.typed == ["sudo rm -rf /"]


class FrozenVM(FakeVM):
    """A VM whose screen never changes (verifier must flag every action)."""

    async def screenshot(self) -> Observation:
        obs = await super().screenshot()
        obs.description = "frozen"
        obs.screenshot_path = "frozen.png"
        return obs


async def test_e2e_verifier_counts_failures_and_recovery():
    reasoner = ScriptedReasoner(
        [Action(type="click", coords=(1, 1)), Action(type="task_complete", text="ok")]
    )
    out = await run_task(
        task="x", max_steps=5, vm=FrozenVM(), reasoner=reasoner, verifier=HeuristicVerifier()
    )
    assert out.verifier_failures == 1
    assert out.recovered is True
    assert out.steps[0].verifier_ok is False
    assert reasoner.notes and "verifier" in reasoner.notes[0]


async def test_e2e_ground_truth_checker_decides_success():
    spec = {"id": "x-01", "app": "fake", "checker": {"type": "answer_contains", "value": "42"}, "plan": []}
    good = ScriptedReasoner([Action(type="task_complete", text="the answer is 42")])
    bad = ScriptedReasoner([Action(type="task_complete", text="no idea")])
    ok = await run_task(task="answer", max_steps=3, vm=FakeVM(), reasoner=good, task_spec=spec)
    ko = await run_task(task="answer", max_steps=3, vm=FakeVM(), reasoner=bad, task_spec=spec)
    assert ok.success is True and ok.ground_truth_checked is True and ok.task_id == "x-01"
    assert ko.success is False and ko.ground_truth_checked is True


async def test_e2e_audit_log_receives_run():
    audit = SQLiteAuditLog(":memory:")
    out = await run_task(
        task="Open the folder", max_steps=6, vm=FakeVM(), reasoner=StubReasoner(), audit=audit
    )
    rows = audit.recent(10)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == out.trace_id
    assert rows[0]["total_steps"] == out.total_steps


async def test_e2e_default_vm_is_built_and_closed(settings):
    out = await run_task(task="Open the folder", max_steps=6, settings=settings)
    assert out.vm == "fake" and out.reasoner == "stub"


def test_build_reasoner_and_verifier_follow_api_key(settings):
    assert isinstance(build_reasoner(settings), StubReasoner)
    assert isinstance(build_verifier(settings), HeuristicVerifier)
    keyed = settings.model_copy(update={"anthropic_api_key": "sk-test"})
    assert isinstance(build_reasoner(keyed), ClaudeComputerUseReasoner)
    assert isinstance(build_verifier(keyed), ClaudeVerifier)
    assert isinstance(build_reasoner(keyed, force_stub=True), StubReasoner)


@pytest.mark.parametrize("bad", [Action(type="click"), Action(type="key", key="alt+f4")])
async def test_e2e_every_blocked_kind_ends_run(bad: Action):
    out = await run_task(task="x", max_steps=3, vm=FakeVM(), reasoner=ScriptedReasoner([bad]))
    assert out.success is False and out.blocked_actions == 1
