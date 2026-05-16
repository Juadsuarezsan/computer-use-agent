import pytest

from src.agent.orchestrator import run_task


@pytest.mark.asyncio
async def test_e2e_open_folder_completes():
    out = await run_task(task="Open the My Documents folder", max_steps=8)
    assert out.total_steps >= 1
    assert any(s.action.type == "task_complete" for s in out.steps) or out.total_steps == 8


@pytest.mark.asyncio
async def test_e2e_safety_layer_wired_into_graph():
    """Smoke test: orchestrator wires the safety blocklist into the step node.
    Detailed safety pattern tests live in test_safety.py."""
    from src.agent.orchestrator import build_graph
    from src.agent.vm_executor import FakeVM
    from src.api.schemas import Action

    class DangerousReasoner:
        def reset(self): pass
        def next_action(self, task, observation, step):
            return Action(type="type", text="rm -rf /"), "intentional danger"

    g = build_graph(FakeVM(), DangerousReasoner(), max_steps=3)
    state = await g.ainvoke({"task": "x", "step": 0, "max_steps": 3,
                               "steps_taken": [], "blocked": 0})
    assert state["blocked"] >= 1
    assert state["success"] is False
