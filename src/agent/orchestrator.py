"""LangGraph loop: screenshot → reason → safety-check → execute → verify."""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from src.agent.reasoner import ClaudeComputerUseReasoner, StubReasoner
from src.agent.vm_executor import FakeVM, build_vm
from src.api.schemas import Action, Step, TaskResponse
from src.config import get_settings
from src.safety.blocklist import check_action


class LoopState(TypedDict, total=False):
    task: str
    step: int
    max_steps: int
    vm_state: dict[str, Any]
    steps_taken: list[Step]
    blocked: int
    finished: bool
    success: bool


def build_graph(vm, reasoner, max_steps: int):
    async def step_node(state: LoopState) -> dict[str, Any]:
        step_n = state.get("step", 0) + 1
        steps_taken: list[Step] = list(state.get("steps_taken", []))
        blocked = state.get("blocked", 0)

        path, observation = await vm.screenshot()

        action, reasoning = reasoner.next_action(state["task"], observation, step_n)

        verdict = check_action(action)
        if not verdict.allowed:
            blocked += 1
            steps_taken.append(Step(
                step=step_n, action=action, screenshot_path=path,
                reasoning=reasoning, verifier_ok=False,
                safety_blocked=True, safety_reason=verdict.reason,
            ))
            logger.warning(f"Safety blocked: {verdict.reason}")
            return {
                "step": step_n, "steps_taken": steps_taken, "blocked": blocked,
                "finished": True, "success": False,
            }

        result = await vm.execute(action)
        finished = action.type == "task_complete"
        steps_taken.append(Step(
            step=step_n, action=action, screenshot_path=path,
            reasoning=f"{reasoning} → {result}", verifier_ok=True,
        ))
        if step_n >= state.get("max_steps", max_steps):
            finished = True
        return {
            "step": step_n, "steps_taken": steps_taken, "blocked": blocked,
            "finished": finished, "success": finished and not state.get("blocked", 0),
        }

    def route(state: LoopState) -> str:
        return END if state.get("finished") else "step"

    g = StateGraph(LoopState)
    g.add_node("step", step_node)
    g.set_entry_point("step")
    g.add_conditional_edges("step", route, {END: END, "step": "step"})
    return g.compile()


async def run_task(*, task: str, max_steps: int | None = None) -> TaskResponse:
    s = get_settings()
    eff_max = max_steps or s.max_steps_per_task
    vm = build_vm()
    if s.anthropic_api_key:
        reasoner = ClaudeComputerUseReasoner(model="claude-sonnet-4-5",
                                              api_key=s.anthropic_api_key,
                                              tool_version=s.computer_use_tool)
    else:
        reasoner = StubReasoner()
    reasoner.reset()
    graph = build_graph(vm, reasoner, eff_max)
    t0 = time.perf_counter()
    state = await graph.ainvoke({"task": task, "step": 0, "max_steps": eff_max,
                                    "steps_taken": [], "blocked": 0})
    return TaskResponse(
        task=task,
        success=bool(state.get("success")),
        steps=state.get("steps_taken", []),
        final_output="completed" if state.get("success") else "max_steps_or_blocked",
        total_steps=state.get("step", 0),
        blocked_actions=state.get("blocked", 0),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
