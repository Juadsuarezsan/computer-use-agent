"""LangGraph agent loop: observe → verify → reason → safety → execute → (loop).

Node names never collide with state keys (``step`` is a key, the nodes are
``observe``/``verify``/``reason``/``safety``/``execute``). Every node logs its
input and output state with the task ``trace_id``.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from typing_extensions import TypedDict

from src.agent.llm import LLMClient
from src.agent.protocols import VM, Observation, Reasoner, Verifier
from src.agent.reasoner import ClaudeComputerUseReasoner, StubReasoner
from src.agent.verifier import ClaudeVerifier, HeuristicVerifier
from src.agent.vm_executor import build_vm
from src.api.schemas import Action, Step, TaskResponse
from src.config import Settings, get_settings
from src.observability import configure_langsmith, log_node, new_trace_id
from src.safety.blocklist import SafetyVerdict, check_action
from src.storage.audit import AuditLog

HitlMode = Literal["block", "allow"]


class LoopState(TypedDict, total=False):
    """State carried through the graph (all keys optional)."""

    trace_id: str
    task: str
    step: int
    max_steps: int
    observation: Observation | None
    prev_observation: Observation | None
    action: Action | None
    reasoning: str
    steps_taken: list[Step]
    blocked: int
    hitl_flags: int
    verifier_failures: int
    finished: bool
    success: bool
    final_answer: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    step_started: float
    step_tokens: tuple[int, int, float]
    safety_verdict: SafetyVerdict | None


def build_graph(
    vm: VM,
    reasoner: Reasoner,
    *,
    verifier: Verifier | None = None,
    safety_enabled: bool = True,
    hitl_mode: HitlMode = "block",
) -> CompiledStateGraph:
    """Compile the agent loop for a given VM / reasoner / verifier combination.

    Args:
        vm: Environment driver.
        reasoner: Action policy.
        verifier: Optional per-step verifier (``None`` disables verification).
        safety_enabled: ``False`` skips the blocklist (ablation only; never in prod).
        hitl_mode: What to do with actions that require human approval when no
            operator is attached: ``block`` (default) or ``allow``.
    """

    async def observe(state: LoopState) -> dict[str, Any]:
        trace_id = state["trace_id"]
        log_node(trace_id, "observe", "in", dict(state))
        observation = await vm.screenshot()
        out: dict[str, Any] = {
            "observation": observation,
            "prev_observation": state.get("observation"),
            "step_started": time.perf_counter(),
        }
        log_node(trace_id, "observe", "out", out)
        return out

    async def verify(state: LoopState) -> dict[str, Any]:
        trace_id = state["trace_id"]
        before = state.get("prev_observation")
        action = state.get("action")
        after = state.get("observation")
        steps_taken = list(state.get("steps_taken", []))
        failures = state.get("verifier_failures", 0)
        out: dict[str, Any] = {"verifier_failures": failures}
        if verifier is None or before is None or action is None or after is None or not steps_taken:
            log_node(trace_id, "verify", "out", {"skipped": True})
            return out
        result = await verifier.verify(before, action, after)
        last = steps_taken[-1].model_copy(update={"verifier_ok": result.ok, "verifier_note": result.note})
        last.tokens_in += result.tokens_in
        last.tokens_out += result.tokens_out
        last.cost_usd = round(last.cost_usd + result.cost_usd, 6)
        steps_taken[-1] = last
        if not result.ok:
            failures += 1
            reasoner.feedback(f"verifier: previous action ({action.summary()}) failed: {result.note}")
        out = {
            "steps_taken": steps_taken,
            "verifier_failures": failures,
            "tokens_in": state.get("tokens_in", 0) + result.tokens_in,
            "tokens_out": state.get("tokens_out", 0) + result.tokens_out,
            "cost_usd": round(state.get("cost_usd", 0.0) + result.cost_usd, 6),
        }
        log_node(trace_id, "verify", "out", {"ok": result.ok, "note": result.note})
        return out

    async def reason(state: LoopState) -> dict[str, Any]:
        trace_id = state["trace_id"]
        step_n = state.get("step", 0) + 1
        observation = state.get("observation")
        assert observation is not None  # observe always runs first
        decision = await reasoner.next_action(state["task"], observation, step_n)
        out: dict[str, Any] = {
            "step": step_n,
            "action": decision.action,
            "reasoning": decision.reasoning,
            "tokens_in": state.get("tokens_in", 0) + decision.tokens_in,
            "tokens_out": state.get("tokens_out", 0) + decision.tokens_out,
            "cost_usd": round(state.get("cost_usd", 0.0) + decision.cost_usd, 6),
            "step_tokens": (decision.tokens_in, decision.tokens_out, decision.cost_usd),
        }
        log_node(trace_id, "reason", "out", out)
        return out

    async def safety(state: LoopState) -> dict[str, Any]:
        trace_id = state["trace_id"]
        action = state["action"]
        assert action is not None
        if not safety_enabled:
            log_node(trace_id, "safety", "out", {"skipped": True})
            return {"safety_verdict": None}
        verdict = check_action(action)
        blocked = state.get("blocked", 0)
        hitl = state.get("hitl_flags", 0)
        out: dict[str, Any] = {"safety_verdict": verdict}
        if verdict.requires_human:
            hitl += 1
            out["hitl_flags"] = hitl
            if hitl_mode == "block":
                verdict.allowed = False
                verdict.reason = f"{verdict.reason} (no operator attached: blocked)"
        if not verdict.allowed:
            blocked += 1
            out["blocked"] = blocked
            logger.bind(trace_id=trace_id).warning(f"safety blocked {action.summary()}: {verdict.reason}")
        log_node(trace_id, "safety", "out", {"allowed": verdict.allowed, "reason": verdict.reason})
        return out

    async def execute(state: LoopState) -> dict[str, Any]:
        trace_id = state["trace_id"]
        action = state["action"]
        assert action is not None
        observation = state.get("observation")
        verdict = state.get("safety_verdict")
        step_n = state.get("step", 0)
        steps_taken = list(state.get("steps_taken", []))
        tokens_in, tokens_out, cost = state.get("step_tokens", (0, 0, 0.0))
        started = state.get("step_started", time.perf_counter())
        finished = False
        success = state.get("success", False)
        final_answer = state.get("final_answer", "")
        if verdict is not None and not verdict.allowed:
            result = "blocked by safety layer"
            finished = True
            success = False
        else:
            result = await vm.execute(action)
            if action.type == "task_complete":
                finished = True
                success = True
                final_answer = action.text or ""
        steps_taken.append(
            Step(
                step=step_n,
                action=action,
                screenshot_path=observation.screenshot_path if observation else None,
                reasoning=state.get("reasoning", ""),
                execution_result=result,
                safety_blocked=bool(verdict is not None and not verdict.allowed),
                safety_reason=(verdict.reason or None) if verdict is not None else None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
            )
        )
        if step_n >= state.get("max_steps", 30) and not finished:
            finished = True
            success = False
            logger.bind(trace_id=trace_id).warning("max_steps reached without task_complete")
        out: dict[str, Any] = {
            "steps_taken": steps_taken,
            "finished": finished,
            "success": success,
            "final_answer": final_answer,
        }
        log_node(trace_id, "execute", "out", out)
        return out

    def route(state: LoopState) -> str:
        return END if state.get("finished") else "observe"

    graph: StateGraph = StateGraph(LoopState)
    graph.add_node("observe", observe)
    graph.add_node("verify", verify)
    graph.add_node("reason", reason)
    graph.add_node("safety", safety)
    graph.add_node("execute", execute)
    graph.set_entry_point("observe")
    graph.add_edge("observe", "verify")
    graph.add_edge("verify", "reason")
    graph.add_edge("reason", "safety")
    graph.add_edge("safety", "execute")
    graph.add_conditional_edges("execute", route, {END: END, "observe": "observe"})
    return graph.compile()


def build_reasoner(settings: Settings, *, force_stub: bool = False) -> Reasoner:
    """Return the Claude reasoner when a key is configured, else the stub."""
    if force_stub or not settings.llm_enabled:
        return StubReasoner()
    llm = LLMClient(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_tokens,
    )
    return ClaudeComputerUseReasoner(
        llm,
        tool_version=settings.computer_use_tool,
        display=(settings.display_width, settings.display_height),
    )


def build_verifier(settings: Settings, *, force_heuristic: bool = False) -> Verifier:
    """Return the Claude verifier when a key is configured, else the heuristic one."""
    if force_heuristic or not settings.llm_enabled:
        return HeuristicVerifier()
    llm = LLMClient(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        max_tokens=256,
    )
    return ClaudeVerifier(llm)


async def run_task(
    *,
    task: str,
    max_steps: int | None = None,
    task_spec: dict[str, Any] | None = None,
    reasoner: Reasoner | None = None,
    vm: VM | None = None,
    verifier: Verifier | None = None,
    safety_enabled: bool = True,
    hitl_mode: HitlMode = "block",
    audit: AuditLog | None = None,
    settings: Settings | None = None,
) -> TaskResponse:
    """Run one task end to end and return the audit-ready response.

    Args:
        task: Natural-language task.
        max_steps: Loop budget (defaults to ``Settings.max_steps_per_task``).
        task_spec: Optional eval task (``app``/``start``/``checker``/``plan``).
            When present the VM is reset to the app and the ground-truth checker
            decides ``success``; otherwise ``success`` means the loop ended with
            ``task_complete`` and no safety block.
        reasoner: Action policy override (default built from settings).
        vm: Environment driver override (default built from settings).
        verifier: Per-step verifier override (``None`` disables verification).
        safety_enabled: Disable only for the safety ablation.
        hitl_mode: Unattended policy for HITL-flagged actions.
        audit: Optional audit log sink.
        settings: Settings override.
    """
    s = settings or get_settings()
    configure_langsmith(s)
    trace_id = new_trace_id()
    eff_max = max_steps or (task_spec or {}).get("max_steps") or s.max_steps_per_task
    own_vm = vm is None
    vm = vm or build_vm(s)
    reasoner = reasoner or build_reasoner(s)
    reasoner.reset(task_spec)
    spec_id = (task_spec or {}).get("id")
    log = logger.bind(trace_id=trace_id)
    log.info(f"run_task start task_id={spec_id} reasoner={reasoner.name} vm={vm.name} max_steps={eff_max}")
    t0 = time.perf_counter()
    try:
        await vm.reset(task_spec)
        graph = build_graph(
            vm, reasoner, verifier=verifier, safety_enabled=safety_enabled, hitl_mode=hitl_mode
        )
        initial: LoopState = {
            "trace_id": trace_id,
            "task": task,
            "step": 0,
            "max_steps": int(eff_max),
            "steps_taken": [],
            "blocked": 0,
            "hitl_flags": 0,
            "verifier_failures": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        state = await graph.ainvoke(initial, config={"recursion_limit": int(eff_max) * 6 + 10})
        loop_success = bool(state.get("success"))
        final_answer = str(state.get("final_answer", ""))
        checked = False
        success = loop_success
        if task_spec and task_spec.get("checker"):
            checked = True
            success = loop_success and await vm.check(task_spec["checker"], final_answer)
    finally:
        if own_vm:
            await vm.close()
    failures = int(state.get("verifier_failures", 0))
    response = TaskResponse(
        trace_id=trace_id,
        task=task,
        task_id=spec_id,
        success=success,
        ground_truth_checked=checked,
        steps=state.get("steps_taken", []),
        final_output=final_answer if loop_success else "max_steps_or_blocked",
        total_steps=int(state.get("step", 0)),
        blocked_actions=int(state.get("blocked", 0)),
        verifier_failures=failures,
        recovered=bool(success and failures > 0),
        latency_ms=int((time.perf_counter() - t0) * 1000),
        tokens_in=int(state.get("tokens_in", 0)),
        tokens_out=int(state.get("tokens_out", 0)),
        cost_usd=float(state.get("cost_usd", 0.0)),
        reasoner=reasoner.name,
        vm=vm.name,
    )
    log.info(
        f"run_task end success={response.success} steps={response.total_steps} "
        f"latency_ms={response.latency_ms} tokens={response.tokens_in}/{response.tokens_out} "
        f"cost_usd={response.cost_usd}"
    )
    if audit is not None:
        audit.record(response)
    return response
