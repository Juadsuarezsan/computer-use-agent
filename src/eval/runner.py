"""Eval harness: run the 20 tasks under a configuration and aggregate metrics.

Metrics per category (spec table): success rate, average steps, latency and
cost per task; plus steps-to-completion distribution, recovery rate and
blocked actions. Every run is saved to ``eval/runs/<date>-<name>.json`` and
``eval/RESULTS.md`` is regenerated from those files only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.orchestrator import build_reasoner, build_verifier, run_task
from src.agent.protocols import VM, Observation, Reasoner, ReasonerOutput
from src.agent.vm_executor import FakeVM
from src.api.schemas import Action
from src.config import REPO_ROOT, Settings, get_settings
from src.eval.tasks import CATEGORIES, CATEGORY_LABELS, TaskSpec, load_tasks

RUNS_DIR = REPO_ROOT / "eval" / "runs"
RESULTS_MD = REPO_ROOT / "eval" / "RESULTS.md"
PENDING_KEY = "pendiente (requiere ANTHROPIC_API_KEY)"


@dataclass
class SuiteConfig:
    """One eval configuration (a row of the results table).

    Attributes:
        name: File-name friendly id (``stub-playwright``).
        label: Human label shown in RESULTS.md.
        reasoner: ``stub`` or ``claude``.
        vm: ``playwright`` or ``fake``.
        verifier: Enable the per-step verifier.
        safety: Enable the safety layer.
        max_steps: Optional budget override.
        task_ids: Optional subset.
        note: Free text about what the row measures.
    """

    name: str
    label: str
    reasoner: str = "stub"
    vm: str = "playwright"
    verifier: bool = True
    safety: bool = True
    max_steps: int | None = None
    task_ids: list[str] | None = None
    note: str = ""


@dataclass
class TaskResult:
    """Flat per-task record stored in the run file."""

    id: str
    category: str
    task: str
    success: bool
    ground_truth_checked: bool
    steps: int
    blocked: int
    verifier_failures: int
    recovered: bool
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    final_output: str
    trace_id: str
    actions: list[str] = field(default_factory=list)


def _build_vm(kind: str, settings: Settings) -> VM:
    if kind == "fake":
        return FakeVM()
    from src.agent.playwright_vm import PlaywrightVM

    return PlaywrightVM(
        chromium_path=settings.chromium_path,
        webapps_dir=settings.webapps_dir,
        screenshot_dir=settings.screenshot_dir,
        width=settings.display_width,
        height=settings.display_height,
    )


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(statistics.mean(ordered), 3),
        "median": round(statistics.median(ordered), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "p95": round(ordered[p95_index], 3),
    }


def aggregate(results: list[TaskResult]) -> dict[str, Any]:
    """Aggregate per-task results globally and per category."""

    def _agg(items: list[TaskResult]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {"n": 0}
        successes = sum(r.success for r in items)
        recovered = sum(r.recovered for r in items)
        failed_with_failures = sum(1 for r in items if r.verifier_failures > 0)
        return {
            "n": n,
            "success_rate": round(successes / n, 4),
            "avg_steps": round(statistics.mean(r.steps for r in items), 2),
            "steps": _summary([float(r.steps) for r in items]),
            "avg_latency_ms": round(statistics.mean(r.latency_ms for r in items), 1),
            "latency_ms": _summary([float(r.latency_ms) for r in items]),
            "avg_cost_usd": round(statistics.mean(r.cost_usd for r in items), 6),
            "total_cost_usd": round(sum(r.cost_usd for r in items), 6),
            "tokens_in": sum(r.tokens_in for r in items),
            "tokens_out": sum(r.tokens_out for r in items),
            "blocked_total": sum(r.blocked for r in items),
            "verifier_failure_tasks": failed_with_failures,
            "recovery_rate": round(recovered / failed_with_failures, 4) if failed_with_failures else None,
        }

    return {
        "overall": _agg(results),
        "by_category": {cat: _agg([r for r in results if r.category == cat]) for cat in CATEGORIES},
    }


async def run_suite(
    config: SuiteConfig,
    *,
    settings: Settings | None = None,
    tasks: list[TaskSpec] | None = None,
    reasoner: Reasoner | None = None,
    vm: VM | None = None,
) -> dict[str, Any]:
    """Run every task under ``config`` and return a serialisable report.

    Args:
        config: Suite configuration.
        settings: Settings override.
        tasks: Task subset (defaults to the catalogue).
        reasoner: Reasoner override (tests).
        vm: VM override (tests).
    """
    s = settings or get_settings()
    catalogue = tasks or load_tasks(s.tasks_file)
    if config.task_ids:
        catalogue = [t for t in catalogue if t.id in set(config.task_ids)]
    own_vm = vm is None
    vm = vm or _build_vm(config.vm, s)
    reasoner = reasoner or build_reasoner(s, force_stub=config.reasoner == "stub")
    verifier = build_verifier(s, force_heuristic=config.reasoner == "stub") if config.verifier else None
    started = datetime.now(UTC)
    results: list[TaskResult] = []
    try:
        for spec in catalogue:
            logger.info(f"[{config.name}] running {spec.id} ({spec.category})")
            response = await run_task(
                task=spec.task,
                max_steps=config.max_steps or spec.max_steps,
                task_spec=spec.as_spec(),
                reasoner=reasoner,
                vm=vm,
                verifier=verifier,
                safety_enabled=config.safety,
                settings=s,
            )
            results.append(
                TaskResult(
                    id=spec.id,
                    category=spec.category,
                    task=spec.task,
                    success=response.success,
                    ground_truth_checked=response.ground_truth_checked,
                    steps=response.total_steps,
                    blocked=response.blocked_actions,
                    verifier_failures=response.verifier_failures,
                    recovered=response.recovered,
                    latency_ms=response.latency_ms,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    cost_usd=response.cost_usd,
                    final_output=response.final_output[:200],
                    trace_id=response.trace_id,
                    actions=[st.action.summary() for st in response.steps],
                )
            )
    finally:
        if own_vm:
            await vm.close()
    report = {
        "name": config.name,
        "label": config.label,
        "config": asdict(config),
        "reasoner": reasoner.name,
        "vm": vm.name,
        "model": s.anthropic_model if config.reasoner == "claude" else None,
        "llm_used": config.reasoner == "claude",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "n_tasks": len(results),
        "aggregate": aggregate(results),
        "results": [asdict(r) for r in results],
    }
    return report


class AdversarialReasoner:
    """Emits a fixed list of dangerous actions, then ``task_complete`` (safety ablation)."""

    name = "adversarial"

    def __init__(self, actions: list[Action]) -> None:
        self._actions = actions
        self._i = 0

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Restart the sequence."""
        self._i = 0

    def feedback(self, note: str) -> None:
        """Ignored."""

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        """Return the next scripted dangerous action."""
        if self._i >= len(self._actions):
            return ReasonerOutput(Action(type="task_complete", text="done"), "adversarial: finished")
        action = self._actions[self._i]
        self._i += 1
        return ReasonerOutput(action, f"adversarial action {self._i}: {action.summary()}")


def load_adversarial_actions(path: Path | None = None) -> list[dict[str, Any]]:
    """Load ``data/eval/adversarial.json`` (dangerous actions with expected rule)."""
    file = path or REPO_ROOT / "data" / "eval" / "adversarial.json"
    payload = json.loads(file.read_text(encoding="utf-8"))
    return list(payload["actions"])


async def run_safety_ablation(
    *, settings: Settings | None = None, vm: VM | None = None, vm_kind: str = "playwright"
) -> dict[str, Any]:
    """Run every adversarial action with and without the safety layer.

    Each action is executed in its own short task so a block does not hide the
    following actions. Reports how many dangerous actions were blocked vs executed.
    """
    s = settings or get_settings()
    cases = load_adversarial_actions()
    own_vm = vm is None
    vm = vm or _build_vm(vm_kind, s)
    spec = {"id": "adv", "app": "forms.html", "start": "#support", "max_steps": 3}
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            action = Action.model_validate(case["action"])
            for safety in (True, False):
                response = await run_task(
                    task="adversarial probe",
                    max_steps=3,
                    task_spec={**spec, "plan": []},
                    reasoner=AdversarialReasoner([action]),
                    vm=vm,
                    verifier=None,
                    safety_enabled=safety,
                    settings=s,
                )
                first = response.steps[0] if response.steps else None
                rows.append(
                    {
                        "id": case["id"],
                        "action": action.summary(),
                        "expected_rule": case.get("expected_rule"),
                        "safety_enabled": safety,
                        "blocked": bool(first and first.safety_blocked),
                        "safety_reason": first.safety_reason if first else None,
                        "executed": bool(first and not first.safety_blocked),
                    }
                )
    finally:
        if own_vm:
            await vm.close()
    with_safety = [r for r in rows if r["safety_enabled"]]
    without = [r for r in rows if not r["safety_enabled"]]
    return {
        "name": "safety-ablation",
        "label": "Safety layer ablation (adversarial actions)",
        "llm_used": False,
        "vm": vm.name,
        "started_at": datetime.now(UTC).isoformat(),
        "n_actions": len(cases),
        "with_safety": {
            "blocked": sum(r["blocked"] for r in with_safety),
            "executed": sum(r["executed"] for r in with_safety),
            "block_rate": round(sum(r["blocked"] for r in with_safety) / max(1, len(with_safety)), 4),
        },
        "without_safety": {
            "blocked": sum(r["blocked"] for r in without),
            "executed": sum(r["executed"] for r in without),
            "block_rate": round(sum(r["blocked"] for r in without) / max(1, len(without)), 4),
        },
        "rows": rows,
    }


def save_run(report: dict[str, Any], *, runs_dir: Path = RUNS_DIR, day: date | None = None) -> Path:
    """Write a run report to ``eval/runs/<date>-<name>.json``."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = (day or date.today()).isoformat()
    path = runs_dir / f"{stamp}-{report['name']}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info(f"saved run to {path}")
    return path


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def render_results_md(
    suites: list[dict[str, Any]],
    *,
    safety: dict[str, Any] | None,
    run_files: list[str],
    model: str,
) -> str:
    """Render ``eval/RESULTS.md`` from saved reports only (no hand-written numbers)."""
    lines: list[str] = []
    lines.append("# Eval results — Computer Use Agent")
    lines.append("")
    lines.append(
        "Generated by `python -m eval.run`. Every number below comes from a run file in "
        "`eval/runs/` listed at the bottom; nothing is typed by hand."
    )
    lines.append("")
    lines.append(
        "> **Read this first.** The rows measured in this repository use the "
        "`StubReasoner`: a deterministic, scripted DOM-oracle baseline **without any LLM**. "
        "They measure the sandbox, the executor, the safety layer and the checkers, "
        "not the capability of the system. The Claude Computer Use rows are "
        f"`{PENDING_KEY}` and are filled only by a real run with `{model}`."
    )
    lines.append("")
    lines.append("## Mandatory table (per task category)")
    lines.append("")
    for suite in suites:
        agg = suite["aggregate"]["by_category"]
        tag = "fallback determinista, sin LLM" if not suite["llm_used"] else f"LLM: {suite['model']}"
        lines.append(f"### {suite['label']} — `{suite['name']}` ({tag})")
        lines.append("")
        if suite.get("config", {}).get("note"):
            lines.append(suite["config"]["note"])
            lines.append("")
        lines.append(
            "| Categoría de tarea | Success Rate | Avg Steps | Latencia (media / p95) | Costo/tarea |"
        )
        lines.append("|---|---|---|---|---|")
        for cat in CATEGORIES:
            row = agg[cat]
            if row.get("n", 0) == 0:
                lines.append(f"| {CATEGORY_LABELS[cat]} | n/a | n/a | n/a | n/a |")
                continue
            cost = f"${row['avg_cost_usd']:.4f}" if suite["llm_used"] else "$0.0000 (sin LLM)"
            lines.append(
                f"| {CATEGORY_LABELS[cat]} | {_fmt_pct(row['success_rate'])} ({int(row['success_rate'] * row['n'])}/{row['n']}) "
                f"| {row['avg_steps']:.1f} | {row['avg_latency_ms'] / 1000:.2f} s / {row['latency_ms']['p95'] / 1000:.2f} s | {cost} |"
            )
        overall = suite["aggregate"]["overall"]
        lines.append(
            f"| **Total** | **{_fmt_pct(overall['success_rate'])}** ({int(overall['success_rate'] * overall['n'])}/{overall['n']}) "
            f"| {overall['avg_steps']:.1f} | {overall['avg_latency_ms'] / 1000:.2f} s / {overall['latency_ms']['p95'] / 1000:.2f} s "
            f"| {'$' + format(overall['avg_cost_usd'], '.4f') if suite['llm_used'] else '$0.0000 (sin LLM)'} |"
        )
        lines.append("")
        steps = overall["steps"]
        lines.append(
            f"Steps to completion: mean {steps['mean']:.1f}, median {steps['median']:.0f}, "
            f"min {steps['min']:.0f}, max {steps['max']:.0f}, p95 {steps['p95']:.0f}. "
            f"Tasks with verifier failures: {overall['verifier_failure_tasks']}; "
            f"recovery rate: {_fmt_pct(overall['recovery_rate'])}; blocked actions: {overall['blocked_total']}."
        )
        lines.append("")
    lines.append(f"### Claude Computer Use (`{model}`, tool `computer_20250124`)")
    lines.append("")
    lines.append("| Categoría de tarea | Success Rate | Avg Steps | Latencia | Costo/tarea |")
    lines.append("|---|---|---|---|---|")
    for cat in CATEGORIES:
        lines.append(
            f"| {CATEGORY_LABELS[cat]} | {PENDING_KEY} | {PENDING_KEY} | {PENDING_KEY} | {PENDING_KEY} |"
        )
    lines.append("")
    lines.append(
        "Also pending with the key: agent **without verifier** vs **with Claude verifier** "
        "(ablation), recovery rate on real failures, and the LLM-as-judge final validator "
        "(rubric in `src/eval/judge.py`)."
    )
    lines.append("")
    if safety is not None:
        lines.append("## Ablation: safety layer on/off (measured, no LLM)")
        lines.append("")
        lines.append(
            f"{safety['n_actions']} adversarial actions from `data/eval/adversarial.json` were sent "
            f"through the loop on the `{safety['vm']}` sandbox, each once with the safety layer and once without."
        )
        lines.append("")
        lines.append(
            "| Configuración | Acciones peligrosas bloqueadas | Ejecutadas en el sandbox | Block rate |"
        )
        lines.append("|---|---|---|---|")
        ws, wo = safety["with_safety"], safety["without_safety"]
        lines.append(
            f"| Con safety layer | {ws['blocked']} | {ws['executed']} | {_fmt_pct(ws['block_rate'])} |"
        )
        lines.append(
            f"| Sin safety layer | {wo['blocked']} | {wo['executed']} | {_fmt_pct(wo['block_rate'])} |"
        )
        lines.append("")
    lines.append("## Baselines and ablations summary")
    lines.append("")
    lines.append("| Fila | Qué mide | Estado |")
    lines.append("|---|---|---|")
    for suite in suites:
        lines.append(f"| `{suite['name']}` | {suite['label']} | medido (sin LLM) |")
    if safety is not None:
        lines.append(
            "| `safety-ablation` | Safety layer on/off sobre acciones adversariales | medido (sin LLM) |"
        )
    lines.append(f"| `claude-playwright` | Agente completo con Claude Computer Use | {PENDING_KEY} |")
    lines.append(f"| `claude-no-verifier` | Ablation: agente sin verificador | {PENDING_KEY} |")
    lines.append("")
    lines.append("## Run files")
    lines.append("")
    for f in run_files:
        lines.append(f"- `{f}`")
    lines.append("")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    config = SuiteConfig(
        name=f"stub-{args.vm}",
        label=f"StubReasoner (scripted DOM oracle) on {args.vm} sandbox",
        reasoner="stub",
        vm=args.vm,
        verifier=not args.no_verifier,
        max_steps=args.max_steps,
    )
    return await run_suite(config, settings=settings)


def main() -> None:
    """CLI: ``python -m src.eval.runner [--vm fake|playwright] [--json]``."""
    parser = argparse.ArgumentParser(description="Run the 20-task eval with the offline stub reasoner.")
    parser.add_argument("--vm", choices=["fake", "playwright"], default="fake")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()
    report = asyncio.run(_main_async(args))
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, default=str) + "\n")
        return
    overall = report["aggregate"]["overall"]
    logger.info(
        f"[{report['name']}] tasks={overall['n']} success_rate={overall['success_rate']:.1%} "
        f"avg_steps={overall['avg_steps']:.1f} (fallback determinista, sin LLM)"
    )
    for cat, info in report["aggregate"]["by_category"].items():
        if info.get("n"):
            logger.info(f"  {cat:<18} success={info['success_rate']:.1%} avg_steps={info['avg_steps']:.1f}")


if __name__ == "__main__":
    main()
