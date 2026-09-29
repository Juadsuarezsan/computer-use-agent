from __future__ import annotations

import json

from src.agent.protocols import Observation
from src.agent.vm_executor import FakeVM
from src.api.schemas import Action
from src.eval.runner import (
    AdversarialReasoner,
    SuiteConfig,
    TaskResult,
    aggregate,
    load_adversarial_actions,
    render_results_md,
    run_safety_ablation,
    run_suite,
    save_run,
)


def _result(**kw) -> TaskResult:
    base = dict(
        id="ff-01",
        category="form_filling",
        task="t",
        success=True,
        ground_truth_checked=True,
        steps=5,
        blocked=0,
        verifier_failures=0,
        recovered=False,
        latency_ms=100,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        final_output="ok",
        trace_id="x",
    )
    base.update(kw)
    return TaskResult(**base)


def test_aggregate_metrics():
    results = [
        _result(),
        _result(id="ff-02", success=False, steps=9, latency_ms=300, verifier_failures=2),
        _result(id="de-01", category="data_extraction", steps=3, verifier_failures=1, recovered=True),
    ]
    agg = aggregate(results)
    assert agg["overall"]["n"] == 3
    assert agg["overall"]["success_rate"] == round(2 / 3, 4)
    assert agg["overall"]["steps"]["max"] == 9
    assert agg["overall"]["recovery_rate"] == 0.5
    assert agg["by_category"]["form_filling"]["n"] == 2
    assert agg["by_category"]["web_navigation"] == {"n": 0}
    assert aggregate([])["overall"] == {"n": 0}


async def test_run_suite_on_fake_vm(settings, tasks):
    config = SuiteConfig(name="t", label="t", vm="fake", task_ids=["de-05", "ff-01"], max_steps=6)
    report = await run_suite(config, settings=settings, tasks=tasks, vm=FakeVM())
    assert report["n_tasks"] == 2
    assert report["llm_used"] is False and report["model"] is None
    ids = {r["id"] for r in report["results"]}
    assert ids == {"de-05", "ff-01"}
    # No DOM on the fake VM: checkers fail, and that is what the report must say.
    assert report["aggregate"]["overall"]["success_rate"] == 0.0
    assert all(r["ground_truth_checked"] for r in report["results"])


async def test_run_suite_without_verifier_builds_vm(settings, tasks):
    config = SuiteConfig(name="t", label="t", vm="fake", verifier=False, task_ids=["wn-04"], max_steps=4)
    report = await run_suite(config, settings=settings, tasks=tasks)
    assert report["vm"] == "fake" and report["n_tasks"] == 1


async def test_adversarial_reasoner_and_actions():
    cases = load_adversarial_actions()
    assert len(cases) >= 12
    r = AdversarialReasoner([Action(type="type", text="rm -rf /")])
    first = await r.next_action("t", Observation(), 1)
    second = await r.next_action("t", Observation(), 2)
    assert first.action.type == "type" and second.action.type == "task_complete"
    r.reset()
    r.feedback("x")
    assert (await r.next_action("t", Observation(), 1)).action.type == "type"


async def test_safety_ablation_on_fake_vm(settings):
    report = await run_safety_ablation(settings=settings, vm=FakeVM())
    n = report["n_actions"]
    assert report["with_safety"]["blocked"] + report["with_safety"]["executed"] == n
    # every adversarial case except HITL-flag-only ones must be blocked with safety on
    hitl = sum(1 for r in report["rows"] if r["safety_enabled"] and r["expected_rule"] == "hitl")
    assert report["with_safety"]["blocked"] == n  # unattended mode blocks HITL too
    assert hitl >= 1
    assert report["without_safety"]["blocked"] == 0
    assert report["without_safety"]["executed"] == n


def test_save_run_and_render(tmp_path):
    suite = {
        "name": "stub-fake",
        "label": "Stub",
        "llm_used": False,
        "model": None,
        "config": {"note": "note here"},
        "aggregate": aggregate([_result(), _result(id="de-01", category="data_extraction")]),
    }
    path = save_run(suite, runs_dir=tmp_path)
    assert path.exists() and json.loads(path.read_text())["name"] == "stub-fake"
    safety = {
        "n_actions": 2,
        "vm": "fake",
        "with_safety": {"blocked": 2, "executed": 0, "block_rate": 1.0},
        "without_safety": {"blocked": 0, "executed": 2, "block_rate": 0.0},
    }
    md = render_results_md([suite], safety=safety, run_files=[path.name], model="claude-sonnet-4-5-20250929")
    assert "| Form filling | 100% (1/1)" in md
    assert "| Web navigation | n/a" in md
    assert "pendiente (requiere ANTHROPIC_API_KEY)" in md
    assert "sin LLM" in md
    assert "| Con safety layer | 2 | 0 | 100% |" in md
    assert path.name in md
