from __future__ import annotations

from eval.run import _local_configs
from src.agent.local_reasoner import LocalVLMReasoner
from src.agent.vm_executor import FakeVM
from src.eval.runner import SuiteConfig, aggregate, render_results_md, run_suite
from tests.test_local_reasoner import ScriptedBackend
from tests.test_runner import _result


def test_aggregate_by_split():
    results = [_result(id="ff-01"), _result(id="ff-05", success=False)]
    agg = aggregate(results, heldout=["ff-05"])
    assert agg["by_split"]["seen"]["success_rate"] == 1.0
    assert agg["by_split"]["heldout"]["success_rate"] == 0.0
    assert "by_split" not in aggregate(results)


async def test_local_suite_reports_model_and_zero_cost(settings, tasks):
    config = SuiteConfig(
        name="local-lora-fake",
        label="local",
        reasoner="local",
        vm="fake",
        task_ids=["ff-01", "ff-05"],
        adapter="models/adapters/x",
        heldout=["ff-05"],
    )
    reasoner = LocalVLMReasoner(ScriptedBackend([]), label="local-lora-fake")
    report = await run_suite(config, settings=settings, tasks=tasks, reasoner=reasoner, vm=FakeVM())
    assert report["llm_used"] is False
    assert report["local_model"].endswith("+ LoRA models/adapters/x")
    assert report["aggregate"]["overall"]["total_cost_usd"] == 0.0
    assert report["aggregate"]["by_split"]["heldout"]["n"] == 1
    md = render_results_md([report], safety=None, run_files=[], model="m")
    assert "modelo local:" in md and "$0.0000 (modelo local)" in md
    assert "held-out tasks (never trained on)" in md
    assert "medido (modelo local, sin API)" in md


def test_base_label_without_adapter(settings):
    from src.eval.runner import _local_label

    assert _local_label(settings, SuiteConfig(name="a", label="a", reasoner="local", adapter="")).endswith(
        "(base, sin ajuste)"
    )


def test_local_configs():
    only_base = _local_configs("playwright", "", ["ff-05"])
    assert [c.name for c in only_base] == ["local-base-playwright"]
    full = _local_configs("playwright", "models/a", ["ff-05"])
    assert [c.name for c in full] == [
        "local-base-playwright",
        "local-lora-playwright",
        "local-lora-playwright-no-verifier",
    ]
    assert all(c.reasoner == "local" and c.heldout == ["ff-05"] for c in full)
    assert full[2].verifier is False
