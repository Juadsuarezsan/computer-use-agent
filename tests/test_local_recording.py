from __future__ import annotations

import json

from src.agent.protocols import Observation, UIElement
from src.agent.reasoner import StubReasoner
from src.api.schemas import Action
from src.eval.tasks import TaskSpec
from src.local_model.actions import parse_action
from src.local_model.recording import RecordingReasoner, generate_dataset, trajectory_records
from tests.test_local_actions import fake_png


def test_element_contains():
    el = UIElement("#b", "button", "OK", 100, 50, 40, 20)
    assert el.contains(80, 40) and el.contains(120, 60)
    assert not el.contains(121, 50)
    assert not UIElement("#c", "button", "X", 10, 10).contains(11, 10)  # unknown size


def test_trajectory_records_and_images(tmp_path):
    png = fake_png(1000, 500)
    button = UIElement("#ok", "button", "OK", 500, 250, 100, 50)
    steps = [
        (Observation(png=png, elements=[button]), Action(type="click", coords=(500, 250))),
        (Observation(png=png), Action(type="screenshot")),
        (Observation(png=png), Action(type="type", text="Alice")),
        (Observation(png=png), Action(type="task_complete", text="done")),
    ]
    images = tmp_path / "images"
    images.mkdir()
    records = trajectory_records(
        steps, task="t", task_id="ff-01", variant=2, split="train", images_dir=images, history_len=1
    )
    assert [r["action_type"] for r in records] == ["click", "type", "task_complete"]
    assert records[0]["target_box"] == [0.45, 0.45, 0.55, 0.55]
    assert records[1]["target_box"] is None
    assert records[0]["history"] == [] and len(records[2]["history"]) == 1  # truncated to 1
    assert records[2]["history"][0] == records[1]["target"]
    assert (tmp_path / records[0]["image"]).read_bytes() == png
    for r in records:
        action, error = parse_action(r["target"], r["width"], r["height"])
        assert error is None


async def test_recording_reasoner_records_and_forwards():
    inner = StubReasoner()
    rec = RecordingReasoner(inner)
    rec.reset({"plan": [{"action": "type", "text": "x"}]})
    await rec.next_action("t", Observation(), 1)
    rec.feedback("note")
    assert rec.name == "recording(stub)"
    assert len(rec.steps) == 1 and inner.feedback_notes == ["note"]
    rec.reset(None)
    assert rec.steps == []


async def test_generate_dataset_on_fake_vm(tmp_path, settings, fake_vm):
    spec = TaskSpec(
        id="ff-01",
        category="form_filling",
        task="Type 'Alice' into the input and finish.",
        app="forms.html",
        checker={"type": "answer_contains", "value": "done"},
        plan=[
            {"action": "click", "target": "#input"},
            {"action": "type", "text": "Alice"},
            {"action": "task_complete", "text": "done"},
        ],
        max_steps=6,
    )
    held = spec.model_copy(update={"id": "ff-05"})
    report = await generate_dataset(
        [spec, held], tmp_path, variants_per_task=3, seed=1, val_fraction=0.0, settings=settings, vm=fake_vm
    )
    assert report.trajectories == {"train": 3, "val": 0, "test": 3}
    assert report.pairs["train"] == 9
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    typed = [r for r in train if r["action_type"] == "type"]
    assert all("Alice" not in r["target"] for r in typed)  # values were substituted
    assert len({r["task"] for r in train}) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["heldout_tasks"][0] == "ff-05" and manifest["discarded_runs"] == 0


async def test_generate_dataset_falls_back_to_visual_only(tmp_path, settings, fake_vm):
    """A value fixed by the app (here: the checker wants 'dem') breaks after substitution."""
    spec = TaskSpec(
        id="wn-01",
        category="web_navigation",
        task="Log in with username 'demo' and finish.",
        app="dashboard.html",
        checker={"type": "answer_contains", "value": "dem"},
        plan=[
            {"action": "type", "text": "demo"},
            {"action": "task_complete", "text": "demo"},
        ],
        max_steps=6,
    )
    report = await generate_dataset(
        [spec], tmp_path, variants_per_task=2, seed=1, settings=settings, vm=fake_vm
    )
    assert report.visual_only_fallbacks == 2
    assert len(report.discarded) == 2
