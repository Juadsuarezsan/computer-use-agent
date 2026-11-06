"""Record ``(screenshot, task, history) -> action`` pairs from the scripted oracle.

The :class:`~src.agent.reasoner.StubReasoner` completes the 20 tasks by
resolving CSS selectors to pixels, so every action it emits on a successful run
is a correct label. The recorder runs it on task variants inside the real
Playwright sandbox and keeps a trajectory only when the ground-truth checker of
the variant passes. No label is written by hand and no API is called.

Output layout (``out_dir``)::

    images/<task>_v<index>_<step>.png
    train.jsonl   val.jsonl   test.jsonl
    manifest.json             # arguments, counts, discarded runs, provenance
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.orchestrator import run_task
from src.agent.playwright_vm import PlaywrightVM
from src.agent.protocols import VM, Observation, Reasoner, ReasonerOutput
from src.agent.reasoner import StubReasoner
from src.api.schemas import Action
from src.config import Settings, get_settings
from src.eval.tasks import TaskSpec
from src.local_model.actions import encode_action, png_size
from src.local_model.variants import (
    DEFAULT_HELDOUT,
    DEFAULT_VARIANTS_PER_TASK,
    Perturbation,
    VariantSpec,
    make_variant,
    split_of,
    typed_literals,
)

DEFAULT_HISTORY = 4
"""Previous actions shown to the model (most recent last)."""

POINTER_ACTIONS = {"click", "double_click", "right_click", "mouse_move"}


class PerturbedPlaywrightVM(PlaywrightVM):
    """Playwright sandbox that applies a :class:`Perturbation` after every reset."""

    name = "playwright-perturbed"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.perturbation = Perturbation()

    async def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Load the app, then resize the viewport and inject the perturbation CSS."""
        await super().reset(spec)
        page = self._page
        p = self.perturbation
        await page.set_viewport_size({"width": p.width, "height": p.height})
        css = p.css()
        if css:
            await page.add_style_tag(content=css)
        await page.wait_for_timeout(50)


class RecordingReasoner:
    """Wraps a reasoner and keeps every ``(observation, action)`` it produces."""

    def __init__(self, inner: Reasoner) -> None:
        self.inner = inner
        self.name = f"recording({inner.name})"
        self.steps: list[tuple[Observation, Action]] = []

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Forget the recorded steps and reset the wrapped reasoner."""
        self.steps = []
        self.inner.reset(spec)

    def feedback(self, note: str) -> None:
        """Forward verifier notes."""
        self.inner.feedback(note)

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        """Delegate and record."""
        out = await self.inner.next_action(task, observation, step)
        self.steps.append((observation, out.action))
        return out


def _target_box(observation: Observation, action: Action, width: int, height: int) -> list[float] | None:
    """Relative bounding box ``[x0, y0, x1, y1]`` of the element a pointer action hits."""
    if action.type not in POINTER_ACTIONS or action.coords is None:
        return None
    x, y = action.coords
    for el in observation.elements:
        if el.w and el.h and el.contains(x, y):
            return [
                round((el.x - el.w / 2) / width, 4),
                round((el.y - el.h / 2) / height, 4),
                round((el.x + el.w / 2) / width, 4),
                round((el.y + el.h / 2) / height, 4),
            ]
    return None


def trajectory_records(
    steps: list[tuple[Observation, Action]],
    *,
    task: str,
    task_id: str,
    variant: int,
    split: str,
    images_dir: Path | None,
    history_len: int = DEFAULT_HISTORY,
    default_size: tuple[int, int] = (1024, 768),
) -> list[dict[str, Any]]:
    """Turn a recorded run into JSONL records (and write its screenshots).

    ``screenshot`` actions are not model outputs and are skipped; they do not
    enter the history either.
    """
    records: list[dict[str, Any]] = []
    history: list[str] = []
    for n, (observation, action) in enumerate(steps, start=1):
        if action.type == "screenshot":
            continue
        width, height = png_size(observation.png) if observation.png else default_size
        target = encode_action(action, width, height)
        image: str | None = None
        if observation.png and images_dir is not None:
            name = f"{task_id}_v{variant:03d}_{n:02d}.png"
            (images_dir / name).write_bytes(observation.png)
            image = f"images/{name}"
        records.append(
            {
                "id": f"{task_id}_v{variant:03d}_{n:02d}",
                "task_id": task_id,
                "variant": variant,
                "step": n,
                "split": split,
                "image": image,
                "width": width,
                "height": height,
                "task": task,
                "history": history[-history_len:] if history_len else [],
                "target": target,
                "action_type": action.type,
                "target_box": _target_box(observation, action, width, height),
            }
        )
        history.append(target)
    return records


@dataclass
class DatasetReport:
    """Counts written to ``manifest.json``."""

    trajectories: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})
    pairs: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})
    visual_only_fallbacks: int = 0
    discarded: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)


async def _record_variant(
    variant: VariantSpec, vm: VM, settings: Settings
) -> tuple[bool, list[tuple[Observation, Action]]]:
    if isinstance(vm, PerturbedPlaywrightVM):
        vm.perturbation = variant.perturbation
    recorder = RecordingReasoner(StubReasoner())
    response = await run_task(
        task=str(variant.spec["task"]),
        task_spec=variant.spec,
        reasoner=recorder,
        vm=vm,
        verifier=None,
        settings=settings,
    )
    ok = bool(response.success and response.ground_truth_checked)
    return ok, recorder.steps


async def generate_dataset(
    tasks: list[TaskSpec],
    out_dir: Path,
    *,
    variants_per_task: int = DEFAULT_VARIANTS_PER_TASK,
    seed: int = 20261106,
    heldout: tuple[str, ...] = DEFAULT_HELDOUT,
    heldout_apps: tuple[str, ...] = (),
    val_fraction: float = 0.1,
    history_len: int = DEFAULT_HISTORY,
    settings: Settings | None = None,
    vm: VM | None = None,
) -> DatasetReport:
    """Record the dataset and return its report (also written to ``manifest.json``).

    Args:
        tasks: Task catalogue (``load_tasks()``).
        out_dir: Output directory (created; existing JSONL files are overwritten).
        variants_per_task: Variants recorded per task (index 1..n).
        seed: Seed of the variants and of the train/val assignment.
        heldout: Task ids reserved for ``test`` (never trained on).
        heldout_apps: Whole fixture apps reserved for ``test``.
        val_fraction: Share of training trajectories moved to ``val``.
        history_len: Previous actions kept in each record.
        settings: Settings override.
        vm: VM override (tests); defaults to the perturbed Playwright sandbox.
    """
    s = settings or get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    handles = {
        name: (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("train", "val", "test")
    }
    report = DatasetReport()
    own_vm = vm is None
    vm = vm or PerturbedPlaywrightVM(
        chromium_path=s.chromium_path,
        webapps_dir=s.webapps_dir,
        screenshot_dir=str(out_dir / "_raw"),
        width=s.display_width,
        height=s.display_height,
    )
    apps = set(heldout_apps)
    try:
        for task in tasks:
            spec = task.as_spec()
            base_split = split_of(task.id, heldout, apps, task.app)
            can_substitute = bool(typed_literals(spec))
            for index in range(1, variants_per_task + 1):
                attempts = [True, False] if can_substitute else [False]
                recorded: tuple[VariantSpec, list[tuple[Observation, Action]]] | None = None
                for substitute in attempts:
                    variant = make_variant(spec, index, seed, substitute=substitute)
                    ok, steps = await _record_variant(variant, vm, s)
                    if ok:
                        recorded = (variant, steps)
                        if can_substitute and not substitute:
                            report.visual_only_fallbacks += 1
                        break
                    report.discarded.append(
                        {
                            "task_id": task.id,
                            "variant": index,
                            "substituted": variant.substituted,
                            "perturbation": variant.perturbation.as_dict(),
                        }
                    )
                if recorded is None:
                    logger.warning(f"{task.id} v{index}: no successful variant, skipped")
                    continue
                variant, steps = recorded
                split = base_split
                if (
                    split == "train"
                    and random.Random(f"{seed}:val:{task.id}:{index}").random() < val_fraction
                ):
                    split = "val"
                records = trajectory_records(
                    steps,
                    task=str(variant.spec["task"]),
                    task_id=task.id,
                    variant=index,
                    split=split,
                    images_dir=images_dir,
                    history_len=history_len,
                    default_size=(s.display_width, s.display_height),
                )
                for record in records:
                    handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                report.trajectories[split] += 1
                report.pairs[split] += len(records)
                report.provenance.append(
                    {
                        "task_id": task.id,
                        "variant": index,
                        "split": split,
                        "pairs": len(records),
                        "substituted": variant.substituted,
                        "perturbation": variant.perturbation.as_dict(),
                    }
                )
            logger.info(f"{task.id}: done ({base_split})")
    finally:
        for handle in handles.values():
            handle.close()
        if own_vm:
            await vm.close()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "variants_per_task": variants_per_task,
        "heldout_tasks": list(heldout),
        "heldout_apps": list(heldout_apps),
        "val_fraction": val_fraction,
        "history_len": history_len,
        "trajectories": report.trajectories,
        "pairs": report.pairs,
        "visual_only_fallbacks": report.visual_only_fallbacks,
        "discarded_runs": len(report.discarded),
        "discarded": report.discarded,
        "provenance": report.provenance,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(f"dataset written to {out_dir}: pairs={report.pairs} discarded={len(report.discarded)}")
    return report
