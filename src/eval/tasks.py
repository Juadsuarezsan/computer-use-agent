"""Eval task catalogue: 20 hand-written tasks with deterministic ground truth.

Tasks live in ``data/eval/tasks.json`` (schema in ``docs/data_schema.md``).
Each task names a fixture app, a start route, a checker and an optional
scripted plan for the offline :class:`~src.agent.reasoner.StubReasoner`.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings
from src.sandbox.checkers import validate_checker

Category = Literal["form_filling", "data_extraction", "web_navigation", "multi_step"]
CATEGORIES: tuple[str, ...] = ("form_filling", "data_extraction", "web_navigation", "multi_step")
CATEGORY_LABELS: dict[str, str] = {
    "form_filling": "Form filling",
    "data_extraction": "Data extraction",
    "web_navigation": "Web navigation",
    "multi_step": "Multi-step workflow",
}
TASKS_PER_CATEGORY = 5


class TaskSpec(BaseModel):
    """One eval task.

    Attributes:
        id: Stable id (``ff-01`` ...).
        category: One of :data:`CATEGORIES`.
        task: Natural-language instruction given to the agent.
        app: Fixture HTML file in ``sandbox/webapps``.
        start: Route/hash to open first (``#contact``).
        checker: Ground-truth checker (see ``src/sandbox/checkers.py``).
        plan: Scripted plan for the stub reasoner (selectors, not pixels).
        max_steps: Step budget for the task.
        ground_truth_source: ``manual`` — written and verified by hand.
        notes: Free text about what makes the task hard.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^(ff|de|wn|ms)-\d{2}$")
    category: Category
    task: str = Field(min_length=10, max_length=500)
    app: str
    start: str = ""
    checker: dict[str, Any]
    plan: list[dict[str, Any]] = Field(default_factory=list)
    max_steps: int = Field(default=30, ge=1, le=100)
    ground_truth_source: str = "manual"
    notes: str = ""

    def as_spec(self) -> dict[str, Any]:
        """Plain dict for the orchestrator / VM."""
        return self.model_dump()


def load_tasks(path: str | Path | None = None) -> list[TaskSpec]:
    """Load and validate the task catalogue.

    Raises:
        ValueError: on duplicated ids, wrong category balance or invalid checkers.
        FileNotFoundError: when the JSON file or a fixture app is missing.
    """
    file = Path(path or get_settings().tasks_file)
    payload = json.loads(file.read_text(encoding="utf-8"))
    raw_tasks = payload["tasks"] if isinstance(payload, dict) else payload
    tasks = [TaskSpec.model_validate(t) for t in raw_tasks]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicated task ids in catalogue")
    counts: Counter[str] = Counter(str(t.category) for t in tasks)
    for category in CATEGORIES:
        if counts.get(category, 0) != TASKS_PER_CATEGORY:
            raise ValueError(
                f"category {category!r} has {counts.get(category, 0)} tasks, expected {TASKS_PER_CATEGORY}"
            )
    for task in tasks:
        validate_checker(task.checker)
    return tasks


@lru_cache(maxsize=1)
def default_tasks() -> tuple[TaskSpec, ...]:
    """Cached catalogue from the configured ``TASKS_FILE``."""
    return tuple(load_tasks())


def get_task(task_id: str) -> TaskSpec | None:
    """Return a task by id from the default catalogue."""
    for task in default_tasks():
        if task.id == task_id:
            return task
    return None
