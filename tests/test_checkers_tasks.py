from __future__ import annotations

import json
from typing import Any

import pytest

from src.eval.tasks import CATEGORIES, TASKS_PER_CATEGORY, get_task, load_tasks
from src.sandbox.checkers import check_answer_only, evaluate_checker, validate_checker


def test_answer_checkers():
    assert check_answer_only({"type": "answer_contains", "value": "4.2.1-legacy"}, "Version 4.2.1-LEGACY")
    assert check_answer_only({"type": "answer_number", "value": 48250}, "Total: 48,250.00 USD")
    assert not check_answer_only({"type": "answer_number", "value": 48250}, "no numbers here")
    assert not check_answer_only({"type": "answer_number", "value": 1}, "1.5")
    assert check_answer_only(
        {"type": "answer_set", "pattern": r"[\w.]+@[\w.]+", "values": ["a@b.io", "c@d.io"]},
        "c@d.io and a@b.io",
    )
    assert not check_answer_only({"type": "answer_set", "pattern": r"\d", "values": ["1", "2"]}, "1")
    assert check_answer_only({"type": "any", "checkers": [{"type": "answer_contains", "value": "x"}]}, "x")
    assert not check_answer_only({"type": "all", "checkers": [{"type": "hash_equals", "value": "#a"}]}, "x")


class FakePage:
    """Minimal stand-in for a Playwright page: a dict of selector -> (attrs, text) and a hash."""

    def __init__(self, dom: dict[str, tuple[dict[str, str], str]], hash_: str = "#home") -> None:
        self.dom = dom
        self.hash = hash_

    async def evaluate(self, js: str, arg: Any = None) -> Any:
        if "location.hash" in js:
            return self.hash
        if "!!document.querySelector" in js:
            return arg in self.dom
        selector, attr = arg
        if selector not in self.dom:
            return None
        attrs, text = self.dom[selector]
        return attrs.get(attr) if attr else text


async def test_evaluate_checker_against_fake_page():
    page = FakePage({"#r": ({"data-name": "Alice"}, "Thanks Alice")}, "#contact")
    assert await evaluate_checker(page, {"type": "hash_equals", "value": "#contact"}, None)
    assert await evaluate_checker(page, {"type": "dom_exists", "selector": "#r"}, None)
    assert not await evaluate_checker(page, {"type": "dom_exists", "selector": "#zz"}, None)
    assert await evaluate_checker(
        page, {"type": "dom_attr_equals", "selector": "#r", "attr": "data-name", "value": "Alice"}, None
    )
    assert not await evaluate_checker(
        page, {"type": "dom_attr_equals", "selector": "#zz", "attr": "x", "value": "1"}, None
    )
    assert await evaluate_checker(
        page, {"type": "dom_text_contains", "selector": "#r", "value": "alice"}, None
    )
    assert await evaluate_checker(
        page, {"type": "dom_text_equals", "selector": "#r", "value": "thanks  alice"}, None
    )
    assert await evaluate_checker(
        page,
        {
            "type": "all",
            "checkers": [
                {"type": "hash_equals", "value": "#contact"},
                {"type": "answer_contains", "value": "ok"},
            ],
        },
        "OK",
    )
    assert not await evaluate_checker(
        page,
        {
            "type": "all",
            "checkers": [{"type": "hash_equals", "value": "#x"}, {"type": "dom_exists", "selector": "#r"}],
        },
        None,
    )
    assert await evaluate_checker(
        page,
        {
            "type": "any",
            "checkers": [{"type": "hash_equals", "value": "#x"}, {"type": "dom_exists", "selector": "#r"}],
        },
        None,
    )
    assert not await evaluate_checker(
        page, {"type": "any", "checkers": [{"type": "hash_equals", "value": "#x"}]}, None
    )
    with pytest.raises(ValueError):
        await evaluate_checker(page, {"type": "magic"}, None)


def test_validate_checker_errors():
    with pytest.raises(ValueError):
        validate_checker({"type": "nope"})
    with pytest.raises(ValueError):
        validate_checker({"type": "all", "checkers": []})
    with pytest.raises(ValueError):
        validate_checker({"type": "dom_attr_equals", "selector": "#a"})
    validate_checker({"type": "all", "checkers": [{"type": "hash_equals", "value": "#a"}]})


def test_catalogue_has_20_balanced_tasks(tasks):
    assert len(tasks) == 20
    for category in CATEGORIES:
        assert sum(1 for t in tasks if t.category == category) == TASKS_PER_CATEGORY
    assert all(t.ground_truth_source == "manual" for t in tasks)
    assert all(t.plan for t in tasks), "every task needs a scripted plan for the offline baseline"
    apps = {t.app for t in tasks}
    assert apps == {"forms.html", "legacy.html", "dashboard.html", "workflow.html"}


def test_catalogue_fixture_apps_exist(tasks):
    from src.config import REPO_ROOT

    for t in tasks:
        assert (REPO_ROOT / "sandbox" / "webapps" / t.app).exists()


def test_get_task_and_missing():
    assert get_task("ff-01") is not None
    assert get_task("zz-99") is None


def test_load_tasks_rejects_unbalanced_and_duplicates(tmp_path, tasks):
    payload = {"tasks": [t.model_dump() for t in tasks[:19]]}
    file = tmp_path / "t.json"
    file.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_tasks(file)
    dup = [t.model_dump() for t in tasks]
    dup[1]["id"] = dup[0]["id"]
    file.write_text(json.dumps({"tasks": dup}))
    with pytest.raises(ValueError):
        load_tasks(file)
    bad = [t.model_dump() for t in tasks]
    bad[0]["checker"] = {"type": "nope"}
    file.write_text(json.dumps({"tasks": bad}))
    with pytest.raises(ValueError):
        load_tasks(file)
