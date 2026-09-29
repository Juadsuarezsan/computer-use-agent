"""Deterministic ground-truth checkers evaluated against the sandbox DOM.

A checker is a JSON object with a ``type`` (see :data:`CHECKER_TYPES`). DOM
checkers read the live page through Playwright; answer checkers look at the
text the agent returned with its ``task_complete`` action. ``all`` / ``any``
combine checkers.
"""

from __future__ import annotations

import re
from typing import Any

CHECKER_TYPES: dict[str, str] = {
    "dom_attr_equals": "attribute of the element at `selector` equals `value`",
    "dom_text_contains": "innerText/value of the element at `selector` contains `value`",
    "dom_text_equals": "innerText/value of the element at `selector` equals `value` (trimmed)",
    "dom_exists": "an element matching `selector` exists",
    "hash_equals": "location.hash equals `value`",
    "answer_contains": "the final answer contains `value` (case-insensitive)",
    "answer_number": "the first number in the final answer equals `value` (tolerance 0.005)",
    "answer_set": "the set of tokens matching `pattern` in the answer equals `values`",
    "all": "every checker in `checkers` passes",
    "any": "at least one checker in `checkers` passes",
}

ANSWER_TYPES = {"answer_contains", "answer_number", "answer_set"}


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def check_answer_only(checker: dict[str, Any], answer: str | None) -> bool:
    """Evaluate a checker that needs no DOM (answer checkers, combinators).

    DOM checkers evaluate to ``False`` here: without a page there is no ground truth.
    """
    kind = checker.get("type")
    if kind == "all":
        return all(check_answer_only(c, answer) for c in checker.get("checkers", []))
    if kind == "any":
        return any(check_answer_only(c, answer) for c in checker.get("checkers", []))
    if kind == "answer_contains":
        return _norm(str(checker["value"])) in _norm(answer)
    if kind == "answer_number":
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", answer or "")
        if not match:
            return False
        try:
            found = float(match.group(0).replace(",", ""))
        except ValueError:
            return False
        return abs(found - float(checker["value"])) <= 0.005
    if kind == "answer_set":
        tokens = {t.lower() for t in re.findall(str(checker["pattern"]), answer or "")}
        return tokens == {str(v).lower() for v in checker["values"]}
    return False


async def _dom_value(page: Any, selector: str, attr: str | None) -> str | None:
    js = """([sel, attr]) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      if (attr) return el.getAttribute(attr);
      if ('value' in el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT')) return el.value;
      return el.innerText;
    }"""
    value = await page.evaluate(js, [selector, attr])
    return None if value is None else str(value)


async def evaluate_checker(page: Any, checker: dict[str, Any], answer: str | None) -> bool:
    """Evaluate any checker against a Playwright ``page`` and the final answer."""
    kind = checker.get("type")
    if kind in ANSWER_TYPES:
        return check_answer_only(checker, answer)
    if kind == "all":
        for sub in checker.get("checkers", []):
            if not await evaluate_checker(page, sub, answer):
                return False
        return True
    if kind == "any":
        for sub in checker.get("checkers", []):
            if await evaluate_checker(page, sub, answer):
                return True
        return False
    if kind == "hash_equals":
        current = await page.evaluate("() => location.hash")
        return str(current) == str(checker["value"])
    if kind == "dom_exists":
        return bool(await page.evaluate("(sel) => !!document.querySelector(sel)", checker["selector"]))
    if kind == "dom_attr_equals":
        value = await _dom_value(page, str(checker["selector"]), str(checker["attr"]))
        return value is not None and value.strip() == str(checker["value"]).strip()
    if kind == "dom_text_contains":
        value = await _dom_value(page, str(checker["selector"]), None)
        return value is not None and _norm(str(checker["value"])) in _norm(value)
    if kind == "dom_text_equals":
        value = await _dom_value(page, str(checker["selector"]), None)
        return value is not None and _norm(value) == _norm(str(checker["value"]))
    raise ValueError(f"unknown checker type: {kind!r}")


def validate_checker(checker: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a checker is malformed (used by the task loader)."""
    kind = checker.get("type")
    if kind not in CHECKER_TYPES:
        raise ValueError(f"unknown checker type: {kind!r}")
    if kind in {"all", "any"}:
        subs = checker.get("checkers")
        if not isinstance(subs, list) or not subs:
            raise ValueError(f"{kind} checker needs a non-empty 'checkers' list")
        for sub in subs:
            validate_checker(sub)
        return
    required = {
        "dom_attr_equals": {"selector", "attr", "value"},
        "dom_text_contains": {"selector", "value"},
        "dom_text_equals": {"selector", "value"},
        "dom_exists": {"selector"},
        "hash_equals": {"value"},
        "answer_contains": {"value"},
        "answer_number": {"value"},
        "answer_set": {"pattern", "values"},
    }[str(kind)]
    missing = required - set(checker)
    if missing:
        raise ValueError(f"{kind} checker missing keys: {sorted(missing)}")
