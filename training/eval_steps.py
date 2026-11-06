"""Offline step accuracy of the local model on recorded steps (no sandbox needed).

For every record of a split the model sees the screenshot, the task and the
recorded history, and its answer is compared with the oracle's action:

* ``type_acc`` — same action type;
* ``click_acc`` — for pointer actions, the predicted point falls inside the
  target element's bounding box (the usual GUI-grounding metric);
* ``text_acc`` — for INPUT / KEY / ANSWER, the same text (case-insensitive);
* ``step_acc`` — the step is fully right (type and, when it applies, point or text).

End-to-end task success is measured by ``python -m eval.run --with-local``.

Usage::

    python -m training.eval_steps --split test --adapter models/adapters/showui-2b-lora
    python -m training.eval_steps --split test --adapter ""      # base model, zero-shot
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from src.api.schemas import Action
from src.config import REPO_ROOT
from src.local_model.actions import build_messages, parse_action
from training.train_lora import DEFAULT_BASE, DEFAULT_PROCESSOR, read_jsonl

POINTER = {"click", "double_click", "right_click", "mouse_move"}
TEXTUAL = {"type", "key", "task_complete"}


def score_step(record: dict[str, Any], predicted: Action) -> dict[str, bool | None]:
    """Compare one prediction with the recorded action."""
    width, height = int(record["width"]), int(record["height"])
    expected, _ = parse_action(record["target"], width, height)
    same_type = predicted.type == expected.type
    click_ok: bool | None = None
    text_ok: bool | None = None
    if expected.type in POINTER:
        box = record.get("target_box")
        if box and predicted.coords is not None and same_type:
            x, y = predicted.coords[0] / width, predicted.coords[1] / height
            click_ok = box[0] <= x <= box[2] and box[1] <= y <= box[3]
        else:
            click_ok = False
    if expected.type in TEXTUAL:
        want = (expected.text or expected.key or "").strip().lower()
        got = (predicted.text or predicted.key or "").strip().lower()
        text_ok = same_type and want == got
    step_ok = same_type and click_ok is not False and text_ok is not False
    return {"type": same_type, "click": click_ok, "text": text_ok, "step": step_ok}


def summarize(scores: list[dict[str, bool | None]]) -> dict[str, Any]:
    """Accuracy per metric (``None`` entries are not applicable and excluded)."""

    def rate(key: str) -> float | None:
        values = [s[key] for s in scores if s[key] is not None]
        return round(sum(bool(v) for v in values) / len(values), 4) if values else None

    return {
        "n": len(scores),
        "type_acc": rate("type"),
        "click_acc": rate("click"),
        "text_acc": rate("text"),
        "step_acc": rate("step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "local_model"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--processor", default=DEFAULT_PROCESSOR)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N records")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    from src.agent.local_reasoner import HFVisionBackend

    data_dir = Path(args.data_dir)
    records = read_jsonl(data_dir / f"{args.split}.jsonl")
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"no records in {data_dir / (args.split + '.jsonl')}")
    backend = HFVisionBackend(
        base_model=args.base,
        processor_model=args.processor,
        adapter=args.adapter,
        load_in_4bit=not args.no_4bit,
    )
    scores: list[dict[str, bool | None]] = []
    rows: list[dict[str, Any]] = []
    for record in records:
        png = (data_dir / record["image"]).read_bytes()
        generation = backend(build_messages(record["task"], list(record["history"])), png)
        predicted, error = parse_action(generation.text, int(record["width"]), int(record["height"]))
        score = score_step(record, predicted)
        scores.append(score)
        rows.append({"id": record["id"], "output": generation.text, "parse_error": error, **score})
    by_type: dict[str, list[dict[str, bool | None]]] = {}
    for record, score in zip(records, scores, strict=True):
        by_type.setdefault(str(record["action_type"]), []).append(score)
    name = "lora" if args.adapter else "base"
    report = {
        "name": f"local-steps-{name}-{args.split}",
        "base": args.base,
        "adapter": args.adapter or None,
        "split": args.split,
        "overall": summarize(scores),
        "by_action_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "rows": rows,
    }
    out = REPO_ROOT / "eval" / "runs" / f"{date.today().isoformat()}-{report['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
