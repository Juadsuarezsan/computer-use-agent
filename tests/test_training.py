from __future__ import annotations

import json

import pytest

from src.api.schemas import Action
from src.local_model.actions import encode_action
from tests.test_local_actions import fake_png
from training.eval_steps import score_step, summarize
from training.train_lora import LORA_TARGETS, SFTCollator, TrainConfig, read_jsonl, validate_records


def _record(action: Action, box=None, **kw):
    return {
        "id": "x",
        "width": 1000,
        "height": 500,
        "target": encode_action(action, 1000, 500),
        "target_box": box,
        "action_type": action.type,
        **kw,
    }


def test_score_click_inside_box():
    rec = _record(Action(type="click", coords=(500, 250)), box=[0.45, 0.45, 0.55, 0.55])
    assert score_step(rec, Action(type="click", coords=(520, 260)))["step"] is True
    miss = score_step(rec, Action(type="click", coords=(100, 100)))
    assert miss["type"] is True and miss["click"] is False and miss["step"] is False
    wrong = score_step(rec, Action(type="type", text="a"))
    assert wrong["type"] is False and wrong["step"] is False


def test_score_text_actions():
    rec = _record(Action(type="type", text="Alice"))
    assert score_step(rec, Action(type="type", text="alice "))["step"] is True
    assert score_step(rec, Action(type="type", text="Bob"))["text"] is False
    rec = _record(Action(type="key", key="ctrl+a"))
    assert score_step(rec, Action(type="key", key="CTRL+A"))["step"] is True
    rec = _record(Action(type="wait"))
    assert score_step(rec, Action(type="wait"))["step"] is True


def test_summarize_excludes_not_applicable():
    scores = [
        {"type": True, "click": True, "text": None, "step": True},
        {"type": True, "click": None, "text": False, "step": False},
    ]
    s = summarize(scores)
    assert s == {"n": 2, "type_acc": 1.0, "click_acc": 1.0, "text_acc": 0.0, "step_acc": 0.5}
    assert summarize([])["type_acc"] is None


def test_read_and_validate(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(fake_png(10, 10))
    good = _record(Action(type="click", coords=(5, 5)), image="images/a.png")
    bad = _record(Action(type="click", coords=(5, 5)), image="images/missing.png", id="bad")
    bad["target"] = "not json"
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (good, bad)) + "\n", encoding="utf-8"
    )
    rows = read_jsonl(tmp_path / "train.jsonl")
    assert len(rows) == 2 and read_jsonl(tmp_path / "nope.jsonl") == []
    problems = validate_records(rows, tmp_path)
    assert len(problems) == 2 and all(p.startswith("bad") for p in problems)


def test_defaults_target_language_model_only():
    assert "qkv" not in LORA_TARGETS and "q_proj" in LORA_TARGETS
    cfg = TrainConfig()
    assert cfg.load_in_4bit and cfg.batch_size * cfg.grad_accum == 8


def test_collator_masks_prompt_tokens(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = " ".join(c.get("text", "<img>") for m in messages for c in m["content"])
            return text + (" <gen>" if add_generation_prompt else "")

        def __call__(self, text, images, return_tensors="pt", padding=False):
            ids = [[1] * len(t.split()) for t in text]
            width = max(len(i) for i in ids)
            input_ids = torch.tensor([i + [0] * (width - len(i)) for i in ids])
            mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids])
            return {"input_ids": input_ids, "attention_mask": mask}

    rec = {"task": "do it", "history": [], "target": '{"action": "WAIT"}', "image": "a.png"}
    long = dict(rec, target='{"action": "INPUT", "value": "a b c d"}')
    batch = SFTCollator(FakeProcessor(), tmp_path)([rec, long])
    labels = batch["labels"]
    n_prompt = len(FakeProcessor().apply_chat_template([], add_generation_prompt=True).split())
    assert (labels[:, :n_prompt] == -100).all()
    assert (labels[0][batch["attention_mask"][0] == 0] == -100).all()
    assert (labels != -100).any()
