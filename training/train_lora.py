"""QLoRA fine-tuning of a small pre-trained GUI model on the recorded dataset.

Usage::

    pip install -e ".[local]"
    python -m training.train_lora --dry-run          # validate data/local_model/ without loading a model
    python -m training.train_lora                    # ShowUI-2B, 4-bit, defaults below
    python -m training.train_lora --base bytedance-research/UI-TARS-2B-SFT --out models/adapters/uitars-2b-lora

Defaults are sized for a 6 GB laptop GPU (RTX 3050): base weights in 4 bits
(NF4), batch 1 with 8 accumulation steps, gradient checkpointing and ~768
visual tokens per screenshot. Only LoRA matrices on the language model's
attention and MLP projections are trained; the vision tower stays frozen.
The loss is computed on the answer tokens only (the prompt and the image are
masked). The prompt is built by :func:`src.local_model.actions.build_messages`,
the same function the agent uses at inference time.

Output: ``<out>/`` with the adapter weights, the processor and ``training_meta.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT
from src.local_model.actions import build_messages, parse_action

DEFAULT_BASE = "showlab/ShowUI-2B"
DEFAULT_PROCESSOR = "Qwen/Qwen2-VL-2B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
"""Projection names of the Qwen2 language model; the vision tower uses ``qkv``/``fc1``/``fc2``."""
IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    """Hyper-parameters (all overridable from the command line)."""

    base: str = DEFAULT_BASE
    processor: str = DEFAULT_PROCESSOR
    data_dir: str = str(REPO_ROOT / "data" / "local_model")
    out: str = str(REPO_ROOT / "models" / "adapters" / "showui-2b-lora")
    epochs: float = 2.0
    lr: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 1
    grad_accum: int = 8
    max_pixels: int = 768 * 28 * 28
    min_pixels: int = 256 * 28 * 28
    load_in_4bit: bool = True
    seed: int = 20261106
    max_steps: int = -1
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    warmup_ratio: float = 0.03


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL split (missing file -> empty list)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_records(records: list[dict[str, Any]], data_dir: Path) -> list[str]:
    """Problems that would break training: missing images, unparseable targets."""
    problems: list[str] = []
    for r in records:
        if not r.get("image") or not (data_dir / r["image"]).exists():
            problems.append(f"{r.get('id')}: missing image {r.get('image')}")
        _, error = parse_action(r["target"], int(r["width"]), int(r["height"]))
        if error:
            problems.append(f"{r.get('id')}: target does not round-trip ({error})")
    return problems


class SFTCollator:
    """Build a batch: chat template, image tensors and labels masked to the answer.

    The processor's tokenizer must pad on the right, so the prompt occupies the
    first positions of every row and can be masked by length.
    """

    def __init__(self, processor: Any, data_dir: Path) -> None:
        self.processor = processor
        self.data_dir = data_dir

    def _template(self, record: dict[str, Any], *, with_answer: bool) -> str:
        target = record["target"] if with_answer else None
        messages = build_messages(record["task"], list(record["history"]), target)
        text: str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not with_answer
        )
        return text

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenise a list of records into model inputs with ``labels``."""
        from PIL import Image

        images = [Image.open(self.data_dir / r["image"]).convert("RGB") for r in batch]
        full = [self._template(r, with_answer=True) for r in batch]
        prompts = [self._template(r, with_answer=False) for r in batch]
        enc = self.processor(text=full, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        for i, (prompt, image) in enumerate(zip(prompts, images, strict=True)):
            n_prompt = int(
                self.processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
            )
            labels[i, :n_prompt] = IGNORE_INDEX
        labels[enc["attention_mask"] == 0] = IGNORE_INDEX
        enc["labels"] = labels
        return dict(enc)


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def train(cfg: TrainConfig) -> Path:
    """Fine-tune and save the adapter; returns the output directory."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    data_dir = Path(cfg.data_dir)
    train_records = read_jsonl(data_dir / "train.jsonl")
    val_records = read_jsonl(data_dir / "val.jsonl")
    if not train_records:
        raise SystemExit(f"no training data in {data_dir} (run python -m training.generate_data first)")

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if cfg.load_in_4bit
        else None
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.base, torch_dtype=torch.bfloat16, quantization_config=quant, device_map="auto"
    )
    model.config.use_cache = False
    # no fp32 up-cast of the frozen weights (prepare_model_for_kbit_training would cost ~1 GB on 6 GB)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        cfg.processor, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels
    )
    processor.tokenizer.padding_side = "right"

    out = Path(cfg.out)
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if val_records else "no",
        eval_steps=cfg.eval_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_records,
        eval_dataset=val_records or None,
        data_collator=SFTCollator(processor, data_dir),
    )
    result = trainer.train()
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(cfg),
        "train_pairs": len(train_records),
        "val_pairs": len(val_records),
        "train_jsonl_sha256": _fingerprint(data_dir / "train.jsonl"),
        "manifest_sha256": _fingerprint(data_dir / "manifest.json"),
        "train_loss": float(result.training_loss),
        "global_steps": int(result.global_step),
        "eval": trainer.evaluate() if val_records else None,
    }
    (out / "training_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(flag, type=type(value), default=value)
    parser.add_argument(
        "--dry-run", action="store_true", help="only validate the dataset; no model is loaded"
    )
    args = vars(parser.parse_args())
    dry_run = args.pop("dry_run")
    cfg = TrainConfig(**args)
    data_dir = Path(cfg.data_dir)
    records = {name: read_jsonl(data_dir / f"{name}.jsonl") for name in ("train", "val", "test")}
    print({name: len(rows) for name, rows in records.items()})
    problems = validate_records(records["train"] + records["val"], data_dir)
    if problems:
        print(f"{len(problems)} problems, first ones:")
        for p in problems[:20]:
            print(" -", p)
        raise SystemExit(1)
    if dry_run:
        print("dataset OK")
        return
    print(f"adapter saved to {train(cfg)}")


if __name__ == "__main__":
    main()
