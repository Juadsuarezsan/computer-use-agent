"""Reasoner backed by a small pre-trained GUI model running locally (no API cost).

The default checkpoint is ShowUI-2B (Qwen2-VL-2B architecture, pre-trained for
GUI grounding and navigation); any Qwen2-VL / Qwen2.5-VL checkpoint works. A
LoRA adapter produced by ``training/train_lora.py`` is loaded on top when
configured. Prompt and action format come from :mod:`src.local_model.actions`,
the same module the training script uses.

Heavy dependencies (``torch``, ``transformers``, ``peft``, ``bitsandbytes``)
are imported lazily and live in the optional ``[local]`` extra, so the rest of
the project and its tests run without them.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.agent.protocols import Observation, ReasonerOutput
from src.api.schemas import Action
from src.local_model.actions import build_messages, encode_action, parse_action, png_size

DEFAULT_BASE_MODEL = "showlab/ShowUI-2B"
DEFAULT_PROCESSOR = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_MAX_PIXELS = 768 * 28 * 28
"""~768 visual tokens per screenshot: fits a 6 GB GPU with the 2B model in 4 bits."""
DEFAULT_MIN_PIXELS = 256 * 28 * 28


@dataclass
class Generation:
    """Raw output of one model call."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0


Backend = Callable[[list[dict[str, Any]], bytes], Generation]
"""``(messages, png) -> Generation``; injectable so tests need no model."""


class HFVisionBackend:
    """Hugging Face backend: Qwen2-VL family + optional LoRA adapter, optional 4-bit.

    Args:
        base_model: Hub id or local path of the base checkpoint.
        processor_model: Hub id or path of the processor (ShowUI reuses Qwen2-VL's).
        adapter: Path of a LoRA adapter (``""`` = base model only).
        load_in_4bit: Quantise the base weights with bitsandbytes NF4.
        max_pixels / min_pixels: Image budget of the processor.
        max_new_tokens: Generation budget (one JSON action is ~30 tokens).
    """

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        processor_model: str = DEFAULT_PROCESSOR,
        adapter: str = "",
        load_in_4bit: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_new_tokens: int = 64,
    ) -> None:
        self.base_model = base_model
        self.processor_model = processor_model
        self.adapter = adapter
        self.load_in_4bit = load_in_4bit
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError('local model needs the optional extra: pip install -e ".[local]"') from exc
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        logger.info(f"loading local model {self.base_model} (4bit={self.load_in_4bit})")
        model = AutoModelForVision2Seq.from_pretrained(self.base_model, **kwargs)
        if self.adapter:
            from peft import PeftModel

            logger.info(f"loading LoRA adapter {self.adapter}")
            model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.processor_model, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self._model = model
        self._torch = torch

    def __call__(self, messages: list[dict[str, Any]], png: bytes) -> Generation:
        """Run greedy decoding on one screenshot and return the new text."""
        self._load()
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[prompt], images=[image], return_tensors="pt").to(self._model.device)
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        n_in = int(inputs["input_ids"].shape[1])
        new_tokens = output[:, n_in:]
        text = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return Generation(text=text, tokens_in=n_in, tokens_out=int(new_tokens.shape[1]))


class LocalVLMReasoner:
    """Next-action policy from a local vision-language model.

    Args:
        backend: Model call (defaults to :class:`HFVisionBackend` with its defaults).
        history_len: Previous actions written into the prompt (must match training).
        display: Fallback screen size when an observation has no PNG.
        label: Name used in traces and eval rows.
    """

    name = "local"

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        history_len: int = 4,
        display: tuple[int, int] = (1024, 768),
        label: str = "local",
    ) -> None:
        self.backend: Backend = backend or HFVisionBackend()
        self.history_len = history_len
        self.display = display
        self.name = label
        self.history: list[str] = []
        self.feedback_notes: list[str] = []

    def reset(self, spec: dict[str, Any] | None = None) -> None:
        """Forget the action history."""
        self.history = []
        self.feedback_notes = []

    def feedback(self, note: str) -> None:
        """Keep verifier notes for the trace (the prompt format has no slot for them)."""
        self.feedback_notes.append(note)

    async def next_action(self, task: str, observation: Observation, step: int) -> ReasonerOutput:
        """Ask the model for the next action on the current screenshot."""
        if not observation.png:
            action = Action(type="task_complete", text="local model needs a screenshot")
            return ReasonerOutput(action=action, reasoning=f"step {step}: observation without PNG")
        width, height = png_size(observation.png)
        history = self.history[-self.history_len :] if self.history_len else []
        messages = build_messages(task, history)
        started = time.perf_counter()
        generation = self.backend(messages, observation.png)
        elapsed = int((time.perf_counter() - started) * 1000)
        action, error = parse_action(generation.text, width, height)
        if error is None and action.type != "screenshot":
            self.history.append(encode_action(action, width, height))
        reasoning = f"step {step}: {generation.text.strip()[:300]} ({elapsed} ms)"
        if error:
            reasoning += f" [parse error: {error}]"
        return ReasonerOutput(
            action=action,
            reasoning=reasoning,
            tokens_in=generation.tokens_in,
            tokens_out=generation.tokens_out,
            cost_usd=0.0,
        )
