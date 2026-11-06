# Small local model: fine-tuning a pre-trained GUI model on this sandbox

Status: **built and unit-tested, not run.** No dataset, adapter or local-model number exists in
this repository yet; `eval/RESULTS.md` shows local rows only after a real run.

## Goal

Replace the per-step Claude call (screenshot in, action out) by a small pre-trained model running
on the operator's GPU, fine-tuned cheaply on this sandbox, so that the cost per task drops to the
electricity of a local GPU while the loop, the safety layer and the verifier stay unchanged.

## Model

| | Default | Alternative |
|---|---|---|
| Checkpoint | `showlab/ShowUI-2B` | `bytedance-research/UI-TARS-2B-SFT`, `Qwen/Qwen2.5-VL-3B-Instruct` |
| Architecture | Qwen2-VL-2B (vision encoder + 1.5B language model) | same family |
| Pre-training | GUI grounding and navigation (web, mobile, desktop screenshots) | GUI agents / general VLM |
| Processor | `Qwen/Qwen2-VL-2B-Instruct` | the checkpoint's own |

Licences must be checked on each model card before any commercial use.

The output format (`src/local_model/actions.py`) is one JSON object per step, close to ShowUI's
native one (`{"action": "CLICK", "value": null, "position": [x, y]}`, relative coordinates), so the
zero-shot base model is not asked for a syntax it has never seen.

## Data: recorded from the scripted oracle

`python -m training.generate_data` (`src/local_model/recording.py`):

1. For each of the 20 tasks, 25 variants (default) are built (`src/local_model/variants.py`):
   * typed values (names, e-mails, subjects ...) replaced by random strings of the same shape,
     consistently in the task text, the plan and the checker;
   * visual perturbation: viewport (5 sizes), root font size (90–120 %), page padding (0–32 px),
     hue rotation and brightness. The original task (1024×768, no CSS, original values) is never a
     variant: it remains the evaluation instance.
2. The `StubReasoner` runs each variant in the Playwright sandbox; it resolves selectors to the
   current pixels, so its actions are correct labels for the perturbed screen.
3. A trajectory is kept only if the variant's ground-truth checker passes. If a substituted value
   breaks the task (a value fixed by the app, e.g. the `demo/demo` login), the variant is retried
   with the original values and only the visual perturbation; failures are listed in
   `manifest.json`.
4. Every step becomes a record: screenshot, task, last 4 actions, target action, bounding box of
   the clicked element.

Split: `ff-05`, `de-05`, `wn-05`, `ms-05` (one task per app) are **never trained on** (`test`);
10 % of the other trajectories go to `val`. `--heldout-apps legacy.html` also holds out a whole app.

Expected size with the defaults: ~500 trajectories, ~3 000 pairs (mean 6.5 steps per task). This
is a rule of thumb for LoRA domain adaptation, not a measured optimum.

## Training: QLoRA

`python -m training.train_lora` — base weights in 4 bits (NF4), LoRA r=16 / α=32 on the language
model's `q,k,v,o,gate,up,down` projections, vision tower frozen, loss on the answer tokens only,
batch 1 × 8 accumulation, 2 epochs, lr 1e-4 cosine, gradient checkpointing, ~768 visual tokens per
screenshot. `--dry-run` validates the dataset without loading a model.

Hardware budget (estimated, not measured): the 2B model in 4 bits takes ~1.5 GB; with gradient
checkpointing and ~1 000 tokens per sample it is expected to fit a 6 GB laptop GPU (RTX 3050).
If it does not, lower `--max-pixels` or rent a 16–24 GB GPU for the 2–3 hours of training.

## Evaluation

* `python -m training.eval_steps --split test` — offline step accuracy on recorded steps: action
  type, click inside the target box, text match. Base model with `--adapter ""`.
* `python -m eval.run --with-local --adapter <dir>` — end-to-end success on the 20 original
  instances with the real checkers: base model, LoRA model with verifier, LoRA without verifier,
  with the 4 held-out tasks reported apart. Cost per task is $0 (no API call); the verifier is
  the heuristic one.
* `REASONER_MODE=cascade` with a key: Claude acts only after 2 consecutive verifier failures.

## Risks and open points

* 20 tasks on four fixture apps is little data. The adapter may learn these apps and not
  generalise; the held-out tasks and `--heldout-apps` measure that, they do not prevent it.
* Copying the scripted oracle is behaviour cloning: the model never sees recovery from its own
  mistakes. The verifier and the cascade are the mitigation, not a fix.
* Accuracy against Claude Sonnet is unknown until both are run on the same tasks; the Claude rows
  need a paid API key and stay pending by default.
* Fitting in 6 GB is an estimate.
* The base model's zero-shot row is expected to be weak because the prompt format is new to it;
  it measures what the fine-tuning adds, not the quality of ShowUI.
