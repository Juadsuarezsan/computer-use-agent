"""Small pre-trained GUI model fine-tuned with LoRA on trajectories of this sandbox.

* :mod:`src.local_model.actions` — action text format and prompt shared by
  training and inference (one source of truth, so both see the same text).
* :mod:`src.local_model.variants` — task variants (typed values and visual
  perturbations) and the train / held-out split.
* :mod:`src.local_model.recording` — runs the scripted oracle on the variants
  and records ``(screenshot, task, history) -> action`` pairs.
"""
