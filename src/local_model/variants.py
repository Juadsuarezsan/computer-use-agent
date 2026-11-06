"""Task variants for the training data and the train / held-out split.

A variant changes what the model sees without changing what the task means:

* **Typed values** — every quoted literal of the task that the scripted plan
  types (names, e-mails, subjects ...) is replaced by a random string with the
  same shape (letters by letters of the same case, digits by digits,
  punctuation kept), consistently in the task text, the plan and the checker.
  Values fixed by the application (logins, menu options) fail the checker and
  the recorder falls back to a visual-only variant, so no wrong label is kept.
* **Visual perturbation** — viewport size, root font size, page padding and
  colour filters. Element coordinates are re-read from the DOM at every step,
  so the labels follow the layout.

The original task (default viewport, no CSS, original values) is never a
variant: it stays as the evaluation instance.
"""

from __future__ import annotations

import copy
import random
import re
import string
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_HELDOUT: tuple[str, ...] = ("ff-05", "de-05", "wn-05", "ms-05")
"""One task per category never used for training (generalisation test)."""

DEFAULT_VARIANTS_PER_TASK = 25
"""25 variants x 20 tasks ~ 500 trajectories ~ 3 000 (screenshot, action) pairs."""

VIEWPORTS: tuple[tuple[int, int], ...] = ((1024, 768), (1280, 800), (1366, 768), (1152, 864), (960, 720))
FONT_SCALES: tuple[int, ...] = (90, 100, 110, 120)
PADDINGS: tuple[int, ...] = (0, 8, 16, 32)
HUE_ROTATIONS: tuple[int, ...] = (0, 30, 90, 180)
BRIGHTNESS: tuple[float, ...] = (0.9, 1.0, 1.1)

_QUOTED_RE = re.compile(r"'([^']{3,})'")
MIN_LITERAL_LENGTH = 3


@dataclass(frozen=True)
class Perturbation:
    """Visual change applied to the sandbox page after each reset."""

    width: int = 1024
    height: int = 768
    font_scale: int = 100
    padding: int = 0
    hue_rotate: int = 0
    brightness: float = 1.0

    @property
    def is_identity(self) -> bool:
        """``True`` when the page looks exactly like the evaluation instance."""
        return self == Perturbation()

    def css(self) -> str:
        """CSS injected into the page (empty for the identity)."""
        rules: list[str] = []
        if self.font_scale != 100:
            rules.append(f"html {{ font-size: {self.font_scale}% !important; }}")
        if self.padding:
            rules.append(f"body {{ padding: {self.padding}px !important; }}")
        filters: list[str] = []
        if self.hue_rotate:
            filters.append(f"hue-rotate({self.hue_rotate}deg)")
        if self.brightness != 1.0:
            filters.append(f"brightness({self.brightness})")
        if filters:
            rules.append(f"html {{ filter: {' '.join(filters)} !important; }}")
        return "\n".join(rules)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for the manifest."""
        return asdict(self)


def random_perturbation(rng: random.Random) -> Perturbation:
    """Draw a non-identity perturbation."""
    while True:
        width, height = rng.choice(VIEWPORTS)
        p = Perturbation(
            width=width,
            height=height,
            font_scale=rng.choice(FONT_SCALES),
            padding=rng.choice(PADDINGS),
            hue_rotate=rng.choice(HUE_ROTATIONS),
            brightness=rng.choice(BRIGHTNESS),
        )
        if not p.is_identity:
            return p


def same_shape(value: str, rng: random.Random) -> str:
    """Random string with the shape of ``value`` (case, digits and punctuation kept).

    Accented letters are replaced by ASCII letters of the same case. The result
    always differs from ``value`` when it contains at least one letter or digit.
    """
    if not any(c.isalnum() for c in value):
        return value
    while True:
        out = []
        for c in value:
            if c.isdigit():
                out.append(rng.choice(string.digits))
            elif c.isalpha():
                out.append(rng.choice(string.ascii_uppercase if c.isupper() else string.ascii_lowercase))
            else:
                out.append(c)
        candidate = "".join(out)
        if candidate != value:
            return candidate


def typed_literals(spec: dict[str, Any]) -> list[str]:
    """Quoted literals of the task text that the scripted plan types verbatim."""
    typed = {str(step.get("text", "")) for step in spec.get("plan", []) if step.get("action") == "type"}
    found: list[str] = []
    for literal in _QUOTED_RE.findall(str(spec.get("task", ""))):
        if len(literal) >= MIN_LITERAL_LENGTH and literal in typed and literal not in found:
            found.append(literal)
    return found


def _replace_all(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, str):
        for old, new in mapping.items():
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, list):
        return [_replace_all(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_all(v, mapping) for k, v in obj.items()}
    return obj


def substitute_values(spec: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], dict[str, str]]:
    """Return a copy of ``spec`` with every typed literal replaced, and the mapping used.

    Only the task text, the plan and the checker change; ids, app and route stay.
    Longer literals are replaced first so a literal contained in another one is
    not rewritten twice.
    """
    literals = sorted(typed_literals(spec), key=len, reverse=True)
    mapping = {lit: same_shape(lit, rng) for lit in literals}
    variant = copy.deepcopy(spec)
    for key in ("task", "plan", "checker"):
        if key in variant:
            variant[key] = _replace_all(variant[key], mapping)
    return variant, mapping


@dataclass(frozen=True)
class VariantSpec:
    """One variant to record: the modified task spec, its perturbation and provenance."""

    task_id: str
    index: int
    spec: dict[str, Any]
    perturbation: Perturbation
    substituted: dict[str, str]


def make_variant(spec: dict[str, Any], index: int, seed: int, *, substitute: bool = True) -> VariantSpec:
    """Build the ``index``-th variant of a task (deterministic for a given seed).

    Args:
        spec: Original task spec (``TaskSpec.as_spec()``).
        index: Variant number (1-based; 0 is reserved for the original task).
        seed: Global seed of the dataset.
        substitute: ``False`` keeps the typed values (visual-only variant).
    """
    if index < 1:
        raise ValueError("variant 0 is the evaluation instance and is never recorded")
    rng = random.Random(f"{seed}:{spec['id']}:{index}")
    perturbation = random_perturbation(rng)
    if substitute:
        variant, mapping = substitute_values(spec, rng)
    else:
        variant, mapping = copy.deepcopy(spec), {}
    variant["id"] = spec["id"]
    return VariantSpec(
        task_id=str(spec["id"]),
        index=index,
        spec=variant,
        perturbation=perturbation,
        substituted=mapping,
    )


def split_of(task_id: str, heldout: tuple[str, ...] | list[str], heldout_apps: set[str], app: str) -> str:
    """``test`` for held-out tasks or apps, ``train`` otherwise."""
    if task_id in heldout or app in heldout_apps:
        return "test"
    return "train"
