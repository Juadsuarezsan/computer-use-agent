from __future__ import annotations

import json
import random

import pytest

from src.local_model.variants import (
    DEFAULT_HELDOUT,
    Perturbation,
    make_variant,
    random_perturbation,
    same_shape,
    split_of,
    substitute_values,
    typed_literals,
)


def _spec(tasks, task_id):
    return next(t for t in tasks if t.id == task_id).as_spec()


def test_same_shape_keeps_format():
    rng = random.Random(1)
    out = same_shape("a@b.com", rng)
    assert out != "a@b.com"
    assert len(out) == 7 and out[1] == "@" and out[3] == "."
    out = same_shape("Calle 45 #12-30", rng)
    assert out[0].isupper() and out[6:8].isdigit() and out[8:10] == " #"
    assert same_shape("--", rng) == "--"


def test_typed_literals_only_typed_values(tasks):
    assert typed_literals(_spec(tasks, "ff-01")) == ["Alice", "a@b.com"]
    # extraction tasks type nothing
    assert typed_literals(_spec(tasks, "de-05")) == []


def test_substitution_is_consistent(tasks):
    spec = _spec(tasks, "ff-01")
    variant, mapping = substitute_values(spec, random.Random(3))
    assert set(mapping) == {"Alice", "a@b.com"}
    dumped = json.dumps(variant)
    for old, new in mapping.items():
        assert old not in variant["task"]
        assert new in variant["task"]
        assert new in dumped
    typed = [s["text"] for s in variant["plan"] if s["action"] == "type"]
    assert typed == [mapping["Alice"], mapping["a@b.com"]]
    values = [c["value"] for c in variant["checker"]["checkers"]]
    assert values == [mapping["Alice"], mapping["a@b.com"]]
    assert spec["task"].count("Alice") == 1  # original untouched


def test_make_variant_deterministic_and_never_identity(tasks):
    spec = _spec(tasks, "ff-01")
    a = make_variant(spec, 3, seed=7)
    b = make_variant(spec, 3, seed=7)
    assert a == b
    assert make_variant(spec, 4, seed=7) != a
    assert not a.perturbation.is_identity
    visual = make_variant(spec, 3, seed=7, substitute=False)
    assert visual.substituted == {} and visual.spec["task"] == spec["task"]
    with pytest.raises(ValueError):
        make_variant(spec, 0, seed=7)


def test_perturbation_css():
    assert Perturbation().css() == "" and Perturbation().is_identity
    css = Perturbation(font_scale=120, padding=16, hue_rotate=90, brightness=1.1).css()
    assert "font-size: 120%" in css and "padding: 16px" in css
    assert "hue-rotate(90deg)" in css and "brightness(1.1)" in css
    rng = random.Random(0)
    assert all(not random_perturbation(rng).is_identity for _ in range(200))


def test_split():
    assert split_of("ff-05", DEFAULT_HELDOUT, set(), "forms.html") == "test"
    assert split_of("de-01", DEFAULT_HELDOUT, {"legacy.html"}, "legacy.html") == "test"
    assert split_of("ff-01", DEFAULT_HELDOUT, set(), "forms.html") == "train"
