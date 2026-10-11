"""Handover: exactly 15 prepared examples (five complex systems + the ten earlier ones), each with
a learning objective, and reviewed corrections that use only the teacher's own edit operations.
(The full release gate, which builds every example, is qa/check_gallery.py.)"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPS = {"rename", "describe", "remove", "merge", "outline", "draw", "title"}


def _manifest():
    return json.load(open(os.path.join(ROOT, "samples", "manifest.json")))


def test_exactly_fifteen_examples_five_complex():
    m = _manifest()
    assert len(m) == 15
    assert len({s["id"] for s in m}) == 15
    complex_ = {s["id"] for s in m if s.get("group") == "complex"}
    assert complex_ == {"skull-lateral", "digestive-system", "heart-internal", "human-organs-notebook", "mixed-circuit"}
    for keep in ("plant-cell", "heart", "water-cycle", "plant-cell-photo", "animal-cell", "earth-layers",
                 "bar-chart", "circuit", "flower", "volcano"):
        assert any(s["id"] == keep for s in m), keep


def test_every_example_has_an_image_objective_and_valid_corrections():
    for s in _manifest():
        assert os.path.exists(os.path.join(ROOT, "samples", s["file"])), s["id"]
        assert s.get("objective"), s["id"]
        for op in s.get("review", []):
            assert len(OPS & set(op)) == 1, (s["id"], op)
            for key in ("polygon", "inside", "outside", "lasso"):
                for pt in op.get(key, []):
                    assert len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt), (s["id"], op)
