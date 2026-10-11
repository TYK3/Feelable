"""Precision audit: ten FEATURED lessons - five basic, five complex, across different subjects - plus
an extended library; each with a learning objective, an annotation version, and reviewed corrections
that use only the teacher's own edit operations. (The full release gate is qa/check_gallery.py.)"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPS = {"rename", "describe", "remove", "merge", "outline", "draw", "title", "trace"}


def _manifest():
    return json.load(open(os.path.join(ROOT, "samples", "manifest.json")))


def test_ten_featured_five_basic_five_complex():
    m = _manifest()
    assert len({s["id"] for s in m}) == len(m)
    feat = [s for s in m if s.get("featured")]
    basic = {s["id"] for s in feat if s.get("group") == "simple"}
    complex_ = {s["id"] for s in feat if s.get("group") == "complex"}
    assert basic == {"flower", "earth-layers", "bar-chart", "water-cycle", "shapes"}
    assert complex_ == {"skull-lateral", "animal-cell", "mixed-circuit", "flowchart", "transit-map"}
    # the earlier examples stay available in the extended library
    for keep in ("plant-cell", "heart", "plant-cell-photo", "circuit", "volcano", "digestive-system",
                 "heart-internal", "human-organs-notebook"):
        assert any(s["id"] == keep and not s.get("featured") for s in m), keep
    assert all(s.get("annotation_version") for s in m)


def test_every_example_has_an_image_objective_and_valid_corrections():
    for s in _manifest():
        assert os.path.exists(os.path.join(ROOT, "samples", s["file"])), s["id"]
        assert s.get("objective"), s["id"]
        for op in s.get("review", []):
            assert len(OPS & set(op)) == 1, (s["id"], op)
            for key in ("polygon", "inside", "outside", "lasso"):
                for pt in op.get(key, []):
                    assert len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt), (s["id"], op)
