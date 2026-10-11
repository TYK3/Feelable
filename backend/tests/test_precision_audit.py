"""Precision audit (11 Oct 2026) regressions: typed targets, label->geometry association, and
the hand-annotated target scores in qa/precision. Scores are checked against annotations written
from the source images, never from Feelable output."""
import os
import sys

import cv2
import numpy as np
import pytest

from engine import grounding, pipeline
from engine.types import Part

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QA = os.path.join(ROOT, "qa")
sys.path.insert(0, QA)


def _part(pid, name, mask, **kw):
    return Part(id=pid, name=name, description="", key="", color=(200, 100, 100), mask=mask,
                polygons=[], bbox=(0, 0, 1, 1), area=int((mask > 0).sum()), **kw)


# ---------------------------------------------------------------- code risks in grounding.py
def test_largest_takes_the_near_piece_not_the_big_far_one():
    m = np.zeros((200, 200), np.uint8)
    m[150:195, 150:195] = 255          # big, far away
    m[20:30, 20:30] = 255              # small, next to the seed
    out = grounding._largest(m, 33, 25)
    assert out[25, 25] and not out[170, 170]


def test_largest_returns_nothing_when_no_piece_is_near():
    m = np.zeros((200, 200), np.uint8)
    m[150:195, 150:195] = 255
    m[0:5, 190:200] = 255
    assert not grounding._largest(m, 20, 20).any()


def test_support_without_evidence_is_not_strong_evidence():
    grad = np.zeros((60, 60), np.float32)               # a flat crop: no edges anywhere
    piece = np.zeros((60, 60), bool)
    piece[20:40, 20:40] = True
    region = np.zeros((60, 60), bool)
    region[20:40, 20:40] = True                         # bounded only by the region's own outline
    assert grounding._support(grad, piece, region) == 0.0


# ---------------------------------------------------------------- typed targets
def test_boundary_is_traced_along_the_shared_border():
    a = np.zeros((200, 300), np.uint8)
    b = np.zeros((200, 300), np.uint8)
    a[20:180, 20:150] = 255
    b[20:180, 150:280] = 255
    parts = [_part("p1", "left bone", a), _part("p2", "right bone", b)]
    m = grounding.trace_boundary(parts, 150, 40, reach=12, min_len=40)
    assert m is not None
    ys, xs = np.nonzero(m)
    assert ys.max() - ys.min() > 120                  # the whole suture, not a spot at the tip
    assert xs.max() - xs.min() < 15                   # a thin line, not a region


def test_boundary_needs_two_regions_meeting():
    a = np.zeros((200, 300), np.uint8)
    a[20:180, 20:150] = 255
    assert grounding.trace_boundary([_part("p1", "bone", a)], 150, 40, reach=12, min_len=40) is None


def test_mirror_twin_joins_a_plural_name():
    img = np.full((300, 400, 3), 245, np.uint8)
    cv2.ellipse(img, (120, 150), (25, 40), 0, 0, 360, (150, 80, 60), -1)
    cv2.ellipse(img, (280, 150), (25, 40), 0, 0, 360, (150, 80, 60), -1)
    left = np.zeros((300, 400), np.uint8)
    cv2.ellipse(left, (120, 150), (25, 40), 0, 0, 360, 255, -1)
    right = np.zeros((300, 400), np.uint8)
    cv2.ellipse(right, (280, 150), (25, 40), 0, 0, 360, 255, -1)
    k = _part("p1", "kidneys", left, label_text="Kidneys", match_method="leader")
    u = _part("p2", "part 2", right, match_method="none")
    parts = [k, u]
    n = grounding.recover_siblings(img, parts, lambda t: t.lower().endswith("s"))
    assert n == 1 and k.mask[150, 280] and k.count == 2


def test_subpart_names():
    assert grounding.is_subpart("mastoid portion") and grounding.is_subpart("zygomatic process")
    assert not grounding.is_subpart("temporal bone") and not grounding.is_subpart("kidneys")
    assert grounding._shares_word("squamous temporal", "temporal bone")
    assert not grounding._shares_word("greater wing of sphenoid bone", "parietal bone")


# ---------------------------------------------------------------- readiness
class _A:
    def __init__(self, title, parts):
        self.title, self.parts, self.labels, self.dropped = title, parts, [], []
        self.unmatched_labels, self.warnings, self.input_kind = [], [], "diagram"
        self.width, self.height, self.trace = 100, 100, {}


def test_unsupported_title_blocks_publication():
    m = np.zeros((100, 100), np.uint8)
    m[10:50, 10:50] = 255
    a = _A("H₂O café 心脏", [_part("p1", "water", m)])
    rd = pipeline.build_readiness(a, [], False, None)
    blk = [i for i in rd["issues"] if i["code"] == "braille_unsupported_title"]
    assert blk and blk[0]["severity"] == "block" and blk[0]["action"] == "retitle"
    assert rd["status"] == "blocked"
    a.title = "H2O cafe"
    rd = pipeline.build_readiness(a, [], False, None)
    assert not [i for i in rd["issues"] if i["code"] == "braille_unsupported_title"]


def test_unresolved_extent_is_reported_not_counted_as_done():
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (50, 50), 5, 255, -1)
    p = _part("m1", "coronal suture", m, target_type="boundary", extent="unresolved", match_method="landmark")
    rd = pipeline.build_readiness(_A("skull", [p]), [], False, None)
    it = [i for i in rd["issues"] if i["code"] == "extent_unresolved"]
    assert it and "location found" in it[0]["message"] and it[0]["action"] == "outline"


# ---------------------------------------------------------------- annotated end-to-end targets
CASES = [  # (input, minimum valid targets, maximum critical errors)
    ("flowchart-control.png", 12, 0),
    ("transit-control.png", 15, 0),
    ("water-cycle-fresh.jpg", 8, 0),
    ("geometry-control.png", 4, 0),
    ("mixed-circuit-rotate.jpg", 3, 0),
    ("human-organs-notebook-fresh.jpg", 6, 0),
    ("skull-lateral-fresh.jpg", 17, 2),
]


@pytest.mark.parametrize("name,min_valid,max_crit", CASES)
def test_annotated_targets(name, min_valid, max_crit):
    from engine import analyze as A
    from target_score import load_truth, score
    path = os.path.join(QA, "precision", "inputs", name)
    img = A.load_image(open(path, "rb").read())
    if isinstance(img, tuple):
        img = img[0]
    a = A.analyze(img, use_ai=False)
    t = load_truth(os.path.join(QA, "precision", "truth"), os.path.splitext(name)[0])
    r = score(a, t)
    assert r["valid"] >= min_valid and r["critical"] <= max_crit, r
