"""Assisted missing-part outline (handover "Required feature") and safe add_part overlaps."""
import cv2
import numpy as np

from engine import grounding
from tests.test_api_v2 import _create, _patch


def _scene():
    img = np.full((400, 600, 3), 250, np.uint8)
    cv2.ellipse(img, (300, 200), (120, 70), 0, 0, 360, (200, 80, 80), -1)     # the "pancreas"
    cv2.ellipse(img, (300, 200), (120, 70), 0, 0, 360, (40, 40, 40), 2)
    cv2.circle(img, (520, 330), 40, (80, 160, 80), -1)                        # a neighbour
    return img


def test_one_tap_proposes_the_structure_under_it():
    m = grounding.propose_outline(_scene(), inside=[(300, 200)], outside=[])
    assert m is not None
    truth = np.zeros((400, 600), np.uint8)
    cv2.ellipse(truth, (300, 200), (120, 70), 0, 0, 360, 255, -1)
    inter = ((m > 0) & (truth > 0)).sum()
    union = ((m > 0) | (truth > 0)).sum()
    assert inter / union > 0.8                          # the outline follows the drawn edge
    assert not m[330, 520]                              # and does not take the neighbour


def test_loose_lasso_is_refined_to_the_edge():
    lasso = [(160, 110), (440, 110), (450, 290), (150, 290)]
    m = grounding.propose_outline(_scene(), inside=[], outside=[], lasso=lasso)
    assert m is not None and m[200, 300] and not m[115, 165]


def test_outline_endpoint_echoes_revision_and_never_saves(client):
    job, tok = _create(client)
    jid = job["id"]
    w, h = job["source"]["width"], job["source"]["height"]
    r = client.post(f"/api/jobs/{jid}/outline", json={"rev": 7, "inside": [[w * 0.33, h * 0.5]]},
                    headers={"X-Edit-Token": tok})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rev"] == 7 and "overlaps" in body
    assert client.get(f"/api/jobs/{jid}").json()["version"] == job["version"]   # nothing saved
    # polygon only: overlap report
    poly = [[w * 0.2, h * 0.3], [w * 0.45, h * 0.3], [w * 0.45, h * 0.7], [w * 0.2, h * 0.7]]
    r = client.post(f"/api/jobs/{jid}/outline", json={"rev": 8, "polygon": poly}, headers={"X-Edit-Token": tok})
    names = {o["name"] for o in r.json()["overlaps"]}
    assert "nucleus" in names
    # view-only callers are refused
    assert client.post(f"/api/jobs/{jid}/outline", json={"inside": [[5, 5]]}).status_code in (401, 403)
    assert client.post(f"/api/jobs/{jid}/outline", json={"rev": 1}, headers={"X-Edit-Token": tok}).status_code == 400


def _nucleus_box(job):
    nuc = next(p for p in job["parts"] if p["name"] == "nucleus")
    x0, y0, x1, y1 = nuc["source_bbox"]
    pad = 4
    return nuc["id"], [[x0 - pad, y0 - pad], [x1 + pad, y0 - pad], [x1 + pad, y1 + pad], [x0 - pad, y1 + pad]]


def test_covering_a_whole_part_needs_an_explicit_choice(client):
    job, tok = _create(client)
    jid = job["id"]
    nid, poly = _nucleus_box(job)
    r = _patch(client, jid, tok, {"add_part": {"name": "big nucleus", "polygon": poly}})
    assert r.status_code == 400 and "Replace it" in r.json()["detail"]
    r = _patch(client, jid, tok, {"add_part": {"name": "big nucleus", "polygon": poly, "overlap": "keep"}})
    assert r.status_code == 200, r.text
    parts = {p["id"]: p for p in r.json()["parts"]}
    assert parts[nid]["include"]                                     # kept, nested
    r = _patch(client, jid, tok, {"add_part": {"name": "new nucleus", "polygon": poly, "overlap": "replace"},
                                  "base_version": r.json()["version"]})
    assert r.status_code == 200, r.text
    assert not {p["id"]: p for p in r.json()["parts"]}[nid]["include"]   # switched off by choice


def test_extend_and_trim_an_existing_part(client):
    job, tok = _create(client)
    jid = job["id"]
    nid, _ = _nucleus_box(job)
    nuc = next(p for p in job["parts"] if p["id"] == nid)
    x0, y0, x1, y1 = nuc["source_bbox"]
    extra = [[x1 - 2, y0], [x1 + 30, y0], [x1 + 30, y1], [x1 - 2, y1]]
    r = _patch(client, jid, tok, {"add_part": {"name": "", "polygon": extra, "into": nid, "overlap": "keep"}})
    assert r.status_code == 200, r.text
    grown = next(p for p in r.json()["parts"] if p["id"] == nid)
    assert grown["source_bbox"][2] > x1 + 20
    assert len(r.json()["parts"]) == len(job["parts"])               # no duplicate part
    r = _patch(client, jid, tok, {"add_part": {"name": "", "polygon": extra, "into": nid, "mode": "remove"},
                                  "base_version": r.json()["version"]})
    assert r.status_code == 200, r.text
    trimmed = next(p for p in r.json()["parts"] if p["id"] == nid)
    assert trimmed["source_bbox"][2] < x1 + 5


def test_closeup_target_needs_a_real_pointer():
    """F08: a pointer stub (stopped at a bracket) gives no target box, so the close-up asks the
    teacher to choose the structure instead of pretending it found it."""
    from engine import pipeline
    from tests.conftest import fake_analysis
    a = fake_analysis()
    a.dropped = ["pancreas (too small to feel)", "spleen (too small to feel)"]
    a.labels += [type(a.labels[0])("pancreas", (250, 20, 290, 30), 0.9, None, "part_label"),
                 type(a.labels[0])("spleen", (250, 40, 290, 50), 0.9, None, "part_label")]
    a.trace = {"leaders": [{"label": "pancreas", "points": [[248, 25], [120, 150]]},
                           {"label": "spleen", "points": [[248, 45], [245, 46]]}]}
    rd = pipeline.build_readiness(a, [], False, None)
    by = {i.get("target"): i for i in rd["issues"] if i.get("action") == "draw"}
    assert by["pancreas"]["target_bbox"] is not None
    x0, y0, x1, y1 = by["pancreas"]["target_bbox"]
    assert x0 <= 120 <= x1 and y0 <= 150 <= y1
    assert by["spleen"]["target_bbox"] is None
